from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
import cupy as cp
from cupy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from optimizers import *
import warnings
from math import isclose
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from numba import cuda, float32, int32

# Initialize warnings
warnings.filterwarnings("ignore")

import multiprocessing

multiprocessing.set_start_method('spawn')

class AEW:
    def __init__(self, data, gamma_init=None):
        # DATA HOLDING OBJECTS
        self.data = cp.array(data.values)  # Convert data to CuPy array (GPU array)
        self.gamma = self.gamma_initializer(gamma_init)
        self.similarity_matrix = None

    def generate_graphs(self, num_neighbors, mode='distance', metric='euclidean'):
        # Generate a sparse k-neighbors graph using CPU, but the rest will be GPU accelerated
        graph = kneighbors_graph(self.data.get(), n_neighbors=num_neighbors, mode=mode, metric=metric, p=2, include_self=True, n_jobs=-1)
        self.similarity_matrix = self.correct_similarity_matrix_diag(csr_matrix(graph))

    def correct_similarity_matrix_diag(self, similarity_matrix):
        # Convert to sparse matrix and adjust diagonal
        if not isinstance(similarity_matrix, csr_matrix):
            similarity_matrix = csr_matrix(similarity_matrix)

        identity_diag_res = cp.ones(similarity_matrix.shape[0]) + 1  # Diagonal correction value
        similarity_matrix.setdiag(identity_diag_res)  # Set the diagonal directly for sparse matrix
        return similarity_matrix

    def gamma_initializer(self, gamma_init=None):
        # Initialize gamma based on the provided method
        if gamma_init is None:
            return cp.ones(self.data.shape[1])  # Default to ones if not provided
        elif gamma_init == 'var':
            return cp.var(self.data, axis=0)  # Variance-based initialization
        elif gamma_init == 'random_int':
            return cp.random.randint(0, 1000, (1, self.data.shape[1]))
        elif gamma_init == 'random_float':
            return cp.random.random(size=(1, self.data.shape[1]))

    def similarity_function(self, pt1_idx, pt2_idx, gamma):
        """GPU-optimized function to calculate similarity between two points."""
        point1 = self.data[pt1_idx]
        point2 = self.data[pt2_idx]

        deg_pt1 = cp.sum(point1)  # Calculate degree
        deg_pt2 = cp.sum(point2)  # Calculate degree

        similarity_measure = cp.sum(cp.where(cp.abs(gamma) > 1e-5, (((point1 - point2) ** 2) / (gamma ** 2)), 0))
        similarity_measure = cp.exp(-similarity_measure, dtype=cp.float64)  # Use float64 for precision

        degree_normalization_term = cp.sqrt(cp.abs(deg_pt1 * deg_pt2))

        if degree_normalization_term != 0 and not abs(degree_normalization_term - 0) <= max(1e-09 * max(abs(degree_normalization_term), abs(0)), 0.0):
            result = similarity_measure / degree_normalization_term
        else:
            result = 0

        return result

    def edge_weight_computation(self, section, gamma):
        """Compute edge weights for the given section using GPU-based matrix operations."""
        print("Computing Edge Weights")

        # Create a matrix of pairwise differences between all points in section and all other points
        points = self.data[section]
        diffs = cp.expand_dims(points, axis=1) - cp.expand_dims(self.data, axis=0)  # Shape: (len(section), num_points, features)
        
        # Calculate squared differences, normalize by gamma, and compute similarity
        squared_diffs = (diffs ** 2) / (gamma ** 2)
        similarity_matrix = cp.exp(-cp.sum(squared_diffs, axis=2))  # Sum over features and apply exp

        # Normalize by degree terms to compute edge weights
        degrees = cp.sum(similarity_matrix, axis=1)

        # To handle division by zero, use cp.nan_to_num to replace any NaN or Inf values that result from division by zero
        normalized_similarity = similarity_matrix / cp.expand_dims(degrees, axis=1)

        # Replace NaN or Inf values that may have been introduced by division by zero with zero
        normalized_similarity = cp.nan_to_num(normalized_similarity, nan=0.0, posinf=0.0, neginf=0.0)

        print("Completed Edge Weights Computation")
        return normalized_similarity

    def optimized_edge_weight_update(self, edge_weight_res, curr_sim_matr):
        """Parallelize the collection of edge weights and update the sparse matrix on GPU."""
        print("Updating Edge Weights")

        # Parallelize the collection of non-zero entries
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(collect_non_zero_entries, edge_weight_res)

        # Flatten results
        row_indices = []
        col_indices = []
        values = []

        for r, c, v in results:
            row_indices.extend(r)
            col_indices.extend(c)
            values.extend(v)

        # Create a sparse matrix and perform the update on GPU
        row_indices = cp.array(row_indices)
        col_indices = cp.array(col_indices)
        values = cp.array(values)

        # Efficient update to the current similarity matrix
        new_sim_matr = csr_matrix((values, (row_indices, col_indices)), shape=curr_sim_matr.shape)
        curr_sim_matr += new_sim_matr

        print("Edge Weights Updated")
        return curr_sim_matr

    def gradient_computation(self, section, similarity_matrix, gamma):
        """Compute the gradient function using GPU-based operations."""
        print("Computing Gradient")
        result = cp.zeros(len(section), dtype=cp.float32)

        for idx in section:
            dii = cp.sum(similarity_matrix[idx].toarray())  # Ensure sparse
            xi_reconstruction = cp.sum([similarity_matrix[idx, y] * self.data[y] for y in range(similarity_matrix.shape[1]) if idx != y], 0)

            if dii != 0 and not isclose(dii, 0, abs_tol=1e-100):
                xi_reconstruction = xi_reconstruction / dii
                first_term = (self.data[idx] - xi_reconstruction) / dii
            else:
                first_term = cp.zeros_like(xi_reconstruction)

            cubed_gamma = cp.where(cp.abs(gamma) > 1e-7, gamma ** (-3), 0)
            dw_dgamma = cp.sum([2 * similarity_matrix[idx, y] * (((self.data[idx] - self.data[y]) ** 2) * cubed_gamma) * self.data[y] for y in range(self.data.shape[0]) if idx != y])
            dD_dgamma = cp.sum([2 * similarity_matrix[idx, y] * (((self.data[idx] - self.data[y]) ** 2) * cubed_gamma) * xi_reconstruction for y in range(self.data.shape[0]) if idx != y])

            result[idx] = first_term * (dw_dgamma - dD_dgamma)

        print("Gradient Computation Complete")
        return cp.sum(result)

    def split(self, a, n):
        # Efficiently split data into n parts
        k, m = divmod(len(a), n)
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    def generate_optimal_edge_weights(self):
        print("Generating Optimal Edge Weights")
        self.similarity_matrix = self.generate_edge_weights(self.gamma)

        sba = SwarmBasedAnnealingOptimizer(self.similarity_matrix, self.generate_edge_weights, self.objective_function, self.gradient_function, self.gamma, 5, len(self.gamma), 10)
        self.gamma, _, _, _ = sba.optimize()

        self.similarity_matrix = self.generate_edge_weights(self.gamma)

    def generate_edge_weights(self, gamma):
        print("Generating Edge Weights")
        curr_sim_matr = csr_matrix(np.zeros_like(self.similarity_matrix.toarray()))

        # Removed parallelization; iterating sequentially over sections
        split_data = self.split(range(self.data.shape[0]), cpu_count())

        edge_weight_res = []
        for section in split_data:
            edge_weight_res.append(self.edge_weight_computation(section, gamma))

        # Efficiently update the sparse matrix on GPU
        curr_sim_matr = self.optimized_edge_weight_update(edge_weight_res, curr_sim_matr)
        curr_sim_matr = self.subtract_identity(curr_sim_matr)

        print("Edge Weight Generation Complete")
        return curr_sim_matr

    def subtract_identity(self, adj_matrix):
        """Subtract identity matrix from the adjacency matrix."""
        if isinstance(adj_matrix, csr_matrix):
            identity_diag_res = cp.ones(adj_matrix.shape[0]) + 2  # Adjust for identity subtraction
            adj_matrix.setdiag(identity_diag_res)
            return csr_matrix(cp.identity(adj_matrix.shape[0])) - adj_matrix
        else:
            identity_diag_res = cp.ones(len(adj_matrix)) + 2
            np.fill_diagonal(adj_matrix, identity_diag_res)
            return np.identity(len(adj_matrix)) - adj_matrix

def collect_non_zero_entries(section):
    """Collect non-zero entries from a section of the edge weight results."""
    cp.cuda.Device(0).use()
    row_indices = []
    col_indices = []
    values = []
    for idx1, idx2 in zip(*cp.where(section != 0)):  # Find non-zero entries
        row_indices.append(idx1)
        col_indices.append(idx2)
        values.append(section[idx1, idx2])
    return row_indices, col_indices, values
