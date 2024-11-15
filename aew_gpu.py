import numpy as np
import pandas as pd
import cupy as cp
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from cuml.sparse import csr_matrix as cu_csr_matrix
from cuml.decomposition import PCA as cuPCA
import warnings
from optimizers import *
from math import isclose
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")

class AEW:
    def __init__(self, data, gamma_init=None):
        # DATA HOLDING OBJECTS
        self.data = data
        self.gamma = self.gamma_initializer(gamma_init)
        self.similarity_matrix = None

    def generate_graphs(self, num_neighbors, mode='distance', metric='euclidean'):
        # Generate a sparse k-neighbors graph directly using cuML (GPU)
        cu_data = cp.asarray(self.data.to_numpy())  # Move data to GPU
        knn = cuNearestNeighbors(n_neighbors=num_neighbors, metric=metric)
        knn.fit(cu_data)
        graph = knn.kneighbors_graph(cu_data, mode=mode)  # Generate the k-NN graph
        self.similarity_matrix = self.correct_similarity_matrix_diag(graph)

    def correct_similarity_matrix_diag(self, similarity_matrix):
        # Correct diagonal for similarity matrix on GPU (CuPy array)
        if not isinstance(similarity_matrix, cu_csr_matrix):
            similarity_matrix = cu_csr_matrix(similarity_matrix)
        
        identity_diag_res = cp.ones(similarity_matrix.shape[0]) + 1  # Diagonal correction value
        similarity_matrix.setdiag(identity_diag_res)  # Set the diagonal directly for sparse matrix
        return similarity_matrix

    def gamma_initializer(self, gamma_init=None):
        # Initialize gamma based on the provided method
        if gamma_init is None:
            return cp.ones(self.data.shape[1])  # Default to ones if not provided
        elif gamma_init == 'var':
            return cp.var(cp.asarray(self.data), axis=0)  # Variance-based initialization
        elif gamma_init == 'random_int':
            return cp.random.randint(0, 1000, (1, self.data.shape[1]))
        elif gamma_init == 'random_float':
            return cp.random.random(size=(1, self.data.shape[1]))

    def similarity_function(self, pt1_idx, pt2_idx, gamma):
        point1 = cp.asarray(self.data.loc[[pt1_idx]]).get()  # Move data to GPU
        point2 = cp.asarray(self.data.loc[[pt2_idx]]).get()

        deg_pt1 = cp.sum(self.similarity_matrix[pt1_idx].toarray())  # Sparse matrix access on GPU
        deg_pt2 = cp.sum(self.similarity_matrix[pt2_idx].toarray())

        # Gaussian Kernel with similarity measure
        similarity_measure = cp.sum(cp.where(cp.abs(gamma) > 1e-5, (((point1 - point2) ** 2) / (gamma ** 2)), 0))
        similarity_measure = cp.exp(-similarity_measure)  # Use CuPy's exp function for GPU

        degree_normalization_term = cp.sqrt(cp.abs(deg_pt1 * deg_pt2))

        if degree_normalization_term != 0 and not isclose(degree_normalization_term, 0, abs_tol=1e-100):
            return similarity_measure / degree_normalization_term
        else:
            return 0

    def objective_computation(self, section, adj_matrix, gamma):
        error_sum = 0

        for idx in section:
            degree_idx = cp.sum(adj_matrix[idx].toarray())  # Move to GPU
            xi_reconstruction = cp.sum([adj_matrix[idx][y] * cp.asarray(self.data.loc[[y]]).get() for y in range(len(adj_matrix[idx])) if idx != y], 0)

            if degree_idx != 0 and not isclose(degree_idx, 0, abs_tol=1e-100):
                xi_reconstruction /= degree_idx
            else:
                xi_reconstruction = cp.zeros(len(self.gamma))

            error_sum += cp.sum((cp.asarray(self.data.loc[[idx]]).get() - xi_reconstruction) ** 2)

        return error_sum

    def objective_function(self, adj_matr, gamma):
        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            errors = [pool.apply_async(self.objective_computation, (section, adj_matr, gamma)) for section in split_data]
            error = cp.sum([error.get() for error in errors])
        return error

    def gradient_computation(self, section, similarity_matrix, gamma):
        gradient = cp.zeros(len(gamma))

        for idx in section:
            dii = cp.sum(similarity_matrix[idx].toarray())
            xi_reconstruction = cp.sum([similarity_matrix[idx][y] * cp.asarray(self.data.loc[[y]]).get() for y in range(len(similarity_matrix[idx])) if idx != y], 0)
            if dii != 0 and not isclose(dii, 0, abs_tol=1e-100):
                xi_reconstruction = xi_reconstruction / dii
                first_term = (cp.asarray(self.data.loc[[idx]]).get() - xi_reconstruction) / dii
            else:
                first_term = cp.zeros_like(xi_reconstruction)

            cubed_gamma = cp.where(cp.abs(gamma) > 1e-7, gamma ** (-3), 0)
            dw_dgamma = cp.sum([2 * similarity_matrix[idx][y] * (((cp.asarray(self.data.loc[[idx]]).get() - cp.asarray(self.data.loc[[y]]).get()) ** 2) * cubed_gamma) * cp.asarray(self.data.loc[[y]]).get() for y in range(self.data.shape[0]) if idx != y])
            dD_dgamma = cp.sum([2 * similarity_matrix[idx][y] * (((cp.asarray(self.data.loc[[idx]]).get() - cp.asarray(self.data.loc[[y]]).get()) ** 2) * cubed_gamma) * xi_reconstruction for y in range(self.data.shape[0]) if idx != y])

            gradient += first_term * (dw_dgamma - dD_dgamma)

        return gradient

    def split(self, a, n):
        # Efficiently split data into n parts
        k, m = divmod(len(a), n)
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    def gradient_function(self, similarity_matrix, gamma):
        gradient = []

        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            gradients = [pool.apply_async(self.gradient_computation, (section, similarity_matrix, gamma)) for section in split_data]
            gradients = [gradient.get() for gradient in gradients]

        return cp.sum(gradients, axis=0)

    def generate_optimal_edge_weights(self):
        print("Generating Optimal Edge Weights")
        self.similarity_matrix = self.generate_edge_weights(self.gamma)

        sba = SwarmBasedAnnealingOptimizer(self.similarity_matrix, self.generate_edge_weights, self.objective_function, self.gradient_function, self.gamma, 1, len(self.gamma), 3)
        self.gamma, _, _, _ = sba.optimize()

        self.similarity_matrix = self.generate_edge_weights(self.gamma)

    def edge_weight_computation(self, section, gamma):
        # Precompute edge weights for the given section in parallel
        res = []
        for idx in section:
            for vertex in range(self.data.shape[0]):
                if vertex != idx:
                    res.append((idx, vertex, self.similarity_function(idx, vertex, gamma)))

        return res

    def generate_edge_weights(self, gamma):
        print("Generating Edge Weights")
        curr_sim_matr = cu_csr_matrix(cp.zeros_like(self.similarity_matrix.toarray()))  # Use CuPy for GPU

        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            edge_weight_res = [pool.apply_async(self.edge_weight_computation, (section, gamma)) for section in split_data]
            edge_weights = [edge_weight.get() for edge_weight in edge_weight_res]

        for section in edge_weights:
            for weight in section:
                if weight[0] != weight[1]:
                    curr_sim_matr[weight[0], weight[1]] = weight[2]
                    curr_sim_matr[weight[1], weight[0]] = weight[2]

        curr_sim_matr = self.subtract_identity(curr_sim_matr)

        print("Edge Weight Generation Complete")
        return curr_sim_matr

    def subtract_identity(self, adj_matrix):
        # Subtract identity matrix from the adjacency matrix (sparse matrix)
        if isinstance(adj_matrix, cu_csr_matrix):
            identity_diag_res = cp.ones(adj_matrix.shape[0]) + 2  # Adjust for identity subtraction
            adj_matrix.setdiag(identity_diag_res)  # Efficient way to set the diagonal for sparse matrix
            return cu_csr_matrix(cp.identity(adj_matrix.shape[0])) - adj_matrix
        else:
            identity_diag_res = cp.ones(len(adj_matrix)) + 2  # Adjust for identity subtraction
            cp.fill_diagonal(adj_matrix, identity_diag_res)
            return cp.identity(len(adj_matrix)) - adj_matrix

