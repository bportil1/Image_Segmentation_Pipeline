import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from optimizers import *
import warnings
from math import isclose
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import cpu_count, Pool
# Initialize warnings
warnings.filterwarnings("ignore")

class AEW:
    def __init__(self, data, gamma_init=None):
        # DATA HOLDING OBJECTS
        self.data = data
        self.gamma = self.gamma_initializer(gamma_init)
        self.similarity_matrix = None

    def generate_graphs(self, num_neighbors, mode='distance', metric='euclidean'):
        # Generate a sparse k-neighbors graph
        graph = kneighbors_graph(self.data, n_neighbors=num_neighbors, mode=mode, metric=metric, p=2, include_self=False, n_jobs=-1)
        self.similarity_matrix = self.correct_similarity_matrix_diag(graph)

    def correct_similarity_matrix_diag(self, similarity_matrix):
        # Convert to sparse matrix and adjust diagonal
        if not isinstance(similarity_matrix, csr_matrix):
            similarity_matrix = csr_matrix(similarity_matrix)
        
        identity_diag_res = np.ones(similarity_matrix.shape[0]) + 1  # Diagonal correction value
        similarity_matrix.setdiag(identity_diag_res)  # Set the diagonal directly for sparse matrix
        return similarity_matrix

    def gamma_initializer(self, gamma_init=None):
        # Initialize gamma based on the provided method
        if gamma_init is None:
            return np.ones(self.data.shape[1])  # Default to ones if not provided
        elif gamma_init == 'var':
            return np.var(self.data, axis=0)  # Variance-based initialization
        elif gamma_init == 'random_int':
            return np.random.randint(0, 1000, (1, self.data.shape[1]))
        elif gamma_init == 'random_float':
            return np.random.random(size=(1, self.data.shape[1]))

    def similarity_function(self, pt1_idx, pt2_idx, gamma):
        point1 = np.asarray(self.data.loc[[pt1_idx]])[0]
        point2 = np.asarray(self.data.loc[[pt2_idx]])[0]

        deg_pt1 = np.sum(self.similarity_matrix[pt1_idx].toarray())  # Sparse matrix access
        deg_pt2 = np.sum(self.similarity_matrix[pt2_idx].toarray())

        # Gaussian Kernel with similarity measure
        similarity_measure = np.sum(np.where(np.abs(gamma) > 1e-5, (((point1 - point2) ** 2) / (gamma ** 2)), 0))
        similarity_measure = np.exp(-similarity_measure, dtype=np.float64)  # Use float64 to reduce precision cost

        degree_normalization_term = np.sqrt(np.abs(deg_pt1 * deg_pt2))

        if degree_normalization_term != 0 and not isclose(degree_normalization_term, 0, abs_tol=1e-100):
            return similarity_measure / degree_normalization_term
        else:
            return 0

    def objective_computation(self, section, adj_matrix, gamma):
        error_sum = 0

        for idx in section:
            degree_idx = np.sum(adj_matrix[idx].toarray())  # Ensure using sparse
            xi_reconstruction = np.sum([adj_matrix[idx, y] * np.asarray(self.data.loc[[y]])[0]
                                    for y in range(adj_matrix[idx].shape[1]) if idx != y], 0)
            if degree_idx != 0 and not isclose(degree_idx, 0, abs_tol=1e-100):
                xi_reconstruction /= degree_idx
            else:
                xi_reconstruction = np.zeros(len(self.gamma))

            error_sum += np.sum((np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction) ** 2)

        return error_sum

    def objective_function(self, adj_matr, gamma):
        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            errors = [pool.apply_async(self.objective_computation, (section, adj_matr, gamma)) for section in split_data]
            error = np.sum([error.get() for error in errors])  # Ensure using np.sum
        return error

    def gradient_computation(self, section, similarity_matrix, gamma):
        gradient = np.zeros(len(gamma))

        for idx in section:
            dii = np.sum(similarity_matrix[idx].toarray())  # Ensure sparse
            xi_reconstruction = np.sum([similarity_matrix[idx, y] * np.asarray(self.data.loc[[y]])[0] 
                            for y in range(similarity_matrix.shape[1]) if idx != y], 0)

            if dii != 0 and not isclose(dii, 0, abs_tol=1e-100):
                xi_reconstruction = xi_reconstruction / dii
                first_term = (np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction) / dii
            else:
                first_term = np.zeros_like(xi_reconstruction)

            cubed_gamma = np.where(np.abs(gamma) > 1e-7, gamma ** (-3), 0)
            print(similarity_matrix[idx][0])
            print(np.asarray(self.data.loc[[idx]])[0] )
            print(self.data.shape[0])
            print(np.asarray(self.data.loc[[0]])[0])
            dw_dgamma = np.sum([2 * similarity_matrix[idx][y] * (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0]) ** 2) * cubed_gamma) * np.asarray(self.data.loc[[y]])[0] for y in range(self.data.shape[0]) if idx != y])
            dD_dgamma = np.sum([2 * similarity_matrix[idx][y] * (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0]) ** 2) * cubed_gamma) * xi_reconstruction for y in range(self.data.shape[0]) if idx != y])

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

        return np.sum(gradients, axis=0)

    def generate_optimal_edge_weights(self):
        print("Generating Optimal Edge Weights")
        self.similarity_matrix = self.generate_edge_weights(self.gamma)

        sba = SwarmBasedAnnealingOptimizer(self.similarity_matrix, self.generate_edge_weights, self.objective_function, self.gradient_function, self.gamma, 1, len(self.gamma), 1)
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

    def optimized_edge_weight_update(self, edge_weight_res, curr_sim_matr):
        # Prepare lists for row indices, column indices, and corresponding values
        row_indices = []
        col_indices = []
        values = []

        for section in edge_weight_res:
            for weight in section:
                idx1, idx2, w = weight
                if idx1 != idx2:  # Avoid self-loops early
                    row_indices.append(idx1)
                    col_indices.append(idx2)
                    values.append(w)
                    row_indices.append(idx2)
                    col_indices.append(idx1)
                    values.append(w)

        # Convert lists to numpy arrays for sparse matrix creation
        row_indices = np.array(row_indices)
        col_indices = np.array(col_indices)
        values = np.array(values)

        # Create a sparse matrix from the collected values
        new_sim_matr = csr_matrix((values, (row_indices, col_indices)), shape=curr_sim_matr.shape)

        # Add the newly computed values to the current similarity matrix
        curr_sim_matr += new_sim_matr

        return curr_sim_matr

    def generate_edge_weights(self, gamma):
        print("Generating Edge Weights")
        curr_sim_matr = csr_matrix(np.zeros_like(self.similarity_matrix.toarray()))

        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            edge_weight_res = [pool.apply_async(self.edge_weight_computation, (section, gamma)) for section in split_data]
            edge_weights = [edge_weight.get() for edge_weight in edge_weight_res]

        # Efficiently update the sparse matrix
        curr_sim_matr = self.optimized_edge_weight_update(edge_weights, curr_sim_matr)

        curr_sim_matr = self.subtract_identity(curr_sim_matr)

        print("Edge Weight Generation Complete")
        return curr_sim_matr

    def subtract_identity(self, adj_matrix):
        # Subtract identity matrix from the adjacency matrix (sparse matrix)
        if isinstance(adj_matrix, csr_matrix):
            identity_diag_res = np.ones(adj_matrix.shape[0]) + 2  # Adjust for identity subtraction
            adj_matrix.setdiag(identity_diag_res)  # Efficient way to set the diagonal for sparse matrix
            return csr_matrix(np.identity(adj_matrix.shape[0])) - adj_matrix
        else:
            identity_diag_res = np.ones(len(adj_matrix)) + 2  # Adjust for identity subtraction
            np.fill_diagonal(adj_matrix, identity_diag_res)
            return np.identity(len(adj_matrix)) - adj_matrix

