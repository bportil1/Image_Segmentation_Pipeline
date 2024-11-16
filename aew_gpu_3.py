import numpy as np
import pandas as pd
#import cupy as cp
#from cupy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
import warnings
from math import isclose
from concurrent.futures import ThreadPoolExecutor
from optimizers import *
from multiprocessing import cpu_count, Pool
import multiprocessing
import os
import cupy as cp
from cupy.sparse import csr_matrix

import scipy.sparse as sp

from numba import cuda, float32

# Initialize warnings
warnings.filterwarnings("ignore")
    
class AEW:
    def __init__(self, data, gamma_init=None):
        # DATA HOLDING OBJECTS
        self.data = cp.asarray(data.values, dtype=cp.float64)  # Convert data to CuPy array (GPU array)
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
        #print("Computing Edge Weights")

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

        #print("Completed Edge Weights Computation")
        return normalized_similarity

    def optimized_edge_weight_update(self, edge_weight_res, curr_sim_matr):
        """Parallelize the collection of edge weights and update the sparse matrix on CPU."""
        print("Updating Edge Weights")
        #os.environ["CUDA_VISIBLE_DEVICES"] = ""
        multiprocessing.set_start_method('spawn', force=True)
        # Use multiprocessing Pool to collect non-zero entries in parallel (CPU)
        with multiprocessing.Pool(processes=cpu_count()) as pool:
            results = pool.map(self.collect_non_zero_entries, edge_weight_res)
        #print('after mp')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
        # Flatten results
        row_indices = []
        col_indices = []
        values = []

        # Collect all non-zero entries
        for r, c, v in results:
            row_indices.extend(r)
            col_indices.extend(c)
            values.extend(v)

        #print('collecting results')

        # Create a sparse matrix using the collected entries on CPU (not GPU)
        row_indices = np.array(row_indices, dtype=np.int32)
        col_indices = np.array(col_indices, dtype=np.int32)
        values = np.array(values, dtype=np.float32)

        #for r, c, v in zip(row_indices, col_indices, values):
        #    curr_sim_matr[r, c] += v
        
        curr_sim_matr = curr_sim_matr.tocsr()

        num_entries = len(row_indices)
    
    
        batch_size=1000
        # Process the updates in batches
        for i in range(0, num_entries, batch_size):
            batch_rows = row_indices[i:i+batch_size]
            batch_cols = col_indices[i:i+batch_size]
            batch_vals = values[i:i+batch_size]
        
            # Create a COO matrix for the batch
            batch_update = sp.coo_matrix((batch_vals, (batch_rows, batch_cols)), shape=curr_sim_matr.shape)
        
            batch_update = cp.sparse.csr_matrix(batch_update)
        
             # Efficiently update the current matrix
            curr_sim_matr += batch_update
        
        
        print("Edge Weights Updated")
        return curr_sim_matr

    def collect_non_zero_entries(self, section):
        """Collect non-zero entries from a section of the edge weight results efficiently."""
        row_indices = []
        col_indices = []
        values = []

        # Check if 'section' is a CuPy array or NumPy array and convert if necessary
        if isinstance(section, cp.ndarray):
            section = section.get()  # Convert CuPy to NumPy array for processing

        # Use sparse matrix methods if possible to avoid holding large arrays in memory
        non_zero = np.nonzero(section)  # Get indices of non-zero entries

        # Collect non-zero entries in the section
        for idx1, idx2 in zip(*non_zero):
            row_indices.append(idx1)
            col_indices.append(idx2)
            values.append(section[idx1, idx2])

        #row_indices = np.array(row_indices)
        #col_indices = np.array(col_indices)
        #values = np.array(values)

        return row_indices, col_indices, values
    '''
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
    '''
    '''
    @cuda.jit
    def gradient_computation_kernel(section, similarity_matrix, gamma, result):
        """GPU-optimized gradient computation."""
        idx = cuda.grid(1)
        if idx < len(section):
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

    def gradient_computation(self, section, similarity_matrix, gamma):
        """Wrapper to call the kernel for gradient computation."""
        print("Computing Gradient")
        result = cp.zeros(len(section), dtype=cp.float32)
        threads_per_block = 256  # Optimal number of threads per block
        blocks_per_grid = (len(section) + threads_per_block - 1) // threads_per_block
        AEW.gradient_computation_kernel[blocks_per_grid, threads_per_block](section, similarity_matrix, gamma, result)
        print("Gradient Computation Complete")
        return cp.sum(result)
    
    @cuda.jit
    def objective_computation_kernel(section, adj_matrix, gamma, result):
        """GPU-optimized kernel to compute the objective function."""
        idx = cuda.grid(1)
        if idx < len(section):
            degree_idx = cp.sum(adj_matrix[idx].toarray())  # Sparse matrix access
            xi_reconstruction = cp.sum([adj_matrix[idx, y] * self.data[y] for y in range(adj_matrix.shape[1]) if idx != y], 0)

            if degree_idx != 0 and not isclose(degree_idx, 0, abs_tol=1e-100):
                xi_reconstruction /= degree_idx
            else:
                xi_reconstruction = cp.zeros(len(gamma))

            result[idx] = cp.sum((self.data[idx] - xi_reconstruction) ** 2)

    def objective_computation(self, section, adj_matrix, gamma):
        """Wrapper to call the kernel for the objective computation."""
        print("Computing Error")
        result = cp.zeros(len(section), dtype=cp.float32)
        threads_per_block = 32  # Optimal number of threads per block
        blocks_per_grid = (len(section) + threads_per_block - 1) // threads_per_block
        AEW.objective_computation_kernel[blocks_per_grid, threads_per_block](section, adj_matrix, gamma, result)
        print("Error Computation Complete")
        return cp.sum(result)
    '''

    @cuda.jit
    def gradient_computation_kernel(similarity_matrix, dii, gamma, result, data, xi_reconstruction):
        """GPU-optimized gradient computation without the `section` variable."""
        idx = cuda.grid(1)
        if idx < similarity_matrix.shape[0]:
            # Get the degree (sum of the similarities) for the current index
            #dii = cp.sum(similarity_matrix[idx].toarray())  # Ensure sparse

            # Reconstruction based on the current similarity matrix row
            #xi_reconstruction = cp.sum([similarity_matrix[idx, y] * data[y] for y in range(similarity_matrix.shape[1]) if idx != y], 0)
            '''
            xi_reconstruction = 0.0  # Initialize the variable to accumulate the sum

            # Manually sum the elements in the list comprehension
            for y in range(similarity_matrix.shape[1]):
                if idx != y:  # Skip the diagonal element
                    xi_reconstruction += similarity_matrix[idx, y] * data[idx][y]
            '''
            
            #xi_reconstruction = [0.0 for _ in range(len(data[idx]))]
            for y in range(len(data[idx])):
            # Compute the first term
                if dii[idx] != 0 and (dii[idx] > 1e-100 or dii[idx] < -1e-100):
                    xi_reconstruction[y] = xi_reconstruction[y] / dii[idx]
                    first_term = (data[idx][y] - xi_reconstruction[y]) / dii[idx]
                else:
                    xi_reconstruction[y] = 0.0

            # Compute the derivative terms w.r.t gamma
            #cubed_gamma = []

            for g in range(len(gamma)):
                if abs(g) > 1e-7:
                    gamma[g] = gamma[g] ** (-3)
                else:
                    gamma[g] = 0
                    
                    
            #dw_dgamma = cp.sum([2 * similarity_matrix[idx, y] * (((data[idx] - data[y]) ** 2) * cubed_gamma) * data[y] for y in range(similarity_matrix.shape[0]) if idx != y])
            dw_dgamma = 0.0  # Initialize the result

            for y in range(len(similarity_matrix)):  # Loop over the rows (or elements)
                if idx != y:  # Avoid diagonal elements (i.e., similarity_matrix[idx, idx])
                    # Manually calculate the terms and accumulate the sum
                    #term = 2 * similarity_matrix[idx][y] * (((data[idx] - data[y]) ** 2) * gamma[y]) * data[y]         
                    term = 0.0
                    for k in range(len(data[idx])):  # Loop through each element in the vector
                        diff = data[idx][k] - data[y][k]  # Element-wise difference
                        term += 2 * similarity_matrix[idx, y] * (diff ** 2) * gamma[y] * data[y][k]
                    
                    
                    
                    dw_dgamma += term
            
            
            #dD_dgamma = cp.sum([2 * similarity_matrix[idx, y] * (((data[idx] - data[y]) ** 2) * cubed_gamma) * xi_reconstruction for y in range(similarity_matrix.shape[0]) if idx != y])
            dD_dgamma = 0.0
            data_diff_squared = 0.0
            # Loop over the rows of similarity_matrix
            for y in range(len(similarity_matrix)):  # Loop over the columns (or elements)
                if idx != y:  # Avoid diagonal elements
                    # Calculate the term for each element manually
                    # Access the matrix elements manually instead of using cp or np methods
                    similarity_val = similarity_matrix[idx][y]
                    for k in range(len(data[idx])): 
                        data_diff_squared += ((data[idx][k] - data[y][k]) ** 2)*gamma[k]*xi_reconstruction[k]
                    term = 2 * similarity_val * data_diff_squared
                    dD_dgamma += term  # Sum up the result manually


            # Update result for the current index
            result[idx] = first_term * (dw_dgamma - dD_dgamma)
            
    def gradient_computation(self, similarity_matrix, gamma):
        """Wrapper to call the kernel for gradient computation."""
        print("Computing Gradient")
        result = cp.zeros(similarity_matrix.shape[0], dtype=cp.float32)
        #dii = cp.sum(similarity_matrix[idx].toarray()) 
        dii = cp.array([cp.sum(row.toarray()).item() for row in similarity_matrix]) 
        similarity_matrix = cp.asarray(similarity_matrix.toarray()).astype(cp.float64)
        data = cp.asarray(self.data, dtype=cp.float64)
        xi_reconstruction = cp.zeros(data.shape[1], dtype=cp.float64)
        # Launching CUDA kernel with grid and block configurations
        threads_per_block = 256  # Optimal number of threads per block
        blocks_per_grid = (similarity_matrix.shape[0] + threads_per_block - 1) // threads_per_block
        AEW.gradient_computation_kernel[blocks_per_grid, threads_per_block](similarity_matrix, dii, gamma, result, data, xi_reconstruction)
        
        print("Gradient Computation Complete")
        return cp.sum(result).get()

    @cuda.jit
    def objective_computation_kernel(adj_matrix, degree_idx, gamma, result, data, xi_reconstruction ):
        """GPU-optimized kernel to compute the objective function without the `section` variable."""
        idx = cuda.grid(1)
        if idx < adj_matrix.shape[0]:
            # Degree for the current index (sum of the similarities)
            #degree_idx = cp.sum(adj_matrix[idx].toarray())  # Sparse matrix access

            # Reconstruct xi using the current adjacency matrix row
            #xi_reconstruction = cp.sum([adj_matrix[idx, y] * data[y] for y in range(adj_matrix.shape[1]) if idx != y], 0)
         
            #xi_reconstruction = [0.0 for _ in range(len(data[idx]))]

            for y in range(len(data[idx])):
                if idx != y:  # Avoid diagonal elements
                    xi_reconstruction[y] = adj_matrix[idx, y] * data[idx][y]

            # Handle division by degree (avoid division by zero)
            for y in range(len(data[idx])):
                if degree_idx[idx] != 0 and (degree_idx[idx] > 1e-100 or degree_idx[idx] < -1e-100):
                    xi_reconstruction[y] = xi_reconstruction[y] / degree_idx[idx]
                else:
                    xi_reconstruction[y] = 0.0


            squared_differences = 0.0
            for i in range(len(data[idx])):
                squared_differences += (data[idx][i] - xi_reconstruction[i]) ** 2

            result[idx] = squared_differences

            # Compute the result for the objective function
            #result[idx] = cp.sum((data[idx] - xi_reconstruction) ** 2)

    def objective_computation(self, adj_matrix, gamma):
        """Wrapper to call the kernel for the objective computation."""
        print("Computing Error")
        result = cp.zeros(adj_matrix.shape[0], dtype=cp.float32)
        degree_idx = cp.array([cp.sum(row.toarray()).item() for row in adj_matrix]) 
        adj_matrix = cp.asarray(adj_matrix.toarray()).astype(cp.float64)
        data = cp.asarray(self.data, dtype=cp.float64)
        xi_reconstruction = cp.zeros(data.shape[1], dtype=cp.float64)
        # Launching CUDA kernel with grid and block configurations
        threads_per_block = 32  # Optimal number of threads per block
        blocks_per_grid = (adj_matrix.shape[0] + threads_per_block - 1) // threads_per_block
        #print(xi_reconstruction)
        AEW.objective_computation_kernel[blocks_per_grid, threads_per_block](adj_matrix, degree_idx, gamma, result, data, xi_reconstruction)

        print("Error Computation Complete")
        return cp.sum(result).get()

    def split(self, a, n):
        # Efficiently split data into n parts
        k, m = divmod(len(a), n)
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    def generate_optimal_edge_weights(self):
        print("Generating Optimal Edge Weights")
        self.similarity_matrix = self.generate_edge_weights(self.gamma)

        sba = SwarmBasedAnnealingOptimizer(self.similarity_matrix, self.generate_edge_weights, self.objective_computation, self.gradient_computation, self.gamma, 1, len(self.gamma), 1)
        self.gamma = cp.array(sba.optimize()[0])
        

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


    @staticmethod
    def init_worker():
        """Initialize worker process environment to disable GPU usage."""
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Ensure that the worker also has GPU disabled
        print("Worker initialized with no GPU access.")

