import numpy as np
import cv2
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import cupy as cp
import cupyx.scipy.sparse as sp
'''
class LabelPropagation:
    def __init__(self, adjacency_matrix, labels, max_iter=100, tolerance=1e-3):
        """
        :param adjacency_matrix: Square matrix representing graph structure (N x N)
        :param labels: Initial labels array (N,) where unassigned nodes should be labeled as 0 or NaN
        :param max_iter: Maximum number of iterations for label propagation
        :param tolerance: Convergence tolerance for stopping criteria
        """
        self.adjacency_matrix = adjacency_matrix
        self.labels = cp.array(labels, dtype=cp.float64)
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.num_nodes = len(labels)
        
        # Normalize the adjacency matrix
        #self.D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(self.adjacency_matrix, axis=1)))
       # self.normalized_adj = self.D_inv_sqrt @ self.adjacency_matrix @ self.D_inv_sqrt

    def propagate_labels(self):
        """
        Propagates labels through the graph using label propagation.
        """
        # Create a copy of labels to work with during the iterations
    
        print(type(self.adjacency_matrix))

        print(self.labels)

        labels = cp.copy(self.labels)
        
        for iteration in range(self.max_iter):
            # Initialize label propagation
            new_labels = cp.zeros_like(labels, dtype=cp.float64)
            for i in range(self.num_nodes):
                row_start = int(self.adjacency_matrix.indptr[i].get())
                row_end = int(self.adjacency_matrix.indptr[i + 1].get())
                for j in range(row_start, row_end):
                    col_index = int(self.adjacency_matrix.indices[j].get())
                    new_labels[i] += self.adjacency_matrix.data[j] * labels[col_index]

            # Keep the original labels fixed (fixed labels should not change)
            mask = self.labels != 0
            new_labels[mask] = self.labels[mask]

            # Check for convergence: If the labels have not changed significantly, stop
            if np.linalg.norm(new_labels - labels) < self.tolerance:
                print(f"Convergence reached at iteration {iteration + 1}")
                break

            labels = new_labels
        
        # Return the final propagated labels
        return labels
'''

import numpy as np
import cupy as cp
import cupyx.scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor

class LabelPropagation:
    def __init__(self, adjacency_matrix, labels, max_iter=100, tolerance=1e-3):
        """
        :param adjacency_matrix: Square matrix representing graph structure (N x N), in CSR format
        :param labels: Initial labels array (N,) where unassigned nodes should be labeled as 0 or NaN
        :param max_iter: Maximum number of iterations for label propagation
        :param tolerance: Convergence tolerance for stopping criteria
        """
        if not sp.isspmatrix_csr(adjacency_matrix):
            raise ValueError("adjacency_matrix must be a CSR matrix")
        
        self.adjacency_matrix = adjacency_matrix
        self.labels = cp.array(labels, dtype=cp.float64)  # Ensure labels are float64
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.num_nodes = len(labels)

    def _propagate_row(self, i, labels):
        """
        Helper function to propagate labels for a single row `i`.
        This will be executed in parallel for each node (row).
        """
        new_label = 0.0
        row_start = int(self.adjacency_matrix.indptr[i].get())
        row_end = int(self.adjacency_matrix.indptr[i + 1].get())
        
        for j in range(row_start, row_end):
            col_index = int(self.adjacency_matrix.indices[j].get())
            new_label += self.adjacency_matrix.data[j] * labels[col_index]
        
        return new_label

    def propagate_labels(self):
        """
        Propagates labels through the graph using label propagation, with parallelization.
        """
        # Create a copy of labels to work with during the iterations
        labels = cp.copy(self.labels)

        for iteration in range(self.max_iter):
            # Initialize new_labels as a zero vector
            new_labels = cp.zeros_like(labels, dtype=cp.float64)  # Ensure new_labels are float64

            # Use multiprocessing to parallelize label propagation for each node
            with ProcessPoolExecutor() as executor:
                results = executor.map(self._propagate_row, range(self.num_nodes), [labels] * self.num_nodes)
                for i, new_label in enumerate(results):
                    new_labels[i] = new_label

            # Keep the original labels fixed (fixed labels should not change)
            mask = self.labels != 0
            new_labels[mask] = self.labels[mask]

            # Check for convergence: If the labels have not changed significantly, stop
            if cp.linalg.norm(new_labels - labels) < self.tolerance:
                print(f"Convergence reached at iteration {iteration + 1}")
                break

            labels = new_labels

        # Return the final propagated labels
        return labels



