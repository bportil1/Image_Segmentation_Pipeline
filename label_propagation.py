import numpy as np
import cv2
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class LabelPropagation:
    def __init__(self, adjacency_matrix, labels, max_iter=100, tolerance=1e-3):
        """
        :param adjacency_matrix: Square matrix representing graph structure (N x N)
        :param labels: Initial labels array (N,) where unassigned nodes should be labeled as 0 or NaN
        :param max_iter: Maximum number of iterations for label propagation
        :param tolerance: Convergence tolerance for stopping criteria
        """
        self.adjacency_matrix = adjacency_matrix
        self.labels = np.array(labels)
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
        labels = np.copy(self.labels)
        
        for iteration in range(self.max_iter):
            # Initialize label propagation
            new_labels = self.adjacency_matrix @ labels

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

