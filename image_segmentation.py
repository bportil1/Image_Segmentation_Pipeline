from aew_gpu_3 import *
from label_propagation import *

import pandas as pd
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from scipy.sparse import issparse

def compute_adjacency_matrix(image, sigma=1.0):
    """
    Computes an adjacency matrix for an image based on pixel similarities (color).

    :param image: The input image (H x W x C), where C is the number of channels (e.g., 3 for RGB).
    :param sigma: Standard deviation for the Gaussian kernel to compute similarity.
    :return: Adjacency matrix of size (N x N) where N is the number of pixels.
    """
    height, width, channels = image.shape
    num_pixels = height * width
    flattened_image = image.reshape((-1, channels))  # Reshape to (N, C)

    # Compute pairwise color distances (Euclidean distance between pixels)
    #dist_matrix = cdist(flattened_image, flattened_image, metric='euclidean')

    aew_obj = AEW(pd.DataFrame(flattened_image), 'var')

    aew_obj.generate_graphs(150)

    aew_obj.generate_optimal_edge_weights()

    if issparse(aew_obj.similarity_matrix):
        aew_obj.similarity_matrix = aew_obj.similarity_matrix.toarray()

    return aew_obj.similarity_matrix

def generate_dynamic_seed_points(image, num_seeds=5):
    """
    Dynamically generate seed points using K-means clustering on pixel colors.

    :param image: The input image (H x W x C).
    :param num_seeds: Number of clusters (seeds) to generate.
    :return: List of seed points as (x, y, label) where (x, y) are the coordinates and 'label' is the label.
    """
    height, width, _ = image.shape
    flattened_image = image.reshape((-1, 3))  # Flatten to (N, 3) where N is number of pixels

    # Apply K-means clustering to the flattened image
    kmeans = KMeans(n_clusters=num_seeds, random_state=42)
    kmeans.fit(flattened_image)

    # Get the cluster centers (RGB values) and the pixel assignments
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Initialize seed points as the centroids of the clusters
    seed_points = []
    for label in range(num_seeds):
        # Find the pixel closest to the cluster center
        cluster_mask = labels == label
        cluster_pixels = np.column_stack(np.where(cluster_mask.reshape(height, width)))  # (x, y) coordinates
        closest_pixel = cluster_pixels[np.argmin(cdist(cluster_centers[label].reshape(1, -1), flattened_image[cluster_mask]))]
        seed_points.append((closest_pixel[0], closest_pixel[1], label + 1))  # Labeling starts from 1

    return seed_points

def initialize_labels(image, seed_points):
    """
    Initialize labels based on seed points.

    :param image: The input image (H x W x C).
    :param seed_points: A list of tuples (x, y, label) where (x, y) are the coordinates
                        of the seed point and 'label' is the label for that point.
    :return: A label array of shape (H * W,) where each element corresponds to a pixel label.
    """
    height, width, _ = image.shape
    labels = np.zeros((height, width), dtype=int)

    # Assign labels to seed points
    for x, y, label in seed_points:
        labels[x, y] = label

    return labels.flatten()  # Flatten to a 1D array (N,)

def segment_image(image, num_seeds=5, max_iter=100, sigma=1.0):
    """
    Perform image segmentation using label propagation and dynamic seed points.

    :param image: The input image (H x W x C).
    :param num_seeds: Number of clusters (seeds) to generate dynamically.
    :param max_iter: Maximum number of iterations for label propagation.
    :param sigma: Standard deviation for the Gaussian similarity kernel.
    :return: Segmented labels for the image.
    """
    # Step 1: Generate dynamic seed points using K-means clustering
    seed_points = generate_dynamic_seed_points(image, num_seeds)

    # Step 2: Compute adjacency matrix based on pixel similarity
    adjacency_matrix = compute_adjacency_matrix(image, sigma)

    # Step 3: Initialize labels for the seed points
    labels = initialize_labels(image, seed_points)

    # Step 4: Perform label propagation
    label_propagator = LabelPropagation(adjacency_matrix, labels, max_iter=max_iter)
    propagated_labels = label_propagator.propagate_labels()

    propagated_labels = propagated_labels.get()

    # Reshape the propagated labels back to the image dimensions
    height, width, _ = image.shape
    segmented_image = propagated_labels.reshape((height, width))

    return segmented_image

def visualize_segmentation(image, segmented_image):
    """
    Visualize the segmented image using a simple color map.

    :param image: The original input image.
    :param segmented_image: The label matrix (segmented output).
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='tab20b')  # Use a colormap to visualize different segments
    plt.title("Segmented Image")

    plt.show()

if __name__ == "__main__":
    image =  cv2.imread('80_12.png')
    segmented_image = segment_image(image)
    visualize_segmentation(image, segmented_image)
