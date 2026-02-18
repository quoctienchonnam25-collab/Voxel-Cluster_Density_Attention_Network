"""
Utility Functions for Saliency Map Processing

This module contains helper functions for:
- Extracting top-K coordinates from saliency maps
- Computing Gaussian distance matrices
"""

import torch
import numpy as np


def extract_top_k_coordinates(cam_3d, k=128):
    """
    Extract top-K voxel coordinates from 3D CAM
    
    Args:
        cam_3d: [H, W, D] numpy array or tensor
    
    Returns:
        coords: [K, 3] coordinates
        values: [K] CAM values
    """
    if isinstance(cam_3d, torch.Tensor):
        cam_3d = cam_3d.cpu().numpy()
    
    # Flatten and get top-K indices
    flat_cam = cam_3d.flatten()
    top_k_flat_indices = np.argpartition(flat_cam, -k)[-k:]
    top_k_flat_indices = top_k_flat_indices[np.argsort(flat_cam[top_k_flat_indices])[::-1]]
    
    # Convert flat indices to 3D coordinates
    coords = np.array(np.unravel_index(top_k_flat_indices, cam_3d.shape)).T
    values = flat_cam[top_k_flat_indices]
    
    return coords, values


def compute_gaussian_distance_matrix(coordinates, sigma=10.0):
    """
    Compute KxK Gaussian distance matrix (VECTORIZED - much faster!)
    
    Args:
        coordinates: [K, 3] numpy array
        sigma: Gaussian bandwidth
    
    Returns:
        matrix: [K, K] Gaussian distance matrix
    """
    # Compute pairwise squared distances using broadcasting
    # coords: [K, 3] -> [K, 1, 3] and [1, K, 3]
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]  # [K, K, 3]
    squared_dist = np.sum(diff ** 2, axis=2)  # [K, K]
    
    # Gaussian kernel
    matrix = np.exp(-squared_dist / (2 * sigma ** 2))
    
    return matrix.astype(np.float32)
