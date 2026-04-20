import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from colors import get_colors, get_single_color
from geodesics import compute_geodesics_S2, compute_geodesics


def sample_uniform_S2(n):
    """
    Sample n points uniformly on S^2.
    """
    x = torch.randn(n, 3)
    return x / x.norm(dim=1, keepdim=True)



def LDD(Z, C_g, r_bins, r_max, n_centers=None, center_idx=None):
    """
    Compute LDD signatures from a precomputed geodesic matrix.
    """
    Z = torch.as_tensor(Z, dtype=torch.float32)
    C_g = torch.as_tensor(C_g, dtype=torch.float32)

    n = Z.shape[0]

    # Check that Z and C_g use the same ordering.
    if C_g.shape != (n, n):
        raise ValueError("C_g must have shape (n, n) matching Z.")

    # Choose which points are used as centers.
    if center_idx is not None:
        center_idx = torch.as_tensor(center_idx, dtype=torch.long)
    elif n_centers is None:
        center_idx = torch.arange(n)
    else:
        if not (1 <= n_centers <= n):
            raise ValueError("n_centers must satisfy 1 <= n_centers <= len(Z).")
        center_idx = torch.randperm(n)[:n_centers]

    # Keep only the rows corresponding to the chosen centers.
    z_points = Z[center_idx]
    C_centers = C_g[center_idx].clone()

    # Remove self-count for each chosen center.
    row_idx = torch.arange(C_centers.shape[0])
    C_centers[row_idx, center_idx] = torch.inf

    # Define the radius grid.
    r = torch.linspace(0.0, float(r_max), steps=r_bins, dtype=torch.float32)

    # Compare every center-to-point distance with every radius bin at once.
    indicators = (C_centers[:, :, None] <= r[None, None, :]).float()

    # Average over all points except the center itself.
    H = indicators.sum(dim=1) / (n - 1)

    return H, r, center_idx, z_points, C_centers


def signature_distance_matrix(H):
    """
    Pairwise RMS distances between LDD signatures.
    """
    H = torch.as_tensor(H, dtype=torch.float32)

    diff = H[:, None, :] - H[None, :, :]
    D = torch.sqrt((diff ** 2).mean(dim=2))

    return D

