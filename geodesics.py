import random
import numpy as np
import torch
import torch.nn as nn


class QuadraticCurve(nn.Module):
    """
    Two-parameter latent curve with fixed endpoints:
        c(0) = z_start
        c(1) = z_end
    """
    def __init__(self, z_start, z_end):
        super().__init__()
        self.register_buffer("z_start", z_start.detach().clone())
        self.register_buffer("z_end", z_end.detach().clone())

        latent_dim = z_start.shape[0]
        self.a = nn.Parameter(torch.zeros(latent_dim))
        self.b = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, t):
        z_start = self.z_start.unsqueeze(0)
        z_end = self.z_end.unsqueeze(0)
        a = self.a.unsqueeze(0)
        b = self.b.unsqueeze(0)
        t = t.unsqueeze(-1)

        linear_part = z_start + (z_end - z_start) * t
        bend_part_1 = a * t * (1 - t)
        bend_part_2 = b * (t ** 2) * (1 - t)
        return linear_part + bend_part_1 + bend_part_2


def decode_to_probabilities(model, z):
    """
    Decode latent vectors to Bernoulli probabilities in image space.
    The decoder returns logits, so we apply sigmoid here.
    """
    logits = model.decode(z)
    return torch.sigmoid(logits)


def curve_energy_in_observation_space(curve, t_grid, model):
    """
    Discrete energy of a latent curve after decoding into image space:
        E[c] = sum_s || f(c_s) - f(c_{s-1}) ||^2
    """
    latent_points = curve(t_grid)
    decoded_points = decode_to_probabilities(model, latent_points)
    decoded_points_flat = decoded_points.view(decoded_points.size(0), -1)

    increments = decoded_points_flat[1:] - decoded_points_flat[:-1]
    squared_norms = increments.pow(2).sum(dim=1)
    return squared_norms.sum()


def ensemble_curve_energy_in_observation_space(curve, t_grid, models, num_mc_samples=10):
    latent_points = curve(t_grid)
    num_segments = latent_points.shape[0] - 1
    total_energy = torch.zeros(1, device=latent_points.device)

    for _ in range(num_mc_samples):
        sample_energy = torch.zeros(1, device=latent_points.device)

        for s in range(num_segments):
            model_left = random.choice(models)
            model_right = random.choice(models)

            decoded_left = decode_to_probabilities(model_left, latent_points[s:s + 1])
            decoded_right = decode_to_probabilities(model_right, latent_points[s + 1:s + 2])

            decoded_left_flat = decoded_left.view(decoded_left.size(0), -1)
            decoded_right_flat = decoded_right.view(decoded_right.size(0), -1)

            difference = decoded_left_flat - decoded_right_flat
            sample_energy += difference.pow(2).sum()

        total_energy += sample_energy

    return total_energy / num_mc_samples


def optimize_quadratic_curve(model, z_start, z_end, num_steps=200, lr=1e-2, num_segments=20):
    """
    Optional utility: optimize a quadratic latent curve between two points.
    """
    device = z_start.device
    curve = QuadraticCurve(z_start, z_end).to(device)
    optimizer = torch.optim.Adam(curve.parameters(), lr=lr)
    t_grid = torch.linspace(0, 1, num_segments + 1, device=device)

    for _ in range(num_steps):
        optimizer.zero_grad()
        energy = curve_energy_in_observation_space(curve, t_grid, model)
        energy.backward()
        optimizer.step()

    return curve


def sample_anchor_indices_by_class(labels, samples_per_class=3, seed=0, sort_indices=True):
    """
    Sample anchor indices within one split artifact, stratified by class.

    Example:
        anchor_indices = sample_anchor_indices_by_class(A_align["y"], samples_per_class=3, seed=0)
        zA_anchor = A_align["z"][anchor_indices]
        zB_anchor = B_align["z"][anchor_indices]
    """
    labels = torch.as_tensor(labels).clone()
    generator = torch.Generator().manual_seed(seed)

    sampled_indices = []
    for class_id in torch.unique(labels):
        class_indices = torch.where(labels == class_id)[0]
        assert len(class_indices) >= samples_per_class, (
            f"Not enough samples for class {int(class_id)}"
        )

        shuffled_class_indices = class_indices[torch.randperm(len(class_indices), generator=generator)]
        sampled_indices.append(shuffled_class_indices[:samples_per_class])

    sampled_indices = torch.cat(sampled_indices)
    if sort_indices:
        sampled_indices = torch.sort(sampled_indices)[0]

    return sampled_indices


def validate_anchor_indices(anchor_indices, labels, samples_per_class=3):
    """
    Check that the sampled anchors contain exactly the requested
    number of samples from each class present in `labels`.
    """
    labels = torch.as_tensor(labels)
    anchor_indices = torch.as_tensor(anchor_indices)

    anchor_labels = labels[anchor_indices]
    unique_anchor_labels, counts = torch.unique(anchor_labels, return_counts=True)
    unique_all_labels = torch.unique(labels)

    assert len(unique_anchor_labels) == len(unique_all_labels), (
        f"Anchor sampling missed classes: sampled={unique_anchor_labels.tolist()}, "
        f"all={unique_all_labels.tolist()}"
    )
    assert torch.all(counts == samples_per_class), (
        f"Invalid anchor sampling: labels={unique_anchor_labels.tolist()}, counts={counts.tolist()}"
    )

    return anchor_labels

def number_of_pairwise_distances(num_points):
    return num_points * (num_points - 1) // 2


def linear_path_energy(model, z_start, z_end, num_segments=20, device=None):
    """
    Linear-path energy approximation between two latent points.

    Returns:
        energy, sqrt_energy
    """
    if device is None:
        device = z_start.device

    z_start = z_start.to(device)
    z_end = z_end.to(device)

    t_grid = torch.linspace(0, 1, num_segments + 1, device=device).unsqueeze(1)
    latent_path = z_start.unsqueeze(0) + (z_end - z_start).unsqueeze(0) * t_grid

    decoded_path = decode_to_probabilities(model, latent_path)
    decoded_path_flat = decoded_path.view(decoded_path.size(0), -1)

    increments = decoded_path_flat[1:] - decoded_path_flat[:-1]
    energy = increments.pow(2).sum(dim=1).sum()
    sqrt_energy = torch.sqrt(torch.clamp(energy, min=1e-12))

    return energy, sqrt_energy


def pairwise_geodesic_distances(model, latent_points, num_segments=20, device="cpu"):
    """
    Compute an NxN symmetric matrix of approximate geodesic distances
    using the linear-path energy approximation:
        d(z_i, z_j) ≈ sqrt(E_linear(z_i, z_j))
    """
    latent_points = torch.as_tensor(latent_points, dtype=torch.float32)
    assert latent_points.ndim == 2, "latent_points must have shape (N, latent_dim)"

    num_points = latent_points.shape[0]
    distance_matrix = torch.zeros(num_points, num_points, dtype=torch.float32)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(num_points):
            for j in range(i + 1, num_points):
                _, distance_ij = linear_path_energy(
                    model,
                    latent_points[i],
                    latent_points[j],
                    num_segments=num_segments,
                    device=device,
                )
                distance_ij = distance_ij.detach().cpu().float()
                distance_matrix[i, j] = distance_ij
                distance_matrix[j, i] = distance_ij

    return distance_matrix.numpy()



def coefficient_of_variation(distances): 
    """
    Coefficient of Variation (CoV) = std / mean
    """
    mean = np.mean(distances)
    std = np.std(distances)
    cov = std / mean
    return cov

