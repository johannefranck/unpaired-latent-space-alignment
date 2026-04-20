import os
import json
import argparse
import numpy as np
import torch

from geodesics import compute_geodesics_S2


# samples the points for the chosen S2 distribution experiment
# saves the points
# saves distribution specific metadata (e.g. distribution parameters)
# computes and save geodesics between the points (arccos)
# saves the canonical point order and any experiment-specific indices



# ============================================================
# Sampling utilities
# ============================================================

def sample_uniform_S2(n):
    """
    Sample n points uniformly on S^2.
    """
    x = torch.randn(n, 3)
    return x / x.norm(dim=1, keepdim=True)


def _orthonormal_basis(mu):
    """
    Build two unit vectors orthogonal to mu.
    """
    mu = np.asarray(mu, dtype=float)
    mu = mu / np.linalg.norm(mu)

    if np.allclose(mu, np.array([1.0, 0.0, 0.0])):
        v1 = np.array([0.0, 1.0, 0.0])
    else:
        v1 = np.cross(mu, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(v1) < 1e-8:
            v1 = np.cross(mu, np.array([0.0, 1.0, 0.0]))

    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(mu, v1)
    v2 = v2 / np.linalg.norm(v2)

    return mu, v1, v2


def sample_symmetric_ring_points(mu, layer_thetas, n_points):
    """
    Construct perfectly symmetric ring samples on S^2 around mu.

    Uses the largest number of points that can be distributed equally
    across all rings.
    """
    mu, v1, v2 = _orthonormal_basis(mu)
    n_layers = len(layer_thetas)

    points_per_layer = n_points // n_layers
    used_points = points_per_layer * n_layers

    layers = []
    layer_ids = []

    for layer_id, theta in enumerate(layer_thetas):
        angles = np.linspace(0.0, 2.0 * np.pi, points_per_layer, endpoint=False)
        layer = (
            np.cos(theta) * mu[None, :]
            + np.sin(theta) * np.cos(angles)[:, None] * v1[None, :]
            + np.sin(theta) * np.sin(angles)[:, None] * v2[None, :]
        )
        layers.append(layer)
        layer_ids.extend([layer_id] * points_per_layer)

    Z = np.vstack(layers)
    layer_ids = np.asarray(layer_ids, dtype=np.int64)

    return (
        torch.as_tensor(Z, dtype=torch.float32),
        torch.as_tensor(layer_ids, dtype=torch.long),
        points_per_layer,
        used_points,
    )


def sample_vmf_approx(mu, kappa, n):
    """
    Approximate isotropic vMF-like samples on S^2.

    Uses Gaussian noise around mu followed by projection.
    """
    mu = np.asarray(mu, dtype=float)
    mu = mu / np.linalg.norm(mu)

    X = kappa * mu[None, :] + np.random.randn(n, 3)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return torch.as_tensor(X, dtype=torch.float32)


def sample_vmf_mixture_approx(mus, kappas, weights, n):
    """
    Sample from a mixture of approximate vMF-like components on S^2.
    """
    mus = [np.asarray(mu, dtype=float) / np.linalg.norm(mu) for mu in mus]
    kappas = np.asarray(kappas, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    comp_ids = np.random.choice(len(weights), size=n, p=weights)
    X = np.zeros((n, 3), dtype=float)

    for c in range(len(weights)):
        idx = np.where(comp_ids == c)[0]
        if len(idx) == 0:
            continue
        Xc = kappas[c] * mus[c][None, :] + np.random.randn(len(idx), 3)
        Xc = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
        X[idx] = Xc

    return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(comp_ids, dtype=torch.long)


# ============================================================
# Center utilities
# ============================================================

def resolve_n_centers(requested_n_centers, n_available):
    if requested_n_centers is None:
        return None
    return min(requested_n_centers, n_available)


def symmetric_ring_center_idx(points_per_layer, n_layers, requested_n_centers):
    """
    Choose centers equally across rings.
    """
    max_centers = points_per_layer * n_layers
    requested_n_centers = min(requested_n_centers, max_centers)

    centers_per_layer = requested_n_centers // n_layers
    used_centers = centers_per_layer * n_layers

    idx = []
    for layer in range(n_layers):
        start = layer * points_per_layer
        idx.extend(range(start, start + centers_per_layer))

    return torch.as_tensor(idx, dtype=torch.long), centers_per_layer, used_centers


# ============================================================
# Build artifact
# ============================================================

def build_experiment(args):
    experiment_name = args.experiment

    if experiment_name == "uniform_s2":
        Z = sample_uniform_S2(args.n_points)
        labels = None
        center_idx = None

        if args.n_centers is not None:
            n_centers = resolve_n_centers(args.n_centers, Z.shape[0])
            center_idx = torch.arange(n_centers, dtype=torch.long)

        metadata = {
            "experiment": experiment_name,
            "n_points_requested": args.n_points,
            "n_points_used": int(Z.shape[0]),
            "distribution": "uniform_on_S2",
        }

    elif experiment_name == "symmetric_vmf":
        mu = np.array([0.0, 0.0, 1.0])

        Z, labels, points_per_layer, used_points = sample_symmetric_ring_points(
            mu=mu,
            layer_thetas=args.layer_thetas,
            n_points=args.n_points,
        )

        center_idx = None
        centers_per_layer = None
        used_centers = None

        if args.n_centers is not None:
            center_idx, centers_per_layer, used_centers = symmetric_ring_center_idx(
                points_per_layer=points_per_layer,
                n_layers=len(args.layer_thetas),
                requested_n_centers=args.n_centers,
            )

        metadata = {
            "experiment": experiment_name,
            "distribution": "symmetric_ring_vmf_like",
            "mu": mu.tolist(),
            "layer_thetas": list(args.layer_thetas),
            "n_points_requested": args.n_points,
            "n_points_used": int(used_points),
            "points_per_layer": int(points_per_layer),
            "n_centers_requested": args.n_centers,
            "n_centers_used": None if center_idx is None else int(len(center_idx)),
            "centers_per_layer": None if centers_per_layer is None else int(centers_per_layer),
            "used_centers": None if used_centers is None else int(used_centers),
        }

    elif experiment_name == "isotropic_vmf":
        mu = np.array([0.0, 0.0, 1.0])
        Z = sample_vmf_approx(mu=mu, kappa=args.kappa, n=args.n_points)
        labels = None
        center_idx = None

        if args.n_centers is not None:
            n_centers = resolve_n_centers(args.n_centers, Z.shape[0])
            center_idx = torch.arange(n_centers, dtype=torch.long)

        metadata = {
            "experiment": experiment_name,
            "distribution": "isotropic_vmf_like",
            "mu": mu.tolist(),
            "kappa": float(args.kappa),
            "n_points_requested": args.n_points,
            "n_points_used": int(Z.shape[0]),
            "n_centers_requested": args.n_centers,
            "n_centers_used": None if center_idx is None else int(len(center_idx)),
        }

    elif experiment_name == "vmf_mixture":
        mus = [
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        kappas = [args.kappa, args.kappa, args.kappa]
        weights = [0.2, 0.7, 0.1]

        Z, labels = sample_vmf_mixture_approx(
            mus=mus,
            kappas=kappas,
            weights=weights,
            n=args.n_points,
        )

        center_idx = None
        if args.n_centers is not None:
            n_centers = resolve_n_centers(args.n_centers, Z.shape[0])
            center_idx = torch.arange(n_centers, dtype=torch.long)

        metadata = {
            "experiment": experiment_name,
            "distribution": "vmf_mixture_like",
            "mus": [mu.tolist() for mu in mus],
            "kappas": list(map(float, kappas)),
            "weights": list(map(float, weights)),
            "n_points_requested": args.n_points,
            "n_points_used": int(Z.shape[0]),
            "n_centers_requested": args.n_centers,
            "n_centers_used": None if center_idx is None else int(len(center_idx)),
        }

    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    C_g = compute_geodesics_S2(Z)

    return Z, C_g, labels, center_idx, metadata


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    artifact_dir = os.path.join(args.artifact_root, args.experiment)
    os.makedirs(artifact_dir, exist_ok=True)

    Z, C_g, labels, center_idx, metadata = build_experiment(args)

    n_points_used = int(Z.shape[0])

    canonical_order = torch.arange(n_points_used, dtype=torch.long)

    filename = f"points_and_geodesics_n{n_points_used}_seed{args.seed}.npz"
    out_path = os.path.join(artifact_dir, filename)

    save_dict = {
        "Z": Z.cpu().numpy().astype(np.float32),
        "C_g": C_g.cpu().numpy().astype(np.float32),
        "canonical_order": canonical_order.cpu().numpy().astype(np.int64),
    }

    if labels is not None:
        save_dict["labels"] = labels.cpu().numpy().astype(np.int64)

    if center_idx is not None:
        save_dict["center_idx"] = center_idx.cpu().numpy().astype(np.int64)

    np.savez_compressed(out_path, **save_dict)

    metadata["seed"] = int(args.seed)
    metadata["artifact_file"] = out_path
    metadata["z_shape"] = tuple(Z.shape)
    metadata["geodesic_shape"] = tuple(C_g.shape)
    metadata["num_points"] = n_points_used
    metadata["has_canonical_order"] = True
    metadata["npz_keys"] = list(save_dict.keys())

    meta_path = out_path.replace(".npz", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved S2 artifact to: {out_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Z shape: {tuple(Z.shape)}")
    print(f"C_g shape: {tuple(C_g.shape)}")
    if center_idx is not None:
        print(f"Number of centers: {len(center_idx)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["uniform_s2", "symmetric_vmf", "isotropic_vmf", "vmf_mixture"],
    )

    parser.add_argument("--artifact-root", type=str, default="artifacts/s2")
    parser.add_argument("--n-points", type=int, default=2000)
    parser.add_argument("--n-centers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--layer-thetas", type=float, nargs="+", default=[0.25, 0.50, 0.75])

    args = parser.parse_args()
    main(args)


# python build_S2_artifacts.py \
#   --experiment uniform_s2 \
#   --artifact-root artifacts/s2 \
#   --n-points 2000 \
#   --n-centers 1000 \
#   --seed 0