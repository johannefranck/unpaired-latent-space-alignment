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



def rotation_matrix_xyz(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx],
    ])

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ])

    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1],
    ])

    return Rz @ Ry @ Rx


def apply_rotation(Z, R):
    R_t = torch.tensor(R, dtype=Z.dtype)
    return Z @ R_t.T

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


def save_one_role(
    role,
    Z,
    C_g,
    labels,
    center_idx,
    canonical_order,
    metadata,
    args,
):
    n_points_used = int(Z.shape[0])

    data_dir = os.path.join(args.data_root, args.experiment)
    geodesic_dir = os.path.join(args.geodesic_root, args.experiment)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(geodesic_dir, exist_ok=True)

    data_path = os.path.join(
        data_dir,
        f"{role}_points_n{n_points_used}_seed{args.seed}_rotseed{args.rotation_seed}.npz",
    )

    geodesic_path = os.path.join(
        geodesic_dir,
        f"{role}_geodesics_n{n_points_used}_seed{args.seed}_rotseed{args.rotation_seed}.npz",
    )

    data_save = {
        "Z": Z.cpu().numpy().astype(np.float32),
        "canonical_order": canonical_order.cpu().numpy().astype(np.int64),
    }

    geodesic_save = {
        "Z": Z.cpu().numpy().astype(np.float32),
        "C_g": C_g.cpu().numpy().astype(np.float32),
        "canonical_order": canonical_order.cpu().numpy().astype(np.int64),
    }

    if labels is not None:
        data_save["labels"] = labels.cpu().numpy().astype(np.int64)
        geodesic_save["labels"] = labels.cpu().numpy().astype(np.int64)

    if center_idx is not None:
        geodesic_save["center_idx"] = center_idx.cpu().numpy().astype(np.int64)

    np.savez_compressed(data_path, **data_save)
    np.savez_compressed(geodesic_path, **geodesic_save)

    role_metadata = dict(metadata)
    role_metadata["role"] = role
    role_metadata["data_file"] = data_path
    role_metadata["geodesic_file"] = geodesic_path
    role_metadata["npz_keys_data"] = list(data_save.keys())
    role_metadata["npz_keys_geodesic"] = list(geodesic_save.keys())

    with open(data_path.replace(".npz", "_metadata.json"), "w") as f:
        json.dump(role_metadata, f, indent=2)

    with open(geodesic_path.replace(".npz", "_metadata.json"), "w") as f:
        json.dump(role_metadata, f, indent=2)

    print(f"Saved {role} data to: {data_path}")
    print(f"Saved {role} geodesics to: {geodesic_path}")


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Z_A, C_A, labels_A, center_idx_A, metadata = build_experiment(args)

    n_points_used = int(Z_A.shape[0])
    canonical_A = torch.arange(n_points_used, dtype=torch.long)

    R_true = rotation_matrix_xyz(
        args.rotation_angles[0],
        args.rotation_angles[1],
        args.rotation_angles[2],
    )

    Z_B = apply_rotation(Z_A, R_true)

    torch.manual_seed(args.rotation_seed)
    perm_B = torch.randperm(n_points_used)

    Z_B = Z_B[perm_B]
    labels_B = None if labels_A is None else labels_A[perm_B]
    canonical_B = canonical_A[perm_B]

    C_B = compute_geodesics_S2(Z_B)

    center_idx_B = None
    if center_idx_A is not None:
        center_idx_B = torch.arange(min(args.n_centers, n_points_used), dtype=torch.long)

    metadata["seed"] = int(args.seed)
    metadata["rotation_seed"] = int(args.rotation_seed)
    metadata["rotation_angles_xyz"] = list(map(float, args.rotation_angles))
    metadata["R_true"] = R_true.tolist()
    metadata["B_is_rotated_and_shuffled_A"] = True
    metadata["num_points"] = n_points_used
    metadata["z_shape"] = tuple(Z_A.shape)
    metadata["geodesic_shape"] = tuple(C_A.shape)

    save_one_role(
        role="A",
        Z=Z_A,
        C_g=C_A,
        labels=labels_A,
        center_idx=center_idx_A,
        canonical_order=canonical_A,
        metadata=metadata,
        args=args,
    )

    save_one_role(
        role="B",
        Z=Z_B,
        C_g=C_B,
        labels=labels_B,
        center_idx=center_idx_B,
        canonical_order=canonical_B,
        metadata=metadata,
        args=args,
    )

    print("Saved controlled S2 pair.")
    print("A = sampled points")
    print("B = rotated and shuffled A")
    print("R_true:")
    print(R_true)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["uniform_s2", "symmetric_vmf", "isotropic_vmf", "vmf_mixture"],
    )

    parser.add_argument("--n-points", type=int, default=2000)
    parser.add_argument("--n-centers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data-root", type=str, default="data/s2")
    parser.add_argument("--geodesic-root", type=str, default="artifacts/s2_geodesics")

    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--layer-thetas", type=float, nargs="+", default=[0.25, 0.50, 0.75])

    parser.add_argument("--rotation-seed", type=int, default=123)
    parser.add_argument(
        "--rotation-angles",
        type=float,
        nargs=3,
        default=[0.7, -0.4, 1.1],
    )

    args = parser.parse_args()
    main(args)


# python build_S2_artifacts.py \
#   --experiment vmf_mixture \
#   --data-root data/s2 \
#   --geodesic-root artifacts/s2_geodesics \
#   --n-points 2000 \
#   --n-centers 1000 \
#   --seed 1 \
#   --rotation-seed 123