import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from colors import get_colors, get_single_color
from geodesics import compute_geodesics_S2
from LDD import LDD, signature_distance_matrix


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
    for theta in layer_thetas:
        angles = np.linspace(0.0, 2.0 * np.pi, points_per_layer, endpoint=False)
        layer = (
            np.cos(theta) * mu[None, :]
            + np.sin(theta) * np.cos(angles)[:, None] * v1[None, :]
            + np.sin(theta) * np.sin(angles)[:, None] * v2[None, :]
        )
        layers.append(layer)

    Z = np.vstack(layers)
    return torch.as_tensor(Z, dtype=torch.float32), points_per_layer, used_points


def sample_vmf_approx(mu, kappa, n):
    """
    Approximate isotropic vMF-like samples on S^2.

    Uses Gaussian noise around mu followed by projection.
    Good enough for experiments, but not exact vMF sampling.
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

    return torch.as_tensor(X, dtype=torch.float32)


def compute_summary(H, tau):
    """
    Compute summary measures from H.
    """
    D_H = signature_distance_matrix(H)
    mask_offdiag = ~torch.eye(D_H.shape[0], dtype=torch.bool)
    D_H_offdiag = D_H[mask_offdiag]

    pair_min = D_H_offdiag.min().item()
    pair_max = D_H_offdiag.max().item()
    pair_mean = D_H_offdiag.mean().item()
    pair_std = D_H_offdiag.std().item()
    pair_q95 = torch.quantile(D_H_offdiag, 0.95).item()
    frac_under_tau = float((D_H_offdiag <= tau).float().mean())

    H_mean = H.mean(dim=0, keepdim=True)
    dev = torch.sqrt(((H - H_mean) ** 2).mean(dim=1))

    dev_min = dev.min().item()
    dev_max = dev.max().item()
    dev_mean = dev.mean().item()
    dev_std = dev.std().item()

    return {
        "pair_min": pair_min,
        "pair_max": pair_max,
        "pair_mean": pair_mean,
        "pair_std": pair_std,
        "pair_q95": pair_q95,
        "frac_under_tau": frac_under_tau,
        "dev_min": dev_min,
        "dev_max": dev_max,
        "dev_mean": dev_mean,
        "dev_std": dev_std,
        "H_mean": H_mean,
    }


def print_summary(name, summary, tau):
    """
    Print table-friendly summary.
    """
    print(f"\nLDD summary ({name})")
    print("\nPairwise signature distances")
    print(f"min   = {summary['pair_min']:.6f}")
    print(f"max   = {summary['pair_max']:.6f}")
    print(f"mean  = {summary['pair_mean']:.6f}")
    print(f"std   = {summary['pair_std']:.6f}")
    print(f"95%   = {summary['pair_q95']:.6f}")
    print(f"<=tau = {summary['frac_under_tau']:.4f}  (tau = {tau:.3f})")

    print("\nDeviation from mean signature")
    print(f"min   = {summary['dev_min']:.6f}")
    print(f"max   = {summary['dev_max']:.6f}")
    print(f"mean  = {summary['dev_mean']:.6f}")
    print(f"std   = {summary['dev_std']:.6f}")


def plot_ldd_signatures(r, H, H_mean, savepath, title, n_plot=10):
    """
    Plot a subset of LDD signatures with the mean signature.
    """
    plt.figure(figsize=(6, 4))
    line_colors = get_colors(H.shape[0])

    for i in range(H.shape[0]):
        plt.plot(
            r.numpy(),
            H[i].numpy(),
            alpha=0.25,
            linewidth=1.8,
            color=line_colors[i]
        )

    plt.plot(
        r.numpy(),
        H_mean.squeeze(0).numpy(),
        color="grey",
        linewidth=1.5,
        linestyle="--",
        label="Mean signature"
    )

    plt.xlabel("Radius")
    plt.ylabel("Share of points within radius")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()


def save_ldd_checkpoint(experiment_name, H, r, center_idx, n_data_points, args):
    """
    Save LDD outputs and config for later reuse.
    """
    save_dir = os.path.join("checkpoints", "ldd", experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    tau_tag = str(args.tau).replace(".", "")
    n_centers_tag = len(center_idx)
    tau_tag = str(args.tau).replace(".", "")
    filename = f"H_n{n_data_points}_c{n_centers_tag}_rb{args.r_bins}_tau{tau_tag}.npz"
    
    np.savez(
        os.path.join(save_dir, filename),
        H=H.cpu().numpy(),
        r=r.cpu().numpy(),
        center_idx=center_idx.cpu().numpy(),
        n_points=args.n_points,
        n_centers=args.n_centers,
        r_bins=args.r_bins,
        tau=args.tau,
    )

    print(f"Saved checkpoint: {os.path.join(save_dir, filename)}")


def resolve_n_centers(requested_n_centers, n_available):
    """
    Clip requested number of centers to available points.
    """
    if requested_n_centers is None:
        return None
    return min(requested_n_centers, n_available)


def symmetric_ring_center_idx(points_per_layer, n_layers, requested_n_centers):
    """
    Choose centers equally across rings.

    Uses the largest number of centers that can be distributed equally
    across all rings.
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



# ---- Main experiment functions ----

def run_uniform_s2(args):
    """
    Uniform S^2 LDD computation.
    """
    Z = sample_uniform_S2(args.n_points)
    C_g = compute_geodesics_S2(Z)

    H, r, center_idx, z_points, C_centers = LDD(
        Z, C_g,
        r_bins=args.r_bins,
        r_max=np.pi,
        n_centers = resolve_n_centers(args.n_centers, Z.shape[0]),
    )
    
    save_ldd_checkpoint("uniform_s2", H, r, center_idx, Z.shape[0], args)

    summary = compute_summary(H, args.tau)
    print_summary("uniform S2", summary, args.tau)

    plot_ldd_signatures(
        r, H, summary["H_mean"],
        savepath=os.path.join(args.plot_dir, "uniform_s2_ldd.png"),
        title="LDD signatures (uniform on S2)",
        n_plot=args.n_plot,
    )


def run_symmetric_vmf(args):
    """
    Symmetric ring-based worst-case symmetry test.
    """
    mu = np.array([0.0, 0.0, 1.0])

    Z, points_per_layer, used_points = sample_symmetric_ring_points(
        mu=mu,
        layer_thetas=args.layer_thetas,
        n_points=args.n_points,
    )
    C_g = compute_geodesics_S2(Z)

    center_idx, centers_per_layer, used_centers = symmetric_ring_center_idx(
        points_per_layer=points_per_layer,
        n_layers=len(args.layer_thetas),
        requested_n_centers=args.n_centers,
    )

    H, r, center_idx, z_points, C_centers = LDD(
        Z, C_g,
        r_bins=args.r_bins,
        r_max=np.pi,
        center_idx=center_idx,
    )

    save_ldd_checkpoint("symmetric_vmf", H, r, center_idx, Z.shape[0], args)

    summary = compute_summary(H, args.tau)
    print_summary("symmetric vMF-like", summary, args.tau)

    print("Requested points:", args.n_points)
    print("Used points:", used_points)
    print("Points per layer:", points_per_layer)
    print("Requested centers:", args.n_centers)
    print("Used centers:", used_centers)
    print("Centers per layer:", centers_per_layer)

    plot_ldd_signatures(
        r, H, summary["H_mean"],
        savepath=os.path.join(args.plot_dir, "symmetric_vmf_ldd.png"),
        title="LDD signatures (symmetric vMF-like)",
        n_plot=args.n_plot,
    )


def run_isotropic_vmf(args):
    """
    Isotropic vMF-like sampling around one mode.
    """
    mu = np.array([0.0, 0.0, 1.0])
    Z = sample_vmf_approx(mu=mu, kappa=args.kappa, n=args.n_points)
    C_g = compute_geodesics_S2(Z)

    H, r, center_idx, z_points, C_centers = LDD(
        Z, C_g,
        r_bins=args.r_bins,
        r_max=np.pi,
        n_centers = resolve_n_centers(args.n_centers, Z.shape[0]),
    )

    save_ldd_checkpoint("isotropic_vmf", H, r, center_idx, Z.shape[0], args)

    summary = compute_summary(H, args.tau)
    print_summary("isotropic vMF-like", summary, args.tau)

    plot_ldd_signatures(
        r, H, summary["H_mean"],
        savepath=os.path.join(args.plot_dir, "isotropic_vmf_ldd.png"),
        title="LDD signatures (isotropic vMF-like)",
        n_plot=args.n_plot,
    )


def run_vmf_mixture(args):
    """
    Mixture of approximate vMF-like components.
    """
    mus = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    kappas = [args.kappa, args.kappa, args.kappa]
    weights = [0.2, 0.7, 0.1]

    Z = sample_vmf_mixture_approx(
        mus=mus,
        kappas=kappas,
        weights=weights,
        n=args.n_points,
    )
    C_g = compute_geodesics_S2(Z)

    H, r, center_idx, z_points, C_centers = LDD(
        Z, C_g,
        r_bins=args.r_bins,
        r_max=np.pi,
        n_centers = resolve_n_centers(args.n_centers, Z.shape[0]),
    )

    save_ldd_checkpoint("vmf_mixture", H, r, center_idx, Z.shape[0], args)

    summary = compute_summary(H, args.tau)
    print_summary("vMF mixture", summary, args.tau)

    plot_ldd_signatures(
        r, H, summary["H_mean"],
        savepath=os.path.join(args.plot_dir, "vmf_mixture_ldd.png"),
        title="LDD signatures (vMF mixture)",
        n_plot=args.n_plot,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["uniform_s2", "symmetric_vmf", "isotropic_vmf", "vmf_mixture"])
    parser.add_argument("--plot_dir", type=str, default="plots/ldd")
    parser.add_argument("--n_points", type=int, default=2000)
    parser.add_argument("--n_centers", type=int, default=1000)
    parser.add_argument("--r_bins", type=int, default=200)
    parser.add_argument("--tau", type=float, default=0.015)
    parser.add_argument("--n_plot", type=int, default=10)
    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--layer_thetas", type=float, nargs="+", default=[0.25, 0.50, 0.75])

    args = parser.parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)

    if args.experiment == "uniform_s2":
        run_uniform_s2(args)
    elif args.experiment == "symmetric_vmf":
        run_symmetric_vmf(args)
    elif args.experiment == "isotropic_vmf":
        run_isotropic_vmf(args)
    elif args.experiment == "vmf_mixture":
        run_vmf_mixture(args)