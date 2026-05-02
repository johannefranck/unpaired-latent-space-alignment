import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from colors import get_single_color


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
    # r = torch.linspace(0.0, float(r_max), steps=r_bins, dtype=torch.float32) # linear spaced r bins
    r_min = 1e-4
    r = torch.logspace(
        np.log10(r_min),
        np.log10(float(r_max)),
        steps=r_bins,
        dtype=torch.float32,
    )

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


def ldd_variation_diagnostics(H, eps=1e-12):
    """
    Measure variation and effective rank of LDD signatures.
    """
    H = torch.as_tensor(H, dtype=torch.float32)

    n, B = H.shape
    H_mean = H.mean(dim=0, keepdim=True)
    Hc = H - H_mean

    # Magnitude of LDD variation
    ldd_global_std = torch.sqrt((Hc ** 2).mean()).item()

    # Covariance over radius bins
    Cov = (Hc.T @ Hc) / max(n - 1, 1)

    eigvals = torch.linalg.eigvalsh(Cov)
    eigvals = torch.clamp(eigvals, min=0.0)

    total_var = eigvals.sum().item()

    if total_var > eps:
        p = eigvals / eigvals.sum()
        effective_rank = torch.exp(-(p * torch.log(p + eps)).sum()).item()
        top_eig_frac = (eigvals[-1] / eigvals.sum()).item()
    else:
        effective_rank = 0.0
        top_eig_frac = 0.0

    rank_cap = max(1, min(n - 1, B))

    return {
        "n_ldds": int(n),
        "r_bins": int(B),
        "rank_cap": int(rank_cap),
        "ldd_global_std": float(ldd_global_std),
        "total_variance": float(total_var),
        "effective_rank": float(effective_rank),
        "effective_rank_pct": float(effective_rank / rank_cap),
        "top_eig_frac": float(top_eig_frac),
    }



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

    variation = ldd_variation_diagnostics(H)

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
        "ldd_variation": variation,
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

    print("\nLDD variation")
    print(f"n_ldds              = {summary['ldd_variation']['n_ldds']}")
    print(f"r_bins              = {summary['ldd_variation']['r_bins']}")
    print(f"rank cap            = {summary['ldd_variation']['rank_cap']}")
    print(f"global std          = {summary['ldd_variation']['ldd_global_std']:.6f}")
    print(f"total variance      = {summary['ldd_variation']['total_variance']:.6f}")
    print(f"effective rank      = {summary['ldd_variation']['effective_rank']:.6f}")
    print(f"effective rank pct  = {summary['ldd_variation']['effective_rank_pct']:.6f}")
    print(f"top eig frac        = {summary['ldd_variation']['top_eig_frac']:.6f}")

def plot_ldd_signatures(r, H, H_mean, savepath, title):
    """
    Plot all LDD signatures with the mean signature.
    """
    plt.figure(figsize=(6, 4))
    curve_color = get_single_color(6)

    for i in range(H.shape[0]):
        plt.plot(
            r.numpy(),
            H[i].numpy(),
            alpha=0.25,
            linewidth=1.8,
            color=curve_color,
        )

    plt.plot(
        r.numpy(),
        H_mean.squeeze(0).numpy(),
        color="grey",
        linewidth=1.5,
        linestyle="--",
        label="Mean signature",
    )

    plt.xlabel("Radius")
    plt.ylabel("Share of points within radius")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()


def resolve_n_centers(requested_n_centers, n_available):
    """
    Clip requested number of centers to available points.
    """
    if requested_n_centers is None:
        return None
    return min(requested_n_centers, n_available)


def load_s2_artifact(npz_path):
    """
    Load saved S2 points + geodesics artifact.
    """
    data = np.load(npz_path)

    payload = {
        "Z": torch.from_numpy(data["Z"]).float(),
        "C_g": torch.from_numpy(data["C_g"]).float(),
        "canonical_order": torch.from_numpy(data["canonical_order"]).long(),
        "labels": torch.from_numpy(data["labels"]).long() if "labels" in data.files else None,
        "center_idx": torch.from_numpy(data["center_idx"]).long() if "center_idx" in data.files else None,
    }
    return payload


def build_artifact_path(args):
    """
    Construct default artifact path from experiment metadata.
    """
    filename = f"points_and_geodesics_n{args.n_points}_seed{args.seed}.npz"
    return os.path.join(args.artifact_root, args.experiment, filename)


def build_ldd_output_paths(args, n_points, n_centers):
    """
    Build output paths for LDD artifact and metadata.
    """
    tau_tag = str(args.tau).replace(".", "")
    filename = f"ldd_H_n{n_points}_c{n_centers}_rb{args.r_bins}_tau{tau_tag}.npz"

    ldd_dir = os.path.join(args.artifact_root, args.experiment)
    os.makedirs(ldd_dir, exist_ok=True)

    ldd_path = os.path.join(ldd_dir, filename)
    meta_path = ldd_path.replace(".npz", "_metadata.json")

    return ldd_path, meta_path


def build_plot_path(args, n_points, n_centers):
    """
    Build plot path.
    """
    os.makedirs(args.plot_root, exist_ok=True)

    tau_tag = str(args.tau).replace(".", "")
    filename = (
        f"{args.experiment}_ldd_signatures_"
        f"n{n_points}_c{n_centers}_rb{args.r_bins}_tau{tau_tag}.png"
    )
    return os.path.join(args.plot_root, filename)


def save_summary_json(summary, tau, out_path, experiment):
    """
    Save LDD summary in JSON-friendly format.
    """
    json_summary = {
        "experiment": experiment,
        "tau": float(tau),

        "pairwise_signature_distances": {
            "min": float(summary["pair_min"]),
            "max": float(summary["pair_max"]),
            "mean": float(summary["pair_mean"]),
            "std": float(summary["pair_std"]),
            "q95": float(summary["pair_q95"]),
            "frac_under_tau": float(summary["frac_under_tau"]),
        },

        "deviation_from_mean_signature": {
            "min": float(summary["dev_min"]),
            "max": float(summary["dev_max"]),
            "mean": float(summary["dev_mean"]),
            "std": float(summary["dev_std"]),
        },

        "ldd_variation": {
            "n_ldds": int(summary["ldd_variation"]["n_ldds"]),
            "r_bins": int(summary["ldd_variation"]["r_bins"]),
            "rank_cap": int(summary["ldd_variation"]["rank_cap"]),
            "ldd_global_std": float(summary["ldd_variation"]["ldd_global_std"]),
            "total_variance": float(summary["ldd_variation"]["total_variance"]),
            "effective_rank": float(summary["ldd_variation"]["effective_rank"]),
            "effective_rank_pct": float(summary["ldd_variation"]["effective_rank_pct"]),
            "top_eig_frac": float(summary["ldd_variation"]["top_eig_frac"]),
        },
    }

    with open(out_path, "w") as f:
        json.dump(json_summary, f, indent=2)

def main(args):
    if args.artifact_file is not None:
        artifact_path = args.artifact_file
    else:
        artifact_path = build_artifact_path(args)

    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Could not find artifact file: {artifact_path}")

    payload = load_s2_artifact(artifact_path)

    Z = payload["Z"]
    C_g = payload["C_g"]
    saved_center_idx = payload["center_idx"]

    n_points = Z.shape[0]

    # Center choice logic:
    # 1) use saved center_idx if present and no override requested
    # 2) otherwise use requested n_centers
    if saved_center_idx is not None and not args.override_centers:
        center_idx = saved_center_idx
    else:
        center_idx = None
        if args.n_centers is not None:
            n_centers = resolve_n_centers(args.n_centers, n_points)
            center_idx = torch.arange(n_centers, dtype=torch.long)

    H, r, center_idx, z_points, C_centers = LDD(
        Z,
        C_g,
        r_bins=args.r_bins,
        r_max=np.pi,
        center_idx=center_idx,
    )

    summary = compute_summary(H, args.tau)

    print_summary(args.experiment, summary, args.tau)

    n_centers_used = len(center_idx)
    ldd_out_path, meta_out_path = build_ldd_output_paths(args, n_points, n_centers_used)

    summary_path = ldd_out_path.replace(".npz", "_summary.json")

    save_summary_json(
        summary,
        args.tau,
        summary_path,
        args.experiment,
    )

    np.savez_compressed(
        ldd_out_path,
        H=H.cpu().numpy().astype(np.float32),
        r=r.cpu().numpy().astype(np.float32),
        center_idx=center_idx.cpu().numpy().astype(np.int64),
        n_points=np.int64(n_points),
        n_centers=np.int64(n_centers_used),
        r_bins=np.int64(args.r_bins),
        tau=np.float32(args.tau),
        artifact_file=np.array(artifact_path),
    )

    ldd_metadata = {
        "experiment": args.experiment,
        "artifact_file": artifact_path,
        "ldd_file": ldd_out_path,
        "n_points": int(n_points),
        "n_centers": int(n_centers_used),
        "r_bins": int(args.r_bins),
        "tau": float(args.tau),
        "used_saved_center_idx": bool(saved_center_idx is not None and not args.override_centers),
        "override_centers": bool(args.override_centers),
        "npz_keys": ["H", "r", "center_idx", "n_points", "n_centers", "r_bins", "tau", "artifact_file"],
    }

    with open(meta_out_path, "w") as f:
        json.dump(ldd_metadata, f, indent=2)

    plot_path = build_plot_path(args, n_points, n_centers_used)
    plot_ldd_signatures(
        r,
        H,
        summary["H_mean"],
        savepath=plot_path,
        title=f"LDD signatures ({args.experiment})",
    )

    print(f"Loaded artifact: {artifact_path}")
    print(f"Saved LDD file to: {ldd_out_path}")
    print(f"Saved LDD metadata to: {meta_out_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["uniform_s2", "symmetric_vmf", "isotropic_vmf", "vmf_mixture"],
    )

    parser.add_argument("--artifact-root", type=str, default="artifacts/s2")
    parser.add_argument("--artifact-file", type=str, default=None)

    parser.add_argument("--plot-root", type=str, default="plots/s2/ldd")

    parser.add_argument("--n-points", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--n-centers", type=int, default=None)
    parser.add_argument("--override-centers", action="store_true")

    parser.add_argument("--r-bins", type=int, default=200)
    parser.add_argument("--tau", type=float, default=0.015)

    args = parser.parse_args()
    main(args)


# Example:
# python LDD_run.py \
#   --experiment uniform_s2 \
#   --artifact-root artifacts/s2 \
#   --plot-root plots/s2/ldd \
#   --n-points 2000 \
#   --seed 0 \
#   --r-bins 200 \
#   --tau 0.015