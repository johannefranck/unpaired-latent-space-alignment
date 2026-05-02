# LDD_sensitivity_s2.py

import os
import csv
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import ot

try:
    from colors import get_single_color, color_segment
except ImportError:
    def get_single_color(i):
        cols = ["tab:cyan", "tab:blue", "tab:purple", "tab:orange", "tab:green"]
        return cols[i % len(cols)]

    def color_segment():
        return "viridis"


# ============================================================
# S2 utilities
# ============================================================

def normalize_rows(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def random_rotation_matrix(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)

    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    Q = Q @ np.diag(signs)

    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0

    return Q


def rotation_from_north(mu):
    mu = np.asarray(mu, dtype=np.float64)
    mu = mu / (np.linalg.norm(mu) + 1e-12)

    north = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    v = np.cross(north, mu)
    c = float(np.dot(north, mu))
    s = float(np.linalg.norm(v))

    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])

    K = np.array([
        [0.0, -v[2],  v[1]],
        [v[2],  0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)

    return np.eye(3) + K + K @ K * ((1.0 - c) / (s ** 2))


# ============================================================
# Sampling
# ============================================================

def sample_uniform_s2(n, rng):
    return normalize_rows(rng.normal(size=(n, 3)))


def sample_vmf_s2(mu, kappa, n, rng):
    mu = np.asarray(mu, dtype=np.float64)
    mu = mu / (np.linalg.norm(mu) + 1e-12)

    if kappa <= 1e-12:
        return sample_uniform_s2(n, rng)

    u = rng.uniform(size=n)

    # On S2, w = cos(theta) has density proportional to exp(kappa*w).
    w = (1.0 / kappa) * np.log(
        np.exp(-kappa) + u * (np.exp(kappa) - np.exp(-kappa))
    )

    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radial = np.sqrt(np.maximum(1.0 - w ** 2, 0.0))

    X_north = np.stack([
        radial * np.cos(phi),
        radial * np.sin(phi),
        w,
    ], axis=1)

    R = rotation_from_north(mu)
    return normalize_rows(X_north @ R.T)


def sample_isotropic_vmf(n, rng):
    return sample_vmf_s2(
        mu=np.array([0.0, 0.0, 1.0]),
        kappa=8.0,
        n=n,
        rng=rng,
    )


def sample_vmf_mixture(n, rng):
    mus = np.array([
        [1.0,  0.0,  0.0],
        [-1.0, 0.0,  0.0],
        [0.0,  1.0,  0.0],
        [0.0, -1.0,  0.0],
    ], dtype=np.float64)

    kappas = np.array([18.0, 18.0, 18.0, 18.0], dtype=np.float64)
    weights = np.ones(len(mus), dtype=np.float64) / len(mus)

    labels = rng.choice(len(mus), size=n, p=weights)
    Z = np.zeros((n, 3), dtype=np.float64)

    for k in range(len(mus)):
        idx = np.where(labels == k)[0]
        if len(idx) > 0:
            Z[idx] = sample_vmf_s2(
                mu=mus[k],
                kappa=kappas[k],
                n=len(idx),
                rng=rng,
            )

    return normalize_rows(Z)


def sample_base_distribution(case, n, rng):
    if case == "uniform_s2":
        return sample_uniform_s2(n, rng)

    if case == "isotropic_vmf":
        return sample_isotropic_vmf(n, rng)

    if case == "vmf_mixture":
        return sample_vmf_mixture(n, rng)

    raise ValueError(f"Unknown case: {case}")


def sample_independent_pair(case, n, seed_a, seed_b, seed_rot):
    """
    Independent sampling:

        A_i ~ mu
        B_j = R* B0_j, where B0_j ~ mu independently.

    There is no pointwise correspondence.
    """
    rng_a = np.random.default_rng(seed_a)
    rng_b = np.random.default_rng(seed_b)
    rng_rot = np.random.default_rng(seed_rot)

    Z_a = sample_base_distribution(case, n, rng_a)
    Z_b0 = sample_base_distribution(case, n, rng_b)

    R_star = random_rotation_matrix(rng_rot)
    Z_b = normalize_rows(Z_b0 @ R_star.T)

    return Z_a, Z_b, R_star


# ============================================================
# Distances and LDDs
# ============================================================

def distance_geo_arccos(Z):
    Z = normalize_rows(Z)
    G = np.clip(Z @ Z.T, -1.0, 1.0)
    return np.arccos(G)


def distance_euc_chord(Z):
    diff = Z[:, None, :] - Z[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def distance_eol_linear_energy(Z):
    return distance_euc_chord(Z) ** 2


def compute_distance_matrix(Z, distance_type):
    if distance_type == "geo":
        return distance_geo_arccos(Z)

    if distance_type == "euc":
        return distance_euc_chord(Z)

    if distance_type == "eol":
        return distance_eol_linear_energy(Z)

    raise ValueError(f"Unknown distance_type: {distance_type}")


def make_log_radii(r_min, r_max, r_bins):
    if r_min <= 0:
        raise ValueError("r_min must be positive.")
    if r_max <= r_min:
        raise ValueError("r_max must be larger than r_min.")

    return np.logspace(
        np.log10(float(r_min)),
        np.log10(float(r_max)),
        num=int(r_bins),
        dtype=np.float64,
    )


def compute_ldd(D, radii):
    """
    Cumulative metric-ball LDD:

        H_i(r) = #{j != i : D_ij <= r} / (n - 1)

    Efficient implementation using sorted distances.
    """
    D = np.asarray(D, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"D must have shape (n,n), got {D.shape}")

    D_work = D.copy()
    np.fill_diagonal(D_work, np.inf)

    D_sorted = np.sort(D_work, axis=1)

    H = np.empty((n, len(radii)), dtype=np.float32)

    for i in range(n):
        H[i] = np.searchsorted(D_sorted[i], radii, side="right") / (n - 1)

    return torch.from_numpy(H).float()


def compute_ldd_cost(H_a, H_b):
    """
    Raw LDD cross-cost:

        M_ij = ||H_a[i] - H_b[j]||_2^2
    """
    H_a = torch.as_tensor(H_a, dtype=torch.float32)
    H_b = torch.as_tensor(H_b, dtype=torch.float32)

    return torch.cdist(H_a, H_b, p=2) ** 2


# ============================================================
# Diagnostics
# ============================================================

def effective_rank_and_entropy_from_energy(energy, eps=1e-12):
    energy = torch.as_tensor(energy, dtype=torch.float64)
    energy = torch.clamp(energy, min=0.0)

    total = energy.sum()
    if total <= eps:
        return np.nan, np.nan

    p = energy / total
    entropy = -(p * torch.log(p + eps)).sum()
    effective_rank = torch.exp(entropy)

    return float(effective_rank.item()), float(entropy.item())


def ldd_diagnostics(H, variance_floor):
    """
    LDD diagnostics.

    ldd_variance:
        amplitude of LDD variation. On S2 this mainly reflects density variation.

    ldd_effective_rank:
        spectral dimensionality of centered LDD curves.

    ldd_spectral_entropy:
        entropy of the centered LDD covariance spectrum.
    """
    H = torch.as_tensor(H, dtype=torch.float32)
    n, _ = H.shape

    Hc = H - H.mean(dim=0, keepdim=True)
    ldd_variance = float((Hc ** 2).mean().item())

    if ldd_variance < variance_floor:
        return {
            "ldd_variance": ldd_variance,
            "ldd_effective_rank": np.nan,
            "ldd_spectral_entropy": np.nan,
        }

    Cov = (Hc.T @ Hc) / max(n - 1, 1)
    eigvals = torch.linalg.eigvalsh(Cov)
    eigvals = torch.clamp(eigvals, min=0.0)

    effective_rank, spectral_entropy = effective_rank_and_entropy_from_energy(eigvals)

    return {
        "ldd_variance": ldd_variance,
        "ldd_effective_rank": effective_rank,
        "ldd_spectral_entropy": spectral_entropy,
    }


def average_ldd_diagnostics(diag_a, diag_b):
    out = {}

    for key in diag_a.keys():
        a = diag_a[key]
        b = diag_b[key]

        if not (np.isfinite(a) and np.isfinite(b)):
            out[f"{key}_mean"] = np.nan
        else:
            out[f"{key}_mean"] = 0.5 * (a + b)

    return out


def init_sinkhorn(M, epsilon, num_itermax):
    M_np = torch.as_tensor(M, dtype=torch.float64).detach().cpu().numpy()

    n_a, n_b = M_np.shape
    a = np.ones(n_a, dtype=np.float64) / n_a
    b = np.ones(n_b, dtype=np.float64) / n_b

    P = ot.sinkhorn(
        a,
        b,
        M_np,
        reg=float(epsilon),
        numItermax=int(num_itermax),
    )

    return torch.from_numpy(P).float()


def coupling_diagnostics(P, M, eps=1e-12):
    """
    Coupling diagnostics.

    Q:
        row-normalized coupling.

    R:
        residual from uniform row coupling.

    R_energy:
        coupling response strength / non-uniformity.

    LDD_gain:
        cost improvement relative to uniform coupling.
    """
    P = torch.as_tensor(P, dtype=torch.float32)
    M = torch.as_tensor(M, dtype=torch.float32)

    n_a, n_b = P.shape

    row_mass = P.sum(dim=1, keepdim=True)
    Q = P / (row_mass + eps)

    uniform_value = 1.0 / n_b
    U_row = torch.full_like(Q, uniform_value)
    R = Q - U_row

    row_peak = Q.max(dim=1).values
    peak_ratio = row_peak / uniform_value

    row_entropy = -(Q * torch.log(Q + eps)).sum(dim=1)
    max_entropy = np.log(n_b)
    inverse_entropy = max_entropy / (row_entropy + eps)

    R_energy = float((R ** 2).sum().item() / n_a)

    P_uniform = torch.ones_like(P) / (n_a * n_b)

    cost_P = float((P * M).sum().item())
    cost_uniform = float((P_uniform * M).sum().item())

    if cost_uniform > eps:
        LDD_gain = 1.0 - cost_P / cost_uniform
    else:
        LDD_gain = np.nan

    return {
        "Q_peak_ratio_mean": float(peak_ratio.mean().item()),
        "Q_peak_ratio_std": float(peak_ratio.std().item()),
        "Q_inverse_entropy_mean": float(inverse_entropy.mean().item()),
        "Q_inverse_entropy_std": float(inverse_entropy.std().item()),
        "R_energy": float(R_energy),
        "LDD_cost_P": float(cost_P),
        "LDD_cost_uniform": float(cost_uniform),
        "LDD_gain_over_uniform": float(LDD_gain),
    }


# ============================================================
# Plotting
# ============================================================

def legend_specs():
    case_colors = {
        "uniform_s2": get_single_color(0),
        "isotropic_vmf": get_single_color(2),
        "vmf_mixture": get_single_color(8),
    }

    case_labels = {
        "uniform_s2": "Uniform",
        "isotropic_vmf": "VMF",
        "vmf_mixture": "Mixture-VMF",
    }

    distance_markers = {
        "geo": "o",
        "euc": "s",
        "eol": "^",
    }

    distance_labels = {
    "geo": "geodesic",
    "euc": "euclidean",
    "eol": "linear energy approximation",
    }

    return case_colors, case_labels, distance_markers, distance_labels


def add_legend(fig, case_colors, case_labels, distance_markers, distance_labels, y):
    case_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=9,
            label=case_labels[key],
        )
        for key, color in case_colors.items()
    ]

    distance_handles = [
        plt.Line2D(
            [0], [0],
            marker=marker,
            linestyle="none",
            color="black",
            markersize=9,
            label=distance_labels[key],
        )
        for key, marker in distance_markers.items()
    ]

    fig.legend(
        handles=case_handles + distance_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=6,
        frameon=False,
        fontsize=10.5,
        handletextpad=0.55,
        columnspacing=1.1,
    )


def plot_sphere_ldd_coupling_control(rows, save_path):
    """
    Main control plot:
        shows what the LDD/coupling diagnostics do on S2.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metrics = [
        ("ldd_variance_mean", "LDD variance\n(density-amplitude signal)"),
        ("ldd_effective_rank_mean", "LDD effective rank\n(LDD spectral complexity)"),
        ("ldd_spectral_entropy_mean", "LDD spectral entropy\n(LDD spectral complexity)"),
        ("R_energy", "Coupling response\n$E_R$"),
        ("Q_peak_ratio_mean", "Coupling peak ratio\n(row sharpness)"),
        ("LDD_gain_over_uniform", "LDD-cost gain\nover uniform coupling"),
    ]

    case_colors, case_labels, distance_markers, distance_labels = legend_specs()

    x_positions = {
        "uniform_s2": 0,
        "isotropic_vmf": 1,
        "vmf_mixture": 2,
    }

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), squeeze=False)

    for ax, (key, title) in zip(axes.ravel(), metrics):
        for row in rows:
            y_val = float(row.get(key, np.nan))
            if not np.isfinite(y_val):
                continue

            x_base = x_positions[row["case"]]
            marker_offset = {"geo": -0.08, "euc": 0.0, "eol": 0.08}[row["distance_type"]]
            rep_offset = 0.012 * int(row["rep"])

            ax.scatter(
                x_base + marker_offset + rep_offset,
                y_val,
                s=58,
                alpha=0.85,
                color=case_colors[row["case"]],
                marker=distance_markers[row["distance_type"]],
                edgecolors="black",
                linewidths=0.25,
            )

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Uniform", "VMF", "Mixture-VMF"], rotation=15)
        ax.set_title(title, fontsize=12.5, pad=9)
        ax.grid(alpha=0.22)

    fig.suptitle(
        "Sphere control: LDD diagnostics and coupling response",
        fontsize=18,
        y=0.982,
    )

    add_legend(fig, case_colors, case_labels, distance_markers, distance_labels, y=0.935)

    fig.subplots_adjust(
        top=0.84,
        bottom=0.085,
        left=0.06,
        right=0.985,
        hspace=0.44,
        wspace=0.28,
    )

    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_sphere_ldd_coupling_relationships(rows, save_path):
    """
    Relationship plot:
        separates amplitude effects from spectral-complexity effects.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pairs = [
        (
            "ldd_variance_mean",
            "R_energy",
            "LDD variance",
            "Coupling response $E_R$",
            "Density-amplitude signal vs coupling response",
        ),
        (
            "ldd_variance_mean",
            "LDD_gain_over_uniform",
            "LDD variance",
            "LDD-cost gain over uniform coupling",
            "Density-amplitude signal vs cost usage",
        ),
        (
            "ldd_effective_rank_mean",
            "R_energy",
            "LDD effective rank",
            "Coupling response $E_R$",
            "LDD rank vs coupling response",
        ),
        (
            "ldd_spectral_entropy_mean",
            "R_energy",
            "LDD spectral entropy",
            "Coupling response $E_R$",
            "LDD spectral entropy vs coupling response",
        ),
    ]

    case_colors, case_labels, distance_markers, distance_labels = legend_specs()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), squeeze=False)

    for ax, (x_key, y_key, x_label, y_label, title) in zip(axes.ravel(), pairs):
        for row in rows:
            x_val = float(row.get(x_key, np.nan))
            y_val = float(row.get(y_key, np.nan))

            if not (np.isfinite(x_val) and np.isfinite(y_val)):
                continue

            ax.scatter(
                x_val,
                y_val,
                s=58,
                alpha=0.85,
                color=case_colors[row["case"]],
                marker=distance_markers[row["distance_type"]],
                edgecolors="black",
                linewidths=0.25,
            )

        ax.set_xlabel(x_label, fontsize=11.5)
        ax.set_ylabel(y_label, fontsize=11.5)
        ax.set_title(title, fontsize=12.2, pad=9)
        ax.grid(alpha=0.22)

    fig.suptitle(
        "Which LDD diagnostics explain the coupling response?",
        fontsize=18,
        y=0.982,
    )

    add_legend(fig, case_colors, case_labels, distance_markers, distance_labels, y=0.935)

    fig.subplots_adjust(
        top=0.83,
        bottom=0.085,
        left=0.075,
        right=0.985,
        hspace=0.40,
        wspace=0.30,
    )

    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_ldd_curves(radii, H_a, H_b, save_path, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    radii = np.asarray(radii)
    H_a = torch.as_tensor(H_a)
    H_b = torch.as_tensor(H_b)

    col_a = get_single_color(6)
    col_b = get_single_color(2)

    plt.figure(figsize=(6, 4))

    n_plot = min(150, H_a.shape[0], H_b.shape[0])
    for i in range(n_plot):
        plt.plot(radii, H_a[i].numpy(), color=col_a, alpha=0.08, linewidth=0.8)
        plt.plot(radii, H_b[i].numpy(), color=col_b, alpha=0.08, linewidth=0.8)

    plt.plot(radii, H_a.mean(dim=0).numpy(), color=col_a, linewidth=2.2, label="A mean")
    plt.plot(radii, H_b.mean(dim=0).numpy(), color=col_b, linewidth=2.2, label="B mean")

    plt.xscale("log")
    plt.xlabel("Radius")
    plt.ylabel("Share inside metric ball")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close()


def plot_coupling(P, save_path, title, eps=1e-12):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    P = torch.as_tensor(P)
    P_np = np.log10(P.detach().cpu().numpy() + eps)

    plt.figure(figsize=(5, 4))
    plt.imshow(P_np, aspect="auto", cmap=color_segment())
    plt.colorbar(label="log10(P)")
    plt.xlabel("Target points B")
    plt.ylabel("Source points A")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close()


# ============================================================
# Saving
# ============================================================

def save_summary_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    keys = sorted({key for row in rows for key in row.keys()})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_grouped_summary_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    group_keys = ["case", "distance_type"]
    metrics = [
        "ldd_variance_mean",
        "ldd_effective_rank_mean",
        "ldd_spectral_entropy_mean",
        "R_energy",
        "Q_peak_ratio_mean",
        "Q_inverse_entropy_mean",
        "LDD_gain_over_uniform",
    ]

    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out_rows = []
    for key, group in grouped.items():
        out = {
            "case": key[0],
            "distance_type": key[1],
            "n": len(group),
        }

        for metric in metrics:
            vals = np.array(
                [float(g.get(metric, np.nan)) for g in group],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                out[f"{metric}_mean"] = np.nan
                out[f"{metric}_std"] = np.nan
            else:
                out[f"{metric}_mean"] = float(vals.mean())
                out[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        out_rows.append(out)

    keys = sorted({key for row in out_rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)


def save_interpretation_note(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)

    def mean_metric(case, metric):
        vals = []
        for row in grouped.get(case, []):
            value = float(row.get(metric, np.nan))
            if np.isfinite(value):
                vals.append(value)
        return float(np.mean(vals)) if vals else np.nan

    values = {
        "uniform_var": mean_metric("uniform_s2", "ldd_variance_mean"),
        "uniform_R": mean_metric("uniform_s2", "R_energy"),
        "uniform_peak": mean_metric("uniform_s2", "Q_peak_ratio_mean"),
        "vmf_var": mean_metric("isotropic_vmf", "ldd_variance_mean"),
        "vmf_rank": mean_metric("isotropic_vmf", "ldd_effective_rank_mean"),
        "vmf_entropy": mean_metric("isotropic_vmf", "ldd_spectral_entropy_mean"),
        "vmf_R": mean_metric("isotropic_vmf", "R_energy"),
        "vmf_peak": mean_metric("isotropic_vmf", "Q_peak_ratio_mean"),
        "mix_var": mean_metric("vmf_mixture", "ldd_variance_mean"),
        "mix_rank": mean_metric("vmf_mixture", "ldd_effective_rank_mean"),
        "mix_entropy": mean_metric("vmf_mixture", "ldd_spectral_entropy_mean"),
        "mix_R": mean_metric("vmf_mixture", "R_energy"),
        "mix_peak": mean_metric("vmf_mixture", "Q_peak_ratio_mean"),
    }

    text = f"""# Interpretation note: S2 LDD/coupling control

On S2, the intrinsic curvature is constant. Hence pointwise LDD variation is not expected to reflect point-dependent curvature variation. In this control experiment, the LDD signal mainly reflects sampling density variation.

Mean values across distance types and repetitions:

## Uniform
- LDD variance: {values["uniform_var"]:.6e}
- Coupling response E_R: {values["uniform_R"]:.6e}
- Coupling peak ratio: {values["uniform_peak"]:.6f}

## One-mode vMF
- LDD variance: {values["vmf_var"]:.6e}
- LDD effective rank: {values["vmf_rank"]:.6f}
- LDD spectral entropy: {values["vmf_entropy"]:.6f}
- Coupling response E_R: {values["vmf_R"]:.6e}
- Coupling peak ratio: {values["vmf_peak"]:.6f}

## Mixture-vMF
- LDD variance: {values["mix_var"]:.6e}
- LDD effective rank: {values["mix_rank"]:.6f}
- LDD spectral entropy: {values["mix_entropy"]:.6f}
- Coupling response E_R: {values["mix_R"]:.6e}
- Coupling peak ratio: {values["mix_peak"]:.6f}

## Suggested write-up

On S2, the LDD signatures form a density-dominated control case. Since the sphere has constant curvature, pointwise LDD variation primarily reflects variation in the sampling density rather than variation in curvature. The uniform distribution gives near-zero LDD variance and near-uniform couplings up to finite-sample noise. The one-mode vMF produces the strongest LDD variance and the strongest coupling response, because its density has a large smooth gradient. The mixture-vMF can exhibit higher LDD effective rank and spectral entropy, indicating richer LDD variation, but this does not necessarily translate into sharper or more non-uniform couplings. Thus, in this S2 control, coupling response is driven more by LDD amplitude than by LDD spectral complexity.
"""

    with open(path, "w") as f:
        f.write(text)


def save_run_npz(path, Z_a, Z_b, R_star, radii, H_a, H_b, M, P, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    np.savez_compressed(
        path,
        Z_a=Z_a.astype(np.float32),
        Z_b=Z_b.astype(np.float32),
        R_star=R_star.astype(np.float32),
        radii=np.asarray(radii, dtype=np.float32),
        H_a=H_a.detach().cpu().numpy().astype(np.float32),
        H_b=H_b.detach().cpu().numpy().astype(np.float32),
        M=M.detach().cpu().numpy().astype(np.float32),
        P=P.detach().cpu().numpy().astype(np.float32),
        case=np.array(row["case"]),
        distance_type=np.array(row["distance_type"]),
        sampling_mode=np.array("independent"),
        rep=np.int64(row["rep"]),
        seed_a=np.int64(row["seed_a"]),
        seed_b=np.int64(row["seed_b"]),
        seed_rot=np.int64(row["seed_rot"]),
    )


# ============================================================
# One run
# ============================================================

def run_one(case, distance_type, rep, args):
    seed_a = args.seed + 10000 * rep + 17
    seed_b = args.seed + 10000 * rep + 29
    seed_rot = args.seed + 10000 * rep + 101

    Z_a, Z_b, R_star = sample_independent_pair(
        case=case,
        n=args.n_points,
        seed_a=seed_a,
        seed_b=seed_b,
        seed_rot=seed_rot,
    )

    D_a = compute_distance_matrix(Z_a, distance_type)
    D_b = compute_distance_matrix(Z_b, distance_type)

    r_max = float(max(D_a.max(), D_b.max())) if args.r_max is None else float(args.r_max)

    radii = make_log_radii(
        r_min=args.r_min,
        r_max=r_max,
        r_bins=args.r_bins,
    )

    H_a = compute_ldd(D_a, radii)
    H_b = compute_ldd(D_b, radii)

    M = compute_ldd_cost(H_a, H_b)

    P = init_sinkhorn(
        M,
        epsilon=args.sinkhorn_epsilon,
        num_itermax=args.sinkhorn_iter,
    )

    diag_a = ldd_diagnostics(H_a, variance_floor=args.ldd_variance_floor)
    diag_b = ldd_diagnostics(H_b, variance_floor=args.ldd_variance_floor)
    diag_mean = average_ldd_diagnostics(diag_a, diag_b)
    diag_p = coupling_diagnostics(P, M)

    row = {
        "case": case,
        "distance_type": distance_type,
        "sampling_mode": "independent",
        "rep": int(rep),
        "seed_a": int(seed_a),
        "seed_b": int(seed_b),
        "seed_rot": int(seed_rot),
        "n_points": int(args.n_points),
        "r_bins": int(args.r_bins),
        "r_min": float(args.r_min),
        "r_max": float(r_max),
        "sinkhorn_epsilon": float(args.sinkhorn_epsilon),
        "ldd_variance_floor": float(args.ldd_variance_floor),
        "M_mean": float(M.mean().item()),
        "M_std": float(M.std().item()),
        "M_max": float(M.max().item()),
    }

    row.update({f"A_{k}": v for k, v in diag_a.items()})
    row.update({f"B_{k}": v for k, v in diag_b.items()})
    row.update(diag_mean)
    row.update(diag_p)

    run_name = f"{case}_independent_{distance_type}_rep{rep:03d}"

    artifact_dir = os.path.join(
        args.out_root,
        "s2",
        "artifacts",
        "independent",
        case,
        distance_type,
    )

    plot_dir = os.path.join(
        args.out_root,
        "s2",
        "plots",
        "independent",
        case,
        distance_type,
    )

    if args.save_arrays:
        save_run_npz(
            path=os.path.join(artifact_dir, f"{run_name}.npz"),
            Z_a=Z_a,
            Z_b=Z_b,
            R_star=R_star,
            radii=radii,
            H_a=H_a,
            H_b=H_b,
            M=M,
            P=P,
            row=row,
        )

    if args.save_run_plots:
        plot_ldd_curves(
            radii=radii,
            H_a=H_a,
            H_b=H_b,
            save_path=os.path.join(plot_dir, f"{run_name}_ldd.png"),
            title=f"LDD | {case} | {distance_type} | rep {rep}",
        )

        plot_coupling(
            P=P,
            save_path=os.path.join(plot_dir, f"{run_name}_P.png"),
            title=f"Coupling P | {case} | {distance_type} | rep {rep}",
        )

    return row


# ============================================================
# Main
# ============================================================

def main(args):
    if args.cases == ["all"]:
        cases = ["uniform_s2", "isotropic_vmf", "vmf_mixture"]
    else:
        cases = args.cases

    if args.distance_types == ["all"]:
        distance_types = ["geo", "euc", "eol"]
    else:
        distance_types = args.distance_types

    rows = []

    for case in cases:
        for distance_type in distance_types:
            for rep in range(args.n_reps):
                print(f"\n=== independent | case={case} | distance={distance_type} | rep={rep} ===")

                row = run_one(case, distance_type, rep, args)
                rows.append(row)

                print(
                    f"LDD var={row['ldd_variance_mean']:.6e} | "
                    f"LDD rank={row['ldd_effective_rank_mean']} | "
                    f"LDD entropy={row['ldd_spectral_entropy_mean']} | "
                    f"E_R={row['R_energy']:.6e} | "
                    f"Peak={row['Q_peak_ratio_mean']:.3f} | "
                    f"Gain={row['LDD_gain_over_uniform']:.4f}"
                )

    artifact_root = os.path.join(args.out_root, "s2", "artifacts", "independent")
    plot_root = os.path.join(args.out_root, "s2", "plots", "independent")

    os.makedirs(artifact_root, exist_ok=True)
    os.makedirs(plot_root, exist_ok=True)

    summary_csv = os.path.join(artifact_root, "ldd_sensitivity_summary.csv")
    grouped_csv = os.path.join(artifact_root, "ldd_sensitivity_grouped_summary.csv")
    summary_json = os.path.join(artifact_root, "ldd_sensitivity_summary.json")
    interpretation_md = os.path.join(artifact_root, "interpretation_note.md")

    control_plot = os.path.join(plot_root, "sphere_ldd_coupling_control_2x3.png")
    relationship_plot = os.path.join(plot_root, "sphere_ldd_coupling_relationships_2x2.png")

    save_summary_csv(rows, summary_csv)
    save_grouped_summary_csv(rows, grouped_csv)
    save_interpretation_note(rows, interpretation_md)

    with open(summary_json, "w") as f:
        json.dump(rows, f, indent=2)

    plot_sphere_ldd_coupling_control(rows, control_plot)
    plot_sphere_ldd_coupling_relationships(rows, relationship_plot)

    print("\nSaved summary CSV:")
    print(summary_csv)

    print("\nSaved grouped summary CSV:")
    print(grouped_csv)

    print("\nSaved summary JSON:")
    print(summary_json)

    print("\nSaved interpretation note:")
    print(interpretation_md)

    print("\nSaved control plot:")
    print(control_plot)

    print("\nSaved relationship plot:")
    print(relationship_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-root", type=str, default="ldds")

    parser.add_argument("--n-points", type=int, default=2000)
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--cases",
        nargs="+",
        default=["all"],
        choices=["all", "uniform_s2", "isotropic_vmf", "vmf_mixture"],
    )

    parser.add_argument(
        "--distance-types",
        nargs="+",
        default=["all"],
        choices=["all", "geo", "euc", "eol"],
    )

    parser.add_argument("--r-bins", type=int, default=100)
    parser.add_argument("--r-min", type=float, default=1e-4)
    parser.add_argument("--r-max", type=float, default=None)

    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iter", type=int, default=1000)

    parser.add_argument(
        "--ldd-variance-floor",
        type=float,
        default=1e-4,
        help="Below this LDD variance, effective rank and spectral entropy are set to NaN.",
    )

    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--save-run-plots", action="store_true")

    args = parser.parse_args()
    main(args)