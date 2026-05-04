# LDD_sensitivity_s2.py

import os
import csv
import argparse
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
import ot

try:
    from colors import get_single_color
except ImportError:
    def get_single_color(i):
        cols = [
            "tab:cyan", "tab:blue", "tab:purple",
            "tab:orange", "tab:green", "tab:red",
            "tab:pink", "tab:gray", "tab:olive"
        ]
        return cols[i % len(cols)]


# ============================================================
# Parameters
# ============================================================

# Broad but clearly non-uniform one-mode vMF.
ONE_MODE_KAPPA = 4.0

# Four-mode local mixture parameters.
# The centers are placed close enough that the mixture does not cover S2
# like a near-uniform distribution, but the component kappas are sharp enough
# that the four-mode structure is visible.
MIX_BASE_KAPPA = 18.0

# Local heterogeneity across the four vMF components.
# These are deliberately sharper than ONE_MODE_KAPPA because each component
# describes a smaller local part of the distribution.
MIX_UNIQUE_KAPPAS = np.array([12.0, 16.0, 22.0, 30.0], dtype=np.float64)

# Mildly non-uniform masses.
# The weight variation should be visible, but not dominate the experiment.
MIX_UNIQUE_WEIGHTS = np.array([0.22, 0.24, 0.26, 0.28], dtype=np.float64)


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
    """
    Exact vMF sampling on S2.

    Larger kappa means sharper concentration.
    kappa = 0 is uniform.
    """
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


def sample_one_mode_vmf(n, rng):
    Z = sample_vmf_s2(
        mu=np.array([0.0, 0.0, 1.0]),
        kappa=ONE_MODE_KAPPA,
        n=n,
        rng=rng,
    )

    labels = np.zeros(n, dtype=np.int64)
    kappas = np.array([ONE_MODE_KAPPA, np.nan, np.nan, np.nan], dtype=np.float64)
    weights = np.array([1.0, np.nan, np.nan, np.nan], dtype=np.float64)

    return Z, labels, kappas, weights


def fixed_four_mode_centers():
    """
    Fixed nearby centers for the local four-mode mixture experiment.

    The centers are intentionally close enough that the four-mode mixture does
    not behave like a near-uniform distribution on S2. At the same time, they
    are not identical, so the mixture still has genuine local multi-component
    structure.

    Only kappa and weights change across mixture variants.
    """
    mus = np.array([
        [0.00,  0.00,  1.00],
        [0.42,  0.05,  0.91],
        [-0.28, 0.34,  0.90],
        [0.08, -0.48,  0.87],
    ], dtype=np.float64)

    return normalize_rows(mus)

def all_mixture_masks():
    masks = []
    for k in range(5):
        for combo in itertools.combinations(range(4), k):
            masks.append(tuple(combo))
    return masks


def mixture_case_name(mask):
    if len(mask) == 0:
        return "mix_u0"
    return "mix_u" + str(len(mask)) + "_" + "".join(str(i + 1) for i in mask)


def mixture_parameters_from_mask(mask):
    """
    Components in mask get locally unique parameters.
    Components outside mask share remaining mass and base kappa.

    unique_count=0:
        all four modes have same kappa and equal weights.

    unique_count=1,2,3:
        all choices of which components are unique are run and later averaged.

    unique_count=4:
        all four modes have unique local parameters.
    """
    mask = tuple(mask)

    kappas = np.full(4, MIX_BASE_KAPPA, dtype=np.float64)
    weights = np.ones(4, dtype=np.float64) / 4.0

    if len(mask) == 0:
        return kappas, weights

    for idx in mask:
        kappas[idx] = MIX_UNIQUE_KAPPAS[idx]

    assigned_mass = MIX_UNIQUE_WEIGHTS[list(mask)].sum()
    remaining_idx = [idx for idx in range(4) if idx not in mask]

    for idx in mask:
        weights[idx] = MIX_UNIQUE_WEIGHTS[idx]

    if len(remaining_idx) > 0:
        remaining_mass = 1.0 - assigned_mass
        if remaining_mass <= 0:
            raise ValueError(f"Bad mixture weights for mask={mask}.")
        for idx in remaining_idx:
            weights[idx] = remaining_mass / len(remaining_idx)
    else:
        weights = MIX_UNIQUE_WEIGHTS.copy()

    weights = weights / weights.sum()

    return kappas, weights


def sample_vmf_mixture_mask(n, mask, rng):
    mus = fixed_four_mode_centers()
    kappas, weights = mixture_parameters_from_mask(mask)

    labels = rng.choice(4, size=n, p=weights)
    Z = np.zeros((n, 3), dtype=np.float64)

    for k in range(4):
        idx = np.where(labels == k)[0]
        if len(idx) > 0:
            Z[idx] = sample_vmf_s2(
                mu=mus[k],
                kappa=float(kappas[k]),
                n=len(idx),
                rng=rng,
            )

    return normalize_rows(Z), labels, kappas, weights


def sample_distribution(row_spec, n, rng):
    kind = row_spec["kind"]

    if kind == "uniform":
        Z = sample_uniform_s2(n, rng)
        labels = -np.ones(n, dtype=np.int64)
        return Z, labels, np.full(4, np.nan), np.full(4, np.nan)

    if kind == "one_vmf":
        return sample_one_mode_vmf(n, rng)

    if kind == "mixture":
        return sample_vmf_mixture_mask(n, row_spec["mask"], rng)

    raise ValueError(f"Unknown row_spec kind: {kind}")


def sample_independent_pair(row_spec, n, seed_a, seed_b, seed_rot):
    """
    Independent sampling:

        A_i ~ mu
        B_j = R* B0_j, where B0_j ~ mu independently.

    There is no pointwise correspondence.
    """
    rng_a = np.random.default_rng(seed_a)
    rng_b = np.random.default_rng(seed_b)
    rng_rot = np.random.default_rng(seed_rot)

    Z_a, labels_a, kappas, weights = sample_distribution(row_spec, n, rng_a)
    Z_b0, labels_b, _, _ = sample_distribution(row_spec, n, rng_b)

    R_star = random_rotation_matrix(rng_rot)
    Z_b = normalize_rows(Z_b0 @ R_star.T)

    return Z_a, Z_b, labels_a, labels_b, R_star, kappas, weights


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


def compute_distance_matrix(Z, distance_type):
    if distance_type == "geo":
        return distance_geo_arccos(Z)

    if distance_type == "euc":
        return distance_euc_chord(Z)

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
    LDD diagnostics from centered LDD matrix.

    H shape:
        n_points x n_radii

    ldd_variance:
        amplitude of LDD variation.

    ldd_effective_rank:
        exp(entropy of covariance eigenvalue distribution).

    ldd_spectral_entropy:
        entropy of normalized covariance eigenvalues.
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
    Row-wise coupling diagnostics.

    Q:
        row-normalized P.

    R_energy:
        mean squared deviation from row-uniform coupling.

    Q_peak_ratio_mean:
        average row maximum relative to 1 / n_targets.

    LDD_gain_over_uniform:
        relative reduction in LDD cost compared with fully uniform coupling.
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
        "R_energy": float(R_energy),
        "Q_peak_ratio_mean": float(peak_ratio.mean().item()),
        "Q_peak_ratio_std": float(peak_ratio.std().item()),
        "Q_inverse_entropy_mean": float(inverse_entropy.mean().item()),
        "Q_inverse_entropy_std": float(inverse_entropy.std().item()),
        "LDD_cost_P": float(cost_P),
        "LDD_cost_uniform": float(cost_uniform),
        "LDD_gain_over_uniform": float(LDD_gain),
    }


# ============================================================
# Experiment specification
# ============================================================

def build_row_specs():
    specs = [
        {
            "kind": "uniform",
            "case": "uniform",
            "plot_group": "uniform",
            "uniqueness_level": -2,
            "mask": (),
            "mask_label": "none",
        },
        {
            "kind": "one_vmf",
            "case": "one_vmf",
            "plot_group": "one_vmf",
            "uniqueness_level": -1,
            "mask": (),
            "mask_label": "none",
        },
    ]

    for mask in all_mixture_masks():
        k = len(mask)
        case = mixture_case_name(mask)
        specs.append({
            "kind": "mixture",
            "case": case,
            "plot_group": f"mix_u{k}",
            "uniqueness_level": k,
            "mask": mask,
            "mask_label": "none" if k == 0 else "-".join(str(i + 1) for i in mask),
        })

    return specs


def spec_offset(spec):
    if spec["kind"] == "uniform":
        return 11
    if spec["kind"] == "one_vmf":
        return 23

    mask = spec["mask"]
    value = 1000 + 100 * len(mask)
    for idx in mask:
        value += 7 * (idx + 1)
    return value


def run_one(row_spec, distance_type, rep, args):
    offset = spec_offset(row_spec)

    seed_a = args.seed + 10000 * rep + offset + 17
    seed_b = args.seed + 10000 * rep + offset + 29
    seed_rot = args.seed + 10000 * rep + offset + 101

    Z_a, Z_b, labels_a, labels_b, R_star, kappas, weights = sample_independent_pair(
        row_spec=row_spec,
        n=args.n_points,
        seed_a=seed_a,
        seed_b=seed_b,
        seed_rot=seed_rot,
    )

    D_a = compute_distance_matrix(Z_a, distance_type)
    D_b = compute_distance_matrix(Z_b, distance_type)

    r_max = float(max(D_a.max(), D_b.max())) if args.r_max is None else float(args.r_max)
    radii = make_log_radii(args.r_min, r_max, args.r_bins)

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
        "kind": row_spec["kind"],
        "case": row_spec["case"],
        "plot_group": row_spec["plot_group"],
        "uniqueness_level": int(row_spec["uniqueness_level"]),
        "mask_label": row_spec["mask_label"],
        "distance_type": distance_type,
        "rep": int(rep),
        "seed_a": int(seed_a),
        "seed_b": int(seed_b),
        "seed_rot": int(seed_rot),
        "n_points": int(args.n_points),
        "r_bins": int(args.r_bins),
        "r_min": float(args.r_min),
        "r_max": float(r_max),
        "sinkhorn_epsilon": float(args.sinkhorn_epsilon),
        "one_mode_kappa": float(ONE_MODE_KAPPA),
        "mix_base_kappa": float(MIX_BASE_KAPPA),
        "kappa_1": float(kappas[0]),
        "kappa_2": float(kappas[1]),
        "kappa_3": float(kappas[2]),
        "kappa_4": float(kappas[3]),
        "weight_1": float(weights[0]),
        "weight_2": float(weights[1]),
        "weight_3": float(weights[2]),
        "weight_4": float(weights[3]),
        "M_mean": float(M.mean().item()),
        "M_std": float(M.std().item()),
        "M_max": float(M.max().item()),
    }

    row.update({f"A_{k}": v for k, v in diag_a.items()})
    row.update({f"B_{k}": v for k, v in diag_b.items()})
    row.update(diag_mean)
    row.update(diag_p)

    return row


# ============================================================
# Saving
# ============================================================

def save_rows_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    keys = sorted({k for row in rows for k in row.keys()})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_specs():
    return [
        ("ldd_variance_mean", "LDD variance"),
        ("ldd_effective_rank_mean", "LDD effective rank"),
        ("ldd_spectral_entropy_mean", "LDD spectral entropy"),
        ("R_energy", "Coupling response $E_R$"),
        ("Q_peak_ratio_mean", "Peak ratio"),
        ("LDD_gain_over_uniform", "LDD-cost gain"),
    ]


def save_grouped_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    metrics = [m[0] for m in metric_specs()]
    group_keys = ["plot_group", "distance_type"]

    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out_rows = []
    for key, group in grouped.items():
        out = {
            "plot_group": key[0],
            "distance_type": key[1],
            "n": len(group),
        }

        for metric in metrics:
            vals = np.array([float(g.get(metric, np.nan)) for g in group], dtype=np.float64)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                out[f"{metric}_mean"] = np.nan
                out[f"{metric}_std"] = np.nan
            else:
                out[f"{metric}_mean"] = float(vals.mean())
                out[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        out_rows.append(out)

    keys = sorted({k for row in out_rows for k in row.keys()})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)


# ============================================================
# Plot helpers
# ============================================================

def grouped_x_order():
    return [
        "uniform",
        "one_vmf",
        "mix_u0",
        "mix_u1",
        "mix_u2",
        "mix_u3",
        "mix_u4",
    ]


def grouped_labels():
    return {
        "uniform": "Uniform",
        "one_vmf": "1-mode\nvMF",
        "mix_u0": "4-mode\nsame",
        "mix_u1": "4-mode\n1 unique",
        "mix_u2": "4-mode\n2 unique",
        "mix_u3": "4-mode\n3 unique",
        "mix_u4": "4-mode\n4 unique",
    }


def group_color(group):
    colors = {
        "uniform": get_single_color(0),
        "one_vmf": get_single_color(2),
        "mix_u0": get_single_color(4),
        "mix_u1": get_single_color(5),
        "mix_u2": get_single_color(6),
        "mix_u3": get_single_color(7),
        "mix_u4": get_single_color(8),
    }
    return colors[group]


def distance_specs():
    markers = {"geo": "o", "euc": "s"}
    labels = {"geo": "geodesic", "euc": "Euclidean chord"}
    colors = {"geo": get_single_color(0), "euc": get_single_color(2)}
    return markers, labels, colors


def set_group_xticks(ax, groups):
    labels = grouped_labels()
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([labels[g] for g in groups], rotation=20, ha="right")


def add_group_legend(fig, groups, y=0.935, ncol=7):
    labels = grouped_labels()

    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color=group_color(g),
            linestyle="none",
            markeredgecolor="black",
            markersize=8,
            label=labels[g].replace("\n", " "),
        )
        for g in groups
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        fontsize=9.8,
    )


def add_distance_legend(fig, y=0.935):
    markers, labels, colors = distance_specs()

    handles = [
        plt.Line2D(
            [0], [0],
            marker=markers[d],
            color=colors[d],
            linestyle="none",
            markeredgecolor="black",
            markersize=8,
            label=labels[d],
        )
        for d in ["geo", "euc"]
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )


# ============================================================
# Plots
# ============================================================

def plot_main_distribution_sensitivity(rows, save_path, main_distance):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    groups = grouped_x_order()
    x_pos = {g: i for i, g in enumerate(groups)}

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0), squeeze=False)

    for ax, (metric, title) in zip(axes.ravel(), metric_specs()):
        for row in rows:
            if row["distance_type"] != main_distance:
                continue

            group = row["plot_group"]
            if group not in x_pos:
                continue

            y = float(row.get(metric, np.nan))
            if not np.isfinite(y):
                continue

            mask_hash = sum(ord(ch) for ch in row["mask_label"])
            jitter = 0.012 * int(row["rep"]) + 0.006 * ((mask_hash % 7) - 3)

            ax.scatter(
                x_pos[group] + jitter,
                y,
                s=48,
                alpha=0.70,
                color=group_color(group),
                marker="o",
                edgecolors="black",
                linewidths=0.25,
            )

        set_group_xticks(ax, groups)
        ax.set_title(title, fontsize=12.5, pad=9)
        ax.grid(alpha=0.22)

    fig.suptitle(
        f"LDD and coupling sensitivity across distributions ({main_distance} distance)",
        fontsize=18,
        y=0.985,
    )

    add_group_legend(fig, groups, y=0.935, ncol=7)

    fig.subplots_adjust(
        top=0.82,
        bottom=0.16,
        left=0.055,
        right=0.985,
        hspace=0.44,
        wspace=0.28,
    )

    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_mean_uniqueness_trend(rows, save_path, main_distance):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    levels = [0, 1, 2, 3, 4]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), squeeze=False)

    for ax, (metric, title) in zip(axes.ravel(), metric_specs()):
        xs, ys, es = [], [], []

        for level in levels:
            subset = [
                r for r in rows
                if r["distance_type"] == main_distance
                and r["kind"] == "mixture"
                and r["uniqueness_level"] == level
            ]

            vals = np.array([float(r.get(metric, np.nan)) for r in subset], dtype=np.float64)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                continue

            xs.append(level)
            ys.append(float(vals.mean()))
            es.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

        ax.errorbar(
            xs,
            ys,
            yerr=es,
            marker="o",
            color=get_single_color(8),
            linewidth=1.8,
            markersize=6.5,
            capsize=3,
        )

        ax.set_xticks(levels)
        ax.set_xlabel("Number of locally unique mixture components")
        ax.set_title(title, fontsize=12.5, pad=9)
        ax.grid(alpha=0.22)

    fig.suptitle(
        f"Averaged trend over which modes are unique ({main_distance} distance)",
        fontsize=18,
        y=0.985,
    )

    fig.subplots_adjust(
        top=0.90,
        bottom=0.10,
        left=0.065,
        right=0.985,
        hspace=0.42,
        wspace=0.28,
    )

    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_selected_contrast(rows, save_path, main_distance):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    groups = ["uniform", "one_vmf", "mix_u0", "mix_u4"]
    x_pos = {g: i for i, g in enumerate(groups)}

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.7), squeeze=False)

    for ax, (metric, title) in zip(axes.ravel(), metric_specs()):
        for row in rows:
            if row["distance_type"] != main_distance:
                continue

            group = row["plot_group"]
            if group not in x_pos:
                continue

            y = float(row.get(metric, np.nan))
            if not np.isfinite(y):
                continue

            jitter = 0.018 * int(row["rep"])

            ax.scatter(
                x_pos[group] + jitter,
                y,
                s=55,
                alpha=0.75,
                color=group_color(group),
                marker="o",
                edgecolors="black",
                linewidths=0.25,
            )

        set_group_xticks(ax, groups)
        ax.set_title(title, fontsize=12.5, pad=9)
        ax.grid(alpha=0.22)

    fig.suptitle(
        f"Selected distribution contrast ({main_distance} distance)",
        fontsize=18,
        y=0.985,
    )

    add_group_legend(fig, groups, y=0.935, ncol=4)

    fig.subplots_adjust(
        top=0.82,
        bottom=0.15,
        left=0.06,
        right=0.985,
        hspace=0.44,
        wspace=0.28,
    )

    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_ldd_vs_coupling(rows, save_path, main_distance):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pairs = [
        ("ldd_variance_mean", "R_energy", "LDD variance", "Coupling response $E_R$", "Variance vs response"),
        ("ldd_effective_rank_mean", "R_energy", "LDD effective rank", "Coupling response $E_R$", "Rank vs response"),
        ("ldd_spectral_entropy_mean", "R_energy", "LDD spectral entropy", "Coupling response $E_R$", "Entropy vs response"),
        ("ldd_variance_mean", "LDD_gain_over_uniform", "LDD variance", "LDD-cost gain", "Variance vs cost gain"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), squeeze=False)

    for ax, (x_key, y_key, x_lab, y_lab, title) in zip(axes.ravel(), pairs):
        for row in rows:
            if row["distance_type"] != main_distance:
                continue

            x = float(row.get(x_key, np.nan))
            y = float(row.get(y_key, np.nan))

            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            group = row["plot_group"]

            ax.scatter(
                x,
                y,
                s=52,
                alpha=0.72,
                color=group_color(group),
                marker="o",
                edgecolors="black",
                linewidths=0.25,
            )

        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        ax.set_title(title, fontsize=12.5, pad=9)
        ax.grid(alpha=0.22)

    groups = grouped_x_order()
    add_group_legend(fig, groups, y=0.94, ncol=7)

    fig.suptitle(
        f"LDD diagnostics versus coupling diagnostics ({main_distance} distance)",
        fontsize=18,
        y=0.99,
    )

    fig.subplots_adjust(
        top=0.82,
        bottom=0.09,
        left=0.075,
        right=0.985,
        hspace=0.40,
        wspace=0.30,
    )

    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_distance_sensitivity_mean(rows, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    groups = ["uniform", "one_vmf", "mix_u0", "mix_u4"]
    x_pos = {g: i for i, g in enumerate(groups)}

    markers, dist_labels, dist_colors = distance_specs()

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), squeeze=False)

    for ax, (metric, title) in zip(axes.ravel(), metric_specs()):
        for d in ["geo", "euc"]:
            xs, ys, es = [], [], []

            for group in groups:
                subset = [
                    r for r in rows
                    if r["distance_type"] == d
                    and r["plot_group"] == group
                ]

                vals = np.array([float(r.get(metric, np.nan)) for r in subset], dtype=np.float64)
                vals = vals[np.isfinite(vals)]

                if len(vals) == 0:
                    continue

                xs.append(x_pos[group])
                ys.append(float(vals.mean()))
                es.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

            ax.errorbar(
                np.asarray(xs) + {"geo": -0.08, "euc": 0.08}[d],
                ys,
                yerr=es,
                marker=markers[d],
                color=dist_colors[d],
                linewidth=1.5,
                markersize=6.5,
                capsize=3,
                label=dist_labels[d],
            )

        set_group_xticks(ax, groups)
        ax.set_title(title, fontsize=12.5, pad=9)
        ax.grid(alpha=0.22)

    fig.suptitle(
        "Mean distance-measure sensitivity for selected distributions",
        fontsize=18,
        y=0.985,
    )

    add_distance_legend(fig, y=0.935)

    fig.subplots_adjust(
        top=0.84,
        bottom=0.16,
        left=0.065,
        right=0.985,
        hspace=0.42,
        wspace=0.28,
    )

    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_sphere_points(ax, Z, labels, title):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)

    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(xs, ys, zs, linewidth=0.25, alpha=0.15)

    if np.all(labels < 0):
        ax.scatter(
            Z[:, 0], Z[:, 1], Z[:, 2],
            s=8,
            alpha=0.45,
            color=get_single_color(0),
            depthshade=False,
        )
    else:
        for k in sorted(set(labels.tolist())):
            idx = labels == k
            ax.scatter(
                Z[idx, 0], Z[idx, 1], Z[idx, 2],
                s=9,
                alpha=0.55,
                color=get_single_color(k + 2),
                label=f"mode {k + 1}" if k >= 0 else "uniform",
                depthshade=False,
            )

    ax.set_title(title, fontsize=11)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_zlim([-1.05, 1.05])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_distribution_sphere_gallery(save_path, n_plot, seed):
    """
    Visual check of the distributions on S2.
    This is only for checking whether kappas are reasonable.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rng = np.random.default_rng(seed)

    specs = [
        {
            "kind": "uniform",
            "case": "uniform",
            "plot_group": "uniform",
            "uniqueness_level": -2,
            "mask": (),
            "mask_label": "none",
        },
        {
            "kind": "one_vmf",
            "case": "one_vmf",
            "plot_group": "one_vmf",
            "uniqueness_level": -1,
            "mask": (),
            "mask_label": "none",
        },
        {
            "kind": "mixture",
            "case": "mix_u0",
            "plot_group": "mix_u0",
            "uniqueness_level": 0,
            "mask": (),
            "mask_label": "none",
        },
        {
            "kind": "mixture",
            "case": "mix_u4_1234",
            "plot_group": "mix_u4",
            "uniqueness_level": 4,
            "mask": (0, 1, 2, 3),
            "mask_label": "1-2-3-4",
        },
    ]

    titles = [
        "Uniform on S2",
        f"1-mode vMF\nkappa={ONE_MODE_KAPPA}",
        "4-mode local vMF: same\n"
        + f"kappa={MIX_BASE_KAPPA}",
        "4-mode local vMF: all unique\n"
        + f"kappa={np.round(MIX_UNIQUE_KAPPAS, 1)}",
    ]

    fig = plt.figure(figsize=(11.5, 9.0))

    for i, (spec, title) in enumerate(zip(specs, titles)):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        Z, labels, kappas, weights = sample_distribution(spec, n_plot, rng)

        if spec["kind"] == "mixture":
            title = title + "\n" + f"weights={np.round(weights, 2)}"

        plot_sphere_points(ax, Z, labels, title)

    fig.suptitle(
    "Visual check: sampled local vMF distributions on the unit sphere",
    fontsize=16,
    y=0.98,
)   

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.03, hspace=0.12)

    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main(args):
    all_distance_types = ["geo", "euc"]
    row_specs = build_row_specs()

    out_root = args.out_root
    plot_root = os.path.join(out_root, "plots")
    csv_root = os.path.join(out_root, "csv")

    os.makedirs(plot_root, exist_ok=True)
    os.makedirs(csv_root, exist_ok=True)

    rows = []

    print("\nOutput folder:")
    print(out_root)

    print("\nParameters:")
    print(f"ONE_MODE_KAPPA={ONE_MODE_KAPPA}")
    print(f"MIX_BASE_KAPPA={MIX_BASE_KAPPA}")
    print(f"MIX_UNIQUE_KAPPAS={MIX_UNIQUE_KAPPAS}")
    print(f"MIX_UNIQUE_WEIGHTS={MIX_UNIQUE_WEIGHTS}")
    print("Mixture centers:")
    print(np.round(fixed_four_mode_centers(), 4))

    print("\nMain distance for distribution-sensitivity plots:")
    print(args.main_distance)

    print("\nExperiment design:")
    print("Uniform")
    print("1-mode broad/moderate vMF around north pole")
    print("4-mode local mixtures with all masks for 0, 1, 2, 3, 4 locally unique components")
    print("All mixture masks for 0, 1, 2, 3, 4 locally unique components")
    print("For 1, 2, and 3 unique components, plots average over which modes are unique.")
    print("Distances used: geodesic arccos and Euclidean chord. EOL is removed.")

    for spec in row_specs:
        print("\n============================================================")
        print(f"case={spec['case']}")
        print(f"kind={spec['kind']}")
        print(f"plot_group={spec['plot_group']}")
        print(f"uniqueness_level={spec['uniqueness_level']}")
        print(f"mask={spec['mask_label']}")

        if spec["kind"] == "mixture":
            kappas, weights = mixture_parameters_from_mask(spec["mask"])
            print(f"kappas={np.round(kappas, 4)}")
            print(f"weights={np.round(weights, 4)}")

        if spec["kind"] == "one_vmf":
            print(f"kappa={ONE_MODE_KAPPA}")

        print("============================================================")

        for distance_type in all_distance_types:
            for rep in range(args.n_reps):
                row = run_one(
                    row_spec=spec,
                    distance_type=distance_type,
                    rep=rep,
                    args=args,
                )
                rows.append(row)

                if distance_type == args.main_distance:
                    print(
                        f"distance={distance_type:>3s} | rep={rep:03d} | "
                        f"LDD var={row['ldd_variance_mean']:.6e} | "
                        f"rank={row['ldd_effective_rank_mean']} | "
                        f"entropy={row['ldd_spectral_entropy_mean']} | "
                        f"E_R={row['R_energy']:.6e} | "
                        f"peak={row['Q_peak_ratio_mean']:.4f} | "
                        f"gain={row['LDD_gain_over_uniform']:.4f}"
                    )

    results_csv = os.path.join(csv_root, "ldd_sensitivity_results.csv")
    grouped_csv = os.path.join(csv_root, "ldd_sensitivity_grouped_summary.csv")

    save_rows_csv(rows, results_csv)
    save_grouped_csv(rows, grouped_csv)

    p0 = os.path.join(plot_root, "00_local_vmf_distribution_gallery.png")
    p1 = os.path.join(plot_root, "01_local_vmf_distribution_sensitivity_grouped_2x3.png")
    p2 = os.path.join(plot_root, "02_local_vmf_uniqueness_mean_trend_2x3.png")
    p3 = os.path.join(plot_root, "03_selected_local_vmf_contrast_2x3.png")
    p4 = os.path.join(plot_root, "04_ldd_vs_coupling_local_vmf_2x2.png")
    p5 = os.path.join(plot_root, "05_distance_measure_sensitivity_mean_2x3.png")

    plot_distribution_sphere_gallery(
        save_path=p0,
        n_plot=args.n_plot_sphere,
        seed=args.seed + 777,
    )
    plot_main_distribution_sensitivity(rows, p1, args.main_distance)
    plot_mean_uniqueness_trend(rows, p2, args.main_distance)
    plot_selected_contrast(rows, p3, args.main_distance)
    plot_ldd_vs_coupling(rows, p4, args.main_distance)
    plot_distance_sensitivity_mean(rows, p5)

    print("\nSaved CSV files:")
    print(results_csv)
    print(grouped_csv)

    print("\nSaved plots:")
    print(p0)
    print(p1)
    print(p2)
    print(p3)
    print(p4)
    print(p5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-root", type=str, default="ldd_sensitivity_local_vmf")

    parser.add_argument("--n-points", type=int, default=1200)
    parser.add_argument("--n-reps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--main-distance",
        type=str,
        default="geo",
        choices=["geo", "euc"],
        help="Distance used for the main distribution-sensitivity plots.",
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

    parser.add_argument(
        "--n-plot-sphere",
        type=int,
        default=1000,
        help="Number of points used in the 3D sphere visualization plot.",
    )

    args = parser.parse_args()
    main(args)