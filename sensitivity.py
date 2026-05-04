import os
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import ot
from scipy.sparse.linalg import svds
from colors import get_single_color

ONE_MODE_KAPPA_RANGE = (4.0, 8.0)
N_COMPONENTS = 4
MIXTURE_WEIGHTS = np.ones(N_COMPONENTS) / N_COMPONENTS
SHAPE_CASES = ["same", "1_unique", "2_unique", "4_unique"]
MU_LAYOUTS = ["close", "spread", "symmetric"]
SHAPE_STD_MAJOR_RANGE = (0.12, 0.38)
SHAPE_RATIO_RANGE = (0.18, 0.85)
SHAPE_ANGLE_RANGE = (0.0, np.pi)
MIN_UNIQUE_SHAPE_DISTANCE = 0.18
MAX_SHAPE_RESAMPLE_ATTEMPTS = 300
CLOSE_RADIUS_RANGE = (0.28, 0.62)
CLOSE_MIN_SEP = 0.10
SPREAD_MIN_SEP_RANGE = (0.85, 1.20)
SYMMETRIC_RADIUS_RANGE = (0.32, 0.85)
SYMMETRIC_RADIUS_JITTER = 0.12

BASELINE_SPECS = [
    {"kind": "uniform", "case": "uniform", "plot_group": "uniform", "shape_case": "baseline", "mu_layout": "baseline"},
    {"kind": "one_vmf", "case": "one_vmf", "plot_group": "one_vmf", "shape_case": "baseline", "mu_layout": "baseline"},
]
SHAPE_LABELS = {"uniform": "Uniform", "one_vmf": "1 vMF", "same": "Same", "1_unique": "1 unique", "2_unique": "2 unique", "4_unique": "4 unique"}
MU_LABELS = {"close": "close centers", "spread": "spread centers", "symmetric": "symmetric centers"}
MU_MARKERS = {"close": "o", "spread": "s", "symmetric": "^"}
MU_COLORS = {"close": get_single_color(2), "spread": get_single_color(5), "symmetric": get_single_color(8)}
X_POS = {"uniform": 0.0, "one_vmf": 1.8, "same": 4.0, "1_unique": 5.6, "2_unique": 7.2, "4_unique": 8.8}
LAYOUT_X_OFFSET = {"close": -0.28, "spread": 0.0, "symmetric": 0.28}

PLOT_METRICS = [
    ("ldd_variance_source", "LDD variance (source)"),
    ("ldd_effective_rank_source", "LDD effective rank (source)"),
    ("coupling_row_nonuniformity", "Coupling row non-uniformity"),
    ("coupling_row_peak_ratio_mean", "Coupling row peak ratio"),
    ("coupling_soft_block_count", "Coupling soft block count"),
    ("coupling_graph_fragmentation", "Coupling graph fragmentation"),
]
LDD_ROW_PAIRS = [
    ("ldd_variance_source", "coupling_row_nonuniformity", "Source LDD variance", "Coupling row non-uniformity", "Source LDD variance vs coupling row non-uniformity"),
    ("ldd_effective_rank_source", "coupling_row_nonuniformity", "Source LDD effective rank", "Coupling row non-uniformity", "Source LDD rank vs coupling row non-uniformity"),
    ("ldd_variance_source", "coupling_row_peak_ratio_mean", "Source LDD variance", "Coupling row peak ratio", "Source LDD variance vs coupling row peak ratio"),
    ("ldd_effective_rank_source", "coupling_row_peak_ratio_mean", "Source LDD effective rank", "Coupling row peak ratio", "Source LDD rank vs coupling row peak ratio"),
]
ROW_STRUCTURE_PAIRS = [
    ("coupling_row_nonuniformity", "coupling_soft_block_count", "Coupling row non-uniformity", "Coupling soft block count", "Row non-uniformity vs soft block count"),
    ("coupling_row_peak_ratio_mean", "coupling_soft_block_count", "Coupling row peak ratio", "Coupling soft block count", "Row peak ratio vs soft block count"),
    ("coupling_row_nonuniformity", "coupling_graph_fragmentation", "Coupling row non-uniformity", "Coupling graph fragmentation", "Row non-uniformity vs graph fragmentation"),
    ("coupling_row_peak_ratio_mean", "coupling_graph_fragmentation", "Coupling row peak ratio", "Coupling graph fragmentation", "Row peak ratio vs graph fragmentation"),
]
CORRELATION_METRICS = ["ldd_variance_source", "ldd_effective_rank_source", "coupling_row_nonuniformity", "coupling_row_peak_ratio_mean", "coupling_row_inverse_entropy_mean", "coupling_soft_block_count", "coupling_graph_fragmentation"]


def normalize_rows(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def normalize_vec(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x) + eps)


def tangent_basis(mu):
    mu = normalize_vec(mu)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(mu, ref))) > 0.92:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = normalize_vec(ref - np.dot(ref, mu) * mu)
    return e1, normalize_vec(np.cross(mu, e1))


def exp_map_s2(mu, V):
    mu = normalize_vec(mu)
    V = np.asarray(V, dtype=np.float64)
    r = np.linalg.norm(V, axis=1, keepdims=True)
    return normalize_rows(np.cos(r) * mu[None, :] + np.sin(r) * V / np.maximum(r, 1e-12))


def sample_uniform_s2(n, rng):
    return normalize_rows(rng.normal(size=(n, 3)))


def random_rotation_matrix(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    s = np.sign(np.diag(R))
    s[s == 0.0] = 1.0
    Q = Q @ np.diag(s)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def rotation_from_north(mu):
    mu = normalize_vec(mu)
    north = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(north, mu), float(np.dot(north, mu))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / s**2)


def sample_vmf_s2(mu, kappa, n, rng):
    if kappa <= 1e-12:
        return sample_uniform_s2(n, rng)
    u = rng.uniform(size=n)
    w = np.log(np.exp(-kappa) + u * (np.exp(kappa) - np.exp(-kappa))) / kappa
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radial = np.sqrt(np.maximum(1.0 - w**2, 0.0))
    X = np.column_stack([radial * np.cos(phi), radial * np.sin(phi), w])
    return normalize_rows(X @ rotation_from_north(mu).T)


def centers_close(rng):
    anchor = normalize_vec(rng.normal(size=3))
    e1, e2 = tangent_basis(anchor)
    radius, centers = rng.uniform(*CLOSE_RADIUS_RANGE), []
    for _ in range(20000):
        rho, phi = radius * np.sqrt(rng.uniform()), rng.uniform(0.0, 2.0 * np.pi)
        z = exp_map_s2(anchor, (rho * np.cos(phi) * e1 + rho * np.sin(phi) * e2)[None, :])[0]
        if len(centers) == 0 or np.min(np.arccos(np.clip(np.asarray(centers) @ z, -1.0, 1.0))) >= CLOSE_MIN_SEP:
            centers.append(z)
        if len(centers) == N_COMPONENTS:
            return normalize_rows(np.asarray(centers)), {"center_radius": float(radius)}
    raise RuntimeError("Could not sample close centers.")


def centers_spread(rng):
    min_sep, centers = rng.uniform(*SPREAD_MIN_SEP_RANGE), []
    for _ in range(30000):
        x = sample_uniform_s2(1, rng)[0]
        if len(centers) == 0 or np.min(np.arccos(np.clip(np.asarray(centers) @ x, -1.0, 1.0))) >= min_sep:
            centers.append(x)
        if len(centers) == N_COMPONENTS:
            return normalize_rows(np.asarray(centers)), {"center_min_sep": float(min_sep)}
    raise RuntimeError("Could not sample spread centers.")


def centers_symmetric(rng):
    anchor = normalize_vec(rng.normal(size=3))
    e1, e2 = tangent_basis(anchor)
    radius, phase = rng.uniform(*SYMMETRIC_RADIUS_RANGE), rng.uniform(0.0, 2.0 * np.pi)
    centers = []
    for k in range(N_COMPONENTS):
        phi = phase + 2.0 * np.pi * k / N_COMPONENTS
        rho = radius * (1.0 + rng.uniform(-SYMMETRIC_RADIUS_JITTER, SYMMETRIC_RADIUS_JITTER))
        centers.append(exp_map_s2(anchor, (rho * np.cos(phi) * e1 + rho * np.sin(phi) * e2)[None, :])[0])
    return normalize_rows(np.asarray(centers)), {"center_radius": float(radius)}


def centers_for_layout(mu_layout, rng):
    return {"close": centers_close, "spread": centers_spread, "symmetric": centers_symmetric}[mu_layout](rng)


def sample_shape(rng):
    std_major = rng.uniform(*SHAPE_STD_MAJOR_RANGE)
    ratio = rng.uniform(*SHAPE_RATIO_RANGE)
    return {"std_major": float(std_major), "std_minor": float(std_major * ratio), "ratio": float(ratio), "angle": float(np.mod(rng.uniform(*SHAPE_ANGLE_RANGE), np.pi))}


def shape_distance(a, b):
    da = abs(a["angle"] - b["angle"])
    da = min(da, np.pi - da) / np.pi
    return float(np.sqrt(np.log(a["std_major"] / b["std_major"])**2 + np.log(a["std_minor"] / b["std_minor"])**2 + da**2))


def sample_shape_distinct_from_many(rng, refs):
    candidate = None
    for _ in range(MAX_SHAPE_RESAMPLE_ATTEMPTS):
        candidate = sample_shape(rng)
        if all(shape_distance(candidate, ref) >= MIN_UNIQUE_SHAPE_DISTANCE for ref in refs):
            return candidate
    return candidate


def choose_unique_indices(shape_case, rng):
    if shape_case == "same":
        return tuple()
    if shape_case == "1_unique":
        return tuple(sorted(rng.choice(N_COMPONENTS, size=1, replace=False).tolist()))
    if shape_case == "2_unique":
        return tuple(sorted(rng.choice(N_COMPONENTS, size=2, replace=False).tolist()))
    if shape_case == "4_unique":
        return tuple(range(N_COMPONENTS))
    raise ValueError(f"Unknown shape_case: {shape_case}")


def component_shapes(shape_case, rng):
    unique_idx = choose_unique_indices(shape_case, rng)
    if shape_case == "same":
        shared = sample_shape(rng)
        return [dict(shared) for _ in range(N_COMPONENTS)], unique_idx
    if shape_case in ["1_unique", "2_unique"]:
        shared = sample_shape(rng)
        shapes, refs = [dict(shared) for _ in range(N_COMPONENTS)], [shared]
        for idx in unique_idx:
            shapes[idx] = sample_shape_distinct_from_many(rng, refs)
            refs.append(shapes[idx])
        return shapes, unique_idx
    shapes = []
    for _ in range(N_COMPONENTS):
        shapes.append(sample_shape_distinct_from_many(rng, shapes))
    return shapes, unique_idx


def sample_anisotropic_s2_component(mu, shape, n, rng):
    e1, e2 = tangent_basis(mu)
    c, s = np.cos(shape["angle"]), np.sin(shape["angle"])
    a1, a2 = c * e1 + s * e2, -s * e1 + c * e2
    V = rng.normal(0.0, shape["std_major"], size=n)[:, None] * a1[None, :] + rng.normal(0.0, shape["std_minor"], size=n)[:, None] * a2[None, :]
    return exp_map_s2(mu, V)


def build_specs(gallery=False):
    specs = [dict(s) for s in BASELINE_SPECS]
    outer, inner = (SHAPE_CASES, MU_LAYOUTS) if gallery else (MU_LAYOUTS, SHAPE_CASES)
    for a in outer:
        for b in inner:
            shape_case, mu_layout = (a, b) if gallery else (b, a)
            prefix = "gallery" if gallery else "mm"
            specs.append({"kind": "mixture", "case": f"{prefix}_{mu_layout}_{shape_case}", "plot_group": f"{mu_layout}_{shape_case}", "shape_case": shape_case, "mu_layout": mu_layout})
    return specs


def materialize_distribution_design(spec, design_seed):
    rng = np.random.default_rng(design_seed)
    if spec["kind"] == "one_vmf":
        return {"mu": sample_uniform_s2(1, rng)[0], "kappa": float(rng.uniform(*ONE_MODE_KAPPA_RANGE))}
    if spec["kind"] == "mixture":
        mus, center_info = centers_for_layout(spec["mu_layout"], rng)
        shapes, unique_idx = component_shapes(spec["shape_case"], rng)
        return {"mus": mus, "shapes": shapes, "center_info": center_info, "unique_idx": unique_idx}
    return None


def sample_distribution(spec, n, rng, design=None):
    if spec["kind"] == "uniform":
        return sample_uniform_s2(n, rng), -np.ones(n, dtype=np.int64), None, None
    if spec["kind"] == "one_vmf":
        return sample_vmf_s2(design["mu"], design["kappa"], n, rng), np.zeros(n, dtype=np.int64), None, None
    mus, shapes = design["mus"], design["shapes"]
    labels = rng.choice(N_COMPONENTS, size=n, p=MIXTURE_WEIGHTS)
    Z = np.zeros((n, 3), dtype=np.float64)
    for k in range(N_COMPONENTS):
        idx = np.where(labels == k)[0]
        if len(idx) > 0:
            Z[idx] = sample_anisotropic_s2_component(mus[k], shapes[k], len(idx), rng)
    return normalize_rows(Z), labels, mus, shapes


def spec_offset(spec):
    if spec["kind"] == "uniform":
        return 11
    if spec["kind"] == "one_vmf":
        return 23
    return {"close": 1000, "spread": 2000, "symmetric": 3000}[spec["mu_layout"]] + {"same": 10, "1_unique": 20, "2_unique": 30, "4_unique": 40}[spec["shape_case"]]


def sample_independent_pair(spec, n, seed_a, seed_b, seed_rot, design_seed):
    design = materialize_distribution_design(spec, design_seed)
    Z_a, labels_a, mus, shapes = sample_distribution(spec, n, np.random.default_rng(seed_a), design)
    Z_b0, labels_b, _, _ = sample_distribution(spec, n, np.random.default_rng(seed_b), design)
    R_star = random_rotation_matrix(np.random.default_rng(seed_rot))
    if spec["kind"] == "mixture":
        center_info, unique_idx, one_vmf_info = design["center_info"], design["unique_idx"], {}
    elif spec["kind"] == "one_vmf":
        center_info, unique_idx = {}, tuple()
        one_vmf_info = {"one_vmf_kappa": design["kappa"], "one_vmf_mu": design["mu"]}
    else:
        center_info, unique_idx, one_vmf_info = {}, tuple(), {}
    return Z_a, normalize_rows(Z_b0 @ R_star.T), labels_a, labels_b, R_star, mus, shapes, center_info, unique_idx, one_vmf_info


def distance_geo_arccos(Z):
    Z = normalize_rows(Z)
    return np.arccos(np.clip(Z @ Z.T, -1.0, 1.0))


def make_radii(r_min, r_max, r_bins, radius_grid):
    if r_min <= 0 or r_max <= r_min:
        raise ValueError("Require 0 < r_min < r_max.")
    if radius_grid == "log":
        return np.logspace(np.log10(float(r_min)), np.log10(float(r_max)), int(r_bins))
    if radius_grid == "linear":
        return np.linspace(float(r_min), float(r_max), int(r_bins))
    raise ValueError(f"Unknown radius_grid: {radius_grid}")


def compute_ldd(D, radii):
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"D must have shape (n,n), got {D.shape}")
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    D_sorted = np.sort(D, axis=1)
    H = np.empty((n, len(radii)), dtype=np.float32)
    for i in range(n):
        H[i] = np.searchsorted(D_sorted[i], radii, side="right") / (n - 1)
    return torch.from_numpy(H).float()


def compute_ldd_cost(H_a, H_b):
    return torch.cdist(torch.as_tensor(H_a, dtype=torch.float32), torch.as_tensor(H_b, dtype=torch.float32), p=2) ** 2


def init_sinkhorn(M, epsilon, num_itermax):
    M_np = torch.as_tensor(M, dtype=torch.float64).detach().cpu().numpy()
    n_a, n_b = M_np.shape
    a, b = np.ones(n_a) / n_a, np.ones(n_b) / n_b
    return torch.from_numpy(ot.sinkhorn(a, b, M_np, reg=float(epsilon), numItermax=int(num_itermax))).float()


def sinkhorn_diagnostics(P):
    P = torch.as_tensor(P, dtype=torch.float64)
    n, m = P.shape
    return {"sinkhorn_total_mass": float(P.sum().item()), "sinkhorn_row_marginal_max_error": float(torch.max(torch.abs(P.sum(dim=1) - 1.0 / n)).item()), "sinkhorn_col_marginal_max_error": float(torch.max(torch.abs(P.sum(dim=0) - 1.0 / m)).item())}


def effective_rank_from_energy(energy, eps=1e-12):
    energy = torch.clamp(torch.as_tensor(energy, dtype=torch.float64), min=0.0)
    total = energy.sum()
    if total <= eps:
        return np.nan
    p = energy / total
    return float(torch.exp(-(p * torch.log(p + eps)).sum()).item())


def ldd_diagnostics(H, variance_floor):
    H = torch.as_tensor(H, dtype=torch.float32)
    Hc = H - H.mean(dim=0, keepdim=True)
    var = float((Hc**2).mean().item())
    if var < variance_floor:
        return {"ldd_variance": var, "ldd_effective_rank": 1.0, "ldd_effective_rank_was_floored": 1.0}
    Cov = (Hc.T @ Hc) / max(H.shape[0] - 1, 1)
    eigvals = torch.clamp(torch.linalg.eigvalsh(Cov), min=0.0)
    return {"ldd_variance": var, "ldd_effective_rank": effective_rank_from_energy(eigvals), "ldd_effective_rank_was_floored": 0.0}


def average_diagnostics(a, b):
    return {f"{k}_mean": 0.5 * (a[k] + b[k]) if np.isfinite(a[k]) and np.isfinite(b[k]) else np.nan for k in a}


def source_ldd_metrics(diag_a):
    return {"ldd_variance_source": diag_a["ldd_variance"], "ldd_effective_rank_source": diag_a["ldd_effective_rank"], "ldd_effective_rank_source_was_floored": diag_a["ldd_effective_rank_was_floored"]}


def row_coupling_diagnostics(P, eps=1e-12):
    P = torch.as_tensor(P, dtype=torch.float32)
    n_a, n_b = P.shape
    Q = P / (P.sum(dim=1, keepdim=True) + eps)
    uniform_value = 1.0 / n_b
    peak_ratio = Q.max(dim=1).values / uniform_value
    row_entropy = -(Q * torch.log(Q + eps)).sum(dim=1)
    inverse_entropy = np.log(n_b) / (row_entropy + eps)
    return {"coupling_row_nonuniformity": float(((Q - uniform_value)**2).sum().item() / n_a), "coupling_row_peak_ratio_mean": float(peak_ratio.mean().item()), "coupling_row_peak_ratio_std": float(peak_ratio.std().item()), "coupling_row_inverse_entropy_mean": float(inverse_entropy.mean().item()), "coupling_row_inverse_entropy_std": float(inverse_entropy.std().item())}


def top_singular_values(K, svd_rank):
    K_np = torch.as_tensor(K, dtype=torch.float64).detach().cpu().numpy()
    min_nm = min(K_np.shape)
    r = min(max(8, int(svd_rank)), min_nm - 1) if min_nm > 1 else 1
    svals = np.linalg.svd(K_np, compute_uv=False) if min_nm <= 64 or r >= min_nm else svds(K_np, k=r, return_singular_vectors=False, which="LM")
    return np.clip(np.sort(np.asarray(svals, dtype=np.float64))[::-1], 0.0, 1.0)


def coupling_soft_block_count(P, tau=0.03, svd_rank=32, eps=1e-12):
    P = torch.as_tensor(P, dtype=torch.float64)
    row_mass, col_mass = P.sum(dim=1).clamp_min(eps), P.sum(dim=0).clamp_min(eps)
    K = P / torch.sqrt(row_mass[:, None] * col_mass[None, :])
    svals = top_singular_values(K, svd_rank)
    missing = max(0, min(P.shape) - len(svals))
    score = float(np.exp(-(1.0 - svals) / tau).sum() + missing * np.exp(-1.0 / tau))
    return {"coupling_soft_block_count": score, "coupling_soft_block_tau": float(tau), "coupling_soft_block_svd_rank_used": int(len(svals))}


def coupling_graph_fragmentation(P, topk=12, tau=0.08, min_signal=1e-8, eps=1e-12):
    P = torch.as_tensor(P, dtype=torch.float64)
    n, m = P.shape
    Q = P / (P.sum(dim=1, keepdim=True) + eps)
    W0 = torch.clamp(Q - (1.0 / m), min=0.0)
    if float((W0**2).sum().item() / n) < min_signal:
        score = 1.0
    else:
        k_row, k_col = max(1, min(int(topk), m)), max(1, min(int(topk), n))
        W = torch.zeros_like(W0)
        vals_r, idx_r = torch.topk(W0, k=k_row, dim=1)
        rows = torch.arange(n, device=W0.device)[:, None].expand(n, k_row)
        W[rows, idx_r] = torch.where(vals_r > 0.0, vals_r, W[rows, idx_r])
        vals_c, idx_c = torch.topk(W0, k=k_col, dim=0)
        cols = torch.arange(m, device=W0.device)[None, :].expand(k_col, m)
        W[idx_c, cols] = torch.maximum(W[idx_c, cols], torch.where(vals_c > 0.0, vals_c, torch.zeros_like(vals_c)))
        row_mass, col_mass = W.sum(dim=1), W.sum(dim=0)
        active_rows, active_cols = row_mass > eps, col_mass > eps
        if int(active_rows.sum().item()) < 2 or int(active_cols.sum().item()) < 2:
            score = 1.0
        else:
            W = W[active_rows][:, active_cols]
            row_mass, col_mass = W.sum(dim=1).clamp_min(eps), W.sum(dim=0).clamp_min(eps)
            K = W / torch.sqrt(row_mass[:, None] * col_mass[None, :])
            svals = np.linalg.svd(K.detach().cpu().numpy(), compute_uv=False)
            svals = np.clip(np.sort(np.asarray(svals, dtype=np.float64))[::-1], 0.0, 1.0)
            score = 1.0 if len(svals) <= 1 else 1.0 + float(np.exp(-(1.0 - svals[1:]) / tau).sum())
    return {"coupling_graph_fragmentation": score, "coupling_graph_topk": int(topk), "coupling_graph_fragmentation_tau": float(tau)}


def seed_bundle(args, rep, spec):
    offset = spec_offset(spec)
    base = args.seed + 10000 * rep + offset
    return {"seed_a": base + 17, "seed_b": base + 29, "seed_rot": base + 101, "design_seed": args.design_seed + 10000 * rep + offset}


def base_row(spec, rep, seeds, args, r_max, M):
    return {**{k: spec[k] for k in ["kind", "case", "plot_group", "shape_case", "mu_layout"]}, **seeds, "rep": int(rep), "distance_type": "geo", "radius_grid": args.radius_grid, "n_points": int(args.n_points), "r_bins": int(args.r_bins), "r_min": float(args.r_min), "r_max": float(r_max), "sinkhorn_epsilon": float(args.sinkhorn_epsilon), "M_mean": float(M.mean().item()), "M_std": float(M.std().item()), "M_max": float(M.max().item()), "one_vmf_kappa_min": float(ONE_MODE_KAPPA_RANGE[0]), "one_vmf_kappa_max": float(ONE_MODE_KAPPA_RANGE[1]), "shape_std_major_min": float(SHAPE_STD_MAJOR_RANGE[0]), "shape_std_major_max": float(SHAPE_STD_MAJOR_RANGE[1]), "shape_ratio_min": float(SHAPE_RATIO_RANGE[0]), "shape_ratio_max": float(SHAPE_RATIO_RANGE[1])}


def add_design_info(row, spec, mus, shapes, center_info, unique_idx, one_vmf_info):
    row["unique_idx"] = "none" if len(unique_idx) == 0 else "-".join(str(i + 1) for i in unique_idx)
    row["unique_count"] = int(len(unique_idx)) if spec["kind"] == "mixture" else (-2 if spec["kind"] == "uniform" else -1)
    if spec["kind"] == "one_vmf":
        row["one_vmf_kappa"] = float(one_vmf_info["one_vmf_kappa"])
        row.update({f"one_vmf_mu_{ax}": float(v) for ax, v in zip(["x", "y", "z"], one_vmf_info["one_vmf_mu"])})
    if spec["kind"] == "mixture":
        row.update({f"mixture_weight_{i + 1}": float(w) for i, w in enumerate(MIXTURE_WEIGHTS)})
        row.update({k: float(v) for k, v in center_info.items()})
        for k, sh in enumerate(shapes):
            row.update({f"shape{k + 1}_std_major": float(sh["std_major"]), f"shape{k + 1}_std_minor": float(sh["std_minor"]), f"shape{k + 1}_ratio": float(sh["ratio"]), f"shape{k + 1}_angle": float(sh["angle"]), f"shape{k + 1}_is_unique": int(k in unique_idx), f"mu{k + 1}_x": float(mus[k, 0]), f"mu{k + 1}_y": float(mus[k, 1]), f"mu{k + 1}_z": float(mus[k, 2])})


def run_one(spec, rep, args):
    seeds = seed_bundle(args, rep, spec)
    Z_a, Z_b, _, _, _, mus, shapes, center_info, unique_idx, one_vmf_info = sample_independent_pair(spec, args.n_points, seeds["seed_a"], seeds["seed_b"], seeds["seed_rot"], seeds["design_seed"])
    D_a, D_b = distance_geo_arccos(Z_a), distance_geo_arccos(Z_b)
    r_max = float(max(D_a.max(), D_b.max())) if args.r_max is None else float(args.r_max)
    radii = make_radii(args.r_min, r_max, args.r_bins, args.radius_grid)
    H_a, H_b = compute_ldd(D_a, radii), compute_ldd(D_b, radii)
    M = compute_ldd_cost(H_a, H_b)
    P = init_sinkhorn(M, args.sinkhorn_epsilon, args.sinkhorn_iter)
    diag_a, diag_b = ldd_diagnostics(H_a, args.ldd_variance_floor), ldd_diagnostics(H_b, args.ldd_variance_floor)
    row = base_row(spec, rep, seeds, args, r_max, M)
    add_design_info(row, spec, mus, shapes, center_info, unique_idx, one_vmf_info)
    row.update({f"A_{k}": v for k, v in diag_a.items()})
    row.update({f"B_{k}": v for k, v in diag_b.items()})
    row.update(average_diagnostics(diag_a, diag_b))
    row.update(source_ldd_metrics(diag_a))
    row.update(sinkhorn_diagnostics(P))
    row.update(row_coupling_diagnostics(P))
    row.update(coupling_soft_block_count(P, args.soft_block_tau, args.soft_block_svd_rank))
    row.update(coupling_graph_fragmentation(P, args.graph_topk, args.graph_fragmentation_tau, args.graph_min_signal))
    return row


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_grouped_csv(rows, path):
    group_keys_ = ["kind", "shape_case", "mu_layout", "plot_group", "distance_type", "radius_grid"]
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in group_keys_), []).append(row)
    out_rows = []
    for key, group in grouped.items():
        out = {k: v for k, v in zip(group_keys_, key)}
        out["n"] = len(group)
        for metric, _ in PLOT_METRICS:
            vals = np.asarray([float(g.get(metric, np.nan)) for g in group], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            out[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            out[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else np.nan)
        out_rows.append(out)
    write_csv(out_rows, path)


def pearson_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    return np.nan if len(x) < 2 else pearson_corr(rankdata_average(x), rankdata_average(y))


def save_correlation_csvs(rows, matrix_path, long_path):
    matrix = {m: np.asarray([float(r.get(m, np.nan)) for r in rows], dtype=np.float64) for m in CORRELATION_METRICS}
    write_csv([{**{"metric": x}, **{y: pearson_corr(matrix[x], matrix[y]) for y in CORRELATION_METRICS}} for x in CORRELATION_METRICS], matrix_path)
    long_rows = []
    for i, x in enumerate(CORRELATION_METRICS):
        for y in CORRELATION_METRICS[i + 1:]:
            mask = np.isfinite(matrix[x]) & np.isfinite(matrix[y])
            long_rows.append({"metric_x": x, "metric_y": y, "n": int(mask.sum()), "pearson": pearson_corr(matrix[x], matrix[y]), "spearman": spearman_corr(matrix[x], matrix[y])})
    write_csv(long_rows, long_path)


def row_color(row):
    if row["kind"] == "uniform":
        return "black"
    if row["kind"] == "one_vmf":
        return get_single_color(7)
    return MU_COLORS[row["mu_layout"]]


def row_marker(row):
    if row["kind"] == "uniform":
        return "X"
    if row["kind"] == "one_vmf":
        return "D"
    return MU_MARKERS[row["mu_layout"]]


def x_for_row(row):
    return X_POS[row["plot_group"]] if row["kind"] in ["uniform", "one_vmf"] else X_POS[row["shape_case"]]


def jitter_for_row(row):
    if row["kind"] in ["uniform", "one_vmf"]:
        return 0.035 * (int(row["rep"]) - 0.5)
    return LAYOUT_X_OFFSET[row["mu_layout"]] + 0.026 * int(row["rep"])


def set_shape_axis(ax):
    order = ["uniform", "one_vmf"] + SHAPE_CASES
    ax.set_xticks([X_POS[x] for x in order])
    ax.set_xticklabels([SHAPE_LABELS[x] for x in order], rotation=25, ha="right")
    ax.margins(x=0.035)


def add_main_legend(fig, y=0.94):
    handles = [plt.Line2D([0], [0], marker=MU_MARKERS[l], color=MU_COLORS[l], linestyle="none", markeredgecolor="black", markersize=8, label=MU_LABELS[l]) for l in MU_LAYOUTS]
    handles += [plt.Line2D([0], [0], marker="X", color="black", linestyle="none", markeredgecolor="black", markersize=8, label="Uniform"), plt.Line2D([0], [0], marker="D", color=get_single_color(7), linestyle="none", markeredgecolor="black", markersize=8, label="1 vMF")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=5, frameon=False, fontsize=10.5)


def group_keys():
    return [("uniform", "baseline", "baseline"), ("one_vmf", "baseline", "baseline")] + [("mixture", s, l) for s in SHAPE_CASES for l in MU_LAYOUTS]


def rows_for_group(rows, kind, shape_case, mu_layout):
    return [r for r in rows if r["kind"] == kind and (kind != "mixture" or (r["shape_case"] == shape_case and r["mu_layout"] == mu_layout))]


def aggregate_for_metric(rows, metric):
    groups = []
    for kind, shape_case, mu_layout in group_keys():
        vals = np.asarray([float(r.get(metric, np.nan)) for r in rows_for_group(rows, kind, shape_case, mu_layout)], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            groups.append({"row": {"kind": kind, "shape_case": shape_case, "mu_layout": mu_layout, "plot_group": kind if kind != "mixture" else f"{mu_layout}_{shape_case}", "rep": 0}, "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "n": int(len(vals))})
    return groups


def aggregated_scatter_points(rows, x_key, y_key):
    groups = []
    for kind, shape_case, mu_layout in group_keys():
        group = rows_for_group(rows, kind, shape_case, mu_layout)
        xy = np.asarray([[float(r.get(x_key, np.nan)), float(r.get(y_key, np.nan))] for r in group], dtype=np.float64)
        xy = xy[np.isfinite(xy).all(axis=1)] if len(xy) else xy
        if len(xy):
            groups.append({"row": {"kind": kind, "shape_case": shape_case, "mu_layout": mu_layout, "plot_group": kind if kind != "mixture" else f"{mu_layout}_{shape_case}", "rep": 0}, "x": float(xy[:, 0].mean()), "y": float(xy[:, 1].mean()), "xerr": float(xy[:, 0].std(ddof=1)) if len(xy) > 1 else 0.0, "yerr": float(xy[:, 1].std(ddof=1)) if len(xy) > 1 else 0.0, "n": int(len(xy))})
    return groups


def plot_metric_grid(rows, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.4), squeeze=False)
    for ax, (metric, title) in zip(axes.ravel(), PLOT_METRICS):
        for row in rows:
            y = float(row.get(metric, np.nan))
            if np.isfinite(y):
                ax.scatter(x_for_row(row) + jitter_for_row(row), y, s=24, alpha=0.16, color=row_color(row), marker=row_marker(row), edgecolors="none")
        for item in aggregate_for_metric(rows, metric):
            row = item["row"]
            x = x_for_row(row) + (LAYOUT_X_OFFSET[row["mu_layout"]] if row["kind"] == "mixture" else 0.0)
            ax.errorbar(x, item["mean"], yerr=item["std"], fmt=row_marker(row), color=row_color(row), markeredgecolor="black", markeredgewidth=0.8, markersize=7.5, linewidth=1.15, capsize=3, alpha=0.96)
        set_shape_axis(ax)
        ax.set_title(title, fontsize=12.5, pad=9)
        ax.grid(alpha=0.22)
    fig.suptitle("LDD metrics (source) and induced coupling metrics", fontsize=18, y=0.985)
    add_main_legend(fig, y=0.935)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.055, right=0.985, hspace=0.48, wspace=0.28)
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_scatter_grid(rows, pairs, save_path, suptitle):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.1, 9.2), squeeze=False)
    for ax, (x_key, y_key, x_lab, y_lab, title) in zip(axes.ravel(), pairs):
        for item in aggregated_scatter_points(rows, x_key, y_key):
            row = item["row"]
            ax.errorbar(item["x"], item["y"], xerr=item["xerr"], yerr=item["yerr"], fmt=row_marker(row), color=row_color(row), markeredgecolor="black", markeredgewidth=0.8, markersize=7.5, linewidth=1.1, capsize=3, alpha=0.96)
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        ax.set_title(title, fontsize=12.5, pad=9)
        ax.grid(alpha=0.22)
    fig.suptitle(suptitle, fontsize=18, y=0.99)
    add_main_legend(fig, y=0.94)
    fig.subplots_adjust(top=0.82, bottom=0.09, left=0.075, right=0.985, hspace=0.40, wspace=0.30)
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_sphere_points(ax, Z, labels, title):
    u, v = np.linspace(0, 2.0 * np.pi, 32), np.linspace(0.0, np.pi, 16)
    ax.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)), np.outer(np.ones_like(u), np.cos(v)), linewidth=0.20, alpha=0.12)
    if np.all(labels < 0):
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=5, alpha=0.38, color=get_single_color(0), depthshade=False)
    else:
        for k in sorted(set(labels.tolist())):
            idx = labels == k
            ax.scatter(Z[idx, 0], Z[idx, 1], Z[idx, 2], s=5, alpha=0.48, color=get_single_color(k + 2), depthshade=False)
    ax.set_title(title, fontsize=8.5, pad=2)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.05, 1.05]); ax.set_ylim([-1.05, 1.05]); ax.set_zlim([-1.05, 1.05])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def gallery_title(spec):
    return SHAPE_LABELS[spec["plot_group"]] if spec["kind"] in ["uniform", "one_vmf"] else f"{spec['mu_layout']} / {SHAPE_LABELS[spec['shape_case']]}"


def plot_distribution_gallery_for_rep(save_path, n_plot, rep, args):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    specs = build_specs(gallery=True)
    fig = plt.figure(figsize=(15.0, 17.5))
    gs = GridSpec(5, 6, figure=fig, height_ratios=[1.0, 1.05, 1.05, 1.05, 1.05], hspace=0.10, wspace=0.02)
    for i, spec in enumerate(specs[:2]):
        ax = fig.add_subplot([gs[0, 1:3], gs[0, 3:5]][i], projection="3d")
        offset = spec_offset(spec)
        rng = np.random.default_rng(args.seed + 10000 * rep + offset + 707)
        design = materialize_distribution_design(spec, args.design_seed + 10000 * rep + offset)
        Z, labels, _, _ = sample_distribution(spec, n_plot, rng, design)
        plot_sphere_points(ax, Z, labels, gallery_title(spec))
    for idx, spec in enumerate(specs[2:]):
        ax = fig.add_subplot(gs[1 + idx // 3, 2 * (idx % 3):2 * (idx % 3) + 2], projection="3d")
        offset = spec_offset(spec)
        rng = np.random.default_rng(args.seed + 10000 * rep + offset + 707)
        design = materialize_distribution_design(spec, args.design_seed + 10000 * rep + offset)
        Z, labels, _, _ = sample_distribution(spec, n_plot, rng, design)
        plot_sphere_points(ax, Z, labels, gallery_title(spec))
    fig.suptitle(f"Distribution gallery, run {rep}", fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.975, bottom=0.015)
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def main(args):
    plot_root, csv_root = os.path.join(args.out_root, "plots"), os.path.join(args.out_root, "csv")
    os.makedirs(plot_root, exist_ok=True)
    os.makedirs(csv_root, exist_ok=True)
    print("\nOutput folder:")
    print(args.out_root)
    print("\nDesign:")
    print("Baselines: Uniform and 1 vMF.")
    print(f"1 vMF kappa is sampled per run from {ONE_MODE_KAPPA_RANGE}.")
    print("Mixture cases: 3 center layouts x 4 component-shape cases; equal weights.")
    print("A and B are sampled independently from the same run-specific parameters; B is globally rotated.")
    print("Only intrinsic spherical distance is used.\n")
    print("Running...")
    rows = []
    for spec in build_specs(gallery=False):
        print("\n============================================================")
        print(f"case={spec['case']}")
        print(f"kind={spec['kind']} | centers={spec['mu_layout']} | shape={spec['shape_case']}")
        print("============================================================")
        for rep in range(args.n_reps):
            row = run_one(spec, rep, args)
            rows.append(row)
            print(f"rep={rep:03d} | LDD var src={row['ldd_variance_source']:.6e} | rank src={row['ldd_effective_rank_source']:.4f} | row_nonunif={row['coupling_row_nonuniformity']:.6e} | row_peak={row['coupling_row_peak_ratio_mean']:.4f} | soft_blocks={row['coupling_soft_block_count']:.3f} | graph_frag={row['coupling_graph_fragmentation']:.3f} | sinkhorn_err={max(row['sinkhorn_row_marginal_max_error'], row['sinkhorn_col_marginal_max_error']):.2e}")
    results_csv = os.path.join(csv_root, "sensitivity_results.csv")
    grouped_csv = os.path.join(csv_root, "sensitivity_grouped_summary.csv")
    corr_csv = os.path.join(csv_root, "sensitivity_metric_correlations.csv")
    corr_long_csv = os.path.join(csv_root, "sensitivity_metric_correlations_long.csv")
    write_csv(rows, results_csv)
    save_grouped_csv(rows, grouped_csv)
    save_correlation_csvs(rows, corr_csv, corr_long_csv)
    p1 = os.path.join(plot_root, "01_source_ldd_and_coupling_metrics_2x3.png")
    p2 = os.path.join(plot_root, "02_source_ldd_vs_row_level_coupling_metrics_2x2.png")
    p3 = os.path.join(plot_root, "03_row_level_vs_graph_level_coupling_metrics_2x2.png")
    plot_metric_grid(rows, p1)
    plot_scatter_grid(rows, LDD_ROW_PAIRS, p2, "LDD metrics (source) vs row-level coupling metrics")
    plot_scatter_grid(rows, ROW_STRUCTURE_PAIRS, p3, "Row-level vs graph-level coupling metrics")
    gallery_paths = []
    if args.save_sphere_gallery:
        for rep in range(args.gallery_reps):
            p = os.path.join(plot_root, f"00_distribution_gallery_run_{rep:02d}.png")
            plot_distribution_gallery_for_rep(p, args.n_plot_sphere, rep, args)
            gallery_paths.append(p)
    print("\nSaved CSV files:")
    for p in [results_csv, grouped_csv, corr_csv, corr_long_csv]:
        print(p)
    print("\nSaved plots:")
    for p in gallery_paths + [p1, p2, p3]:
        print(p)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="sensitivity")
    parser.add_argument("--n-points", type=int, default=2000)
    parser.add_argument("--n-reps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--design-seed", type=int, default=777)
    parser.add_argument("--radius-grid", type=str, default="log", choices=["log", "linear"])
    parser.add_argument("--r-bins", type=int, default=100)
    parser.add_argument("--r-min", type=float, default=1e-4)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iter", type=int, default=2000)
    parser.add_argument("--ldd-variance-floor", type=float, default=1e-4)
    parser.add_argument("--soft-block-tau", type=float, default=0.03)
    parser.add_argument("--soft-block-svd-rank", type=int, default=32)
    parser.add_argument("--graph-topk", type=int, default=12)
    parser.add_argument("--graph-fragmentation-tau", type=float, default=0.08)
    parser.add_argument("--graph-min-signal", type=float, default=1e-8)
    parser.add_argument("--save-sphere-gallery", action="store_true")
    parser.add_argument("--gallery-reps", type=int, default=2)
    parser.add_argument("--n-plot-sphere", type=int, default=650)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
