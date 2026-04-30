import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import ot

from matplotlib.colors import to_rgb
from sklearn.decomposition import PCA

from utils_GW import ZukoFlowMap
from LDD_run import LDD, compute_summary, plot_ldd_signatures, print_summary
from colors import *

# --------------------------------------------------
# loading
# --------------------------------------------------

def load_npz_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")
    return np.load(path)


def load_geom_file(path):
    """
    Load saved points + geodesics.

    MNIST case:
        z_mu
        D_geodesic

    S2 case:
        Z
        C_g
    """
    data = load_npz_file(path)

    if "z_mu" in data.files:
        Z = torch.from_numpy(data["z_mu"]).float()
    elif "Z" in data.files:
        Z = torch.from_numpy(data["Z"]).float()
    else:
        raise ValueError(f"Could not find latent/point array in: {path}")

    if "D_geodesic" in data.files:
        C = torch.from_numpy(data["D_geodesic"]).float()
    elif "C_g" in data.files:
        C = torch.from_numpy(data["C_g"]).float()
    else:
        raise ValueError(f"Could not find geodesic matrix in: {path}")

    payload = {
        "Z": Z,
        "C": C,
        "keys": list(data.files),
        "geom_file": path,
    }

    if "x" in data.files:
        payload["x"] = torch.from_numpy(data["x"]).float()
    if "y" in data.files:
        payload["y"] = torch.from_numpy(data["y"]).long()
    elif "labels" in data.files:
        payload["y"] = torch.from_numpy(data["labels"]).long()
    if "chosen_idx_within_latent_file" in data.files:
        payload["chosen_idx_within_latent_file"] = torch.from_numpy(
            data["chosen_idx_within_latent_file"]
        ).long()
    if "dataset_indices_raw" in data.files:
        payload["dataset_indices_raw"] = torch.from_numpy(data["dataset_indices_raw"]).long()
    if "canonical_order" in data.files:
        payload["canonical_order"] = torch.from_numpy(data["canonical_order"]).long()
    if "center_idx" in data.files:
        payload["center_idx"] = torch.from_numpy(data["center_idx"]).long()

    return payload


# --------------------------------------------------
# helpers for safe naming
# --------------------------------------------------

def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def make_ldd_paths(artifact_root, plot_root, experiment_name, model_folder, role, geom_path, r_bins, tau):
    artifact_dir = os.path.join(artifact_root, experiment_name, model_folder)
    plot_dir = os.path.join(plot_root, experiment_name, model_folder)

    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    tau_tag = str(tau).replace(".", "")
    name = f"{role}_{stem(geom_path)}_H_ldd_rb{r_bins}_tau{tau_tag}"

    npz_path = os.path.join(artifact_dir, f"{name}.npz")
    meta_path = os.path.join(artifact_dir, f"{name}_metadata.json")
    summary_path = os.path.join(artifact_dir, f"{name}_summary.json")
    plot_path = os.path.join(plot_dir, f"{name}.png")

    return npz_path, meta_path, summary_path, plot_path


def make_map_paths(
    artifact_root,
    plot_root,
    experiment_name,
    model_folder,
    source_geom_path,
    target_geom_path,
    epsilon,
    max_iter,
    threshold,
):
    artifact_dir = os.path.join(artifact_root, experiment_name, model_folder)
    plot_dir = os.path.join(plot_root, experiment_name, model_folder)

    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    eps_tag = str(epsilon).replace(".", "")
    thr_tag = f"{threshold:.0e}".replace("-", "m").replace("+", "")
    name = (
        f"P_coupling_{stem(source_geom_path)}__to__{stem(target_geom_path)}"
        f"_eps{eps_tag}_gwiter{max_iter}_tol{thr_tag}"
    )

    npz_path = os.path.join(artifact_dir, f"{name}.npz")
    meta_path = os.path.join(artifact_dir, f"{name}_metadata.json")
    plot_path = os.path.join(plot_dir, f"{name}.png")

    return npz_path, meta_path, plot_path


# --------------------------------------------------
# LDD from loaded geodesics
# --------------------------------------------------

def compute_ldd_from_geom_payload(geom_payload, r_bins, tau, r_max=None, n_centers=None):
    """
    Compute LDD directly from already saved geometry/geodesics.

    Important:
    This uses the exact same point ordering as the loaded geodesic file.
    """
    Z = geom_payload["Z"]
    C = geom_payload["C"]

    if r_max is None:
        r_max = float(C.max().item())

    center_idx = None
    if n_centers is not None:
        n_centers = min(n_centers, Z.shape[0])
        center_idx = torch.arange(n_centers, dtype=torch.long)

    H, r, center_idx, z_points, C_centers = LDD(
        Z,
        C,
        r_bins=r_bins,
        r_max=r_max,
        center_idx=center_idx,
    )

    summary = compute_summary(H, tau)

    payload = {
        "H": H,
        "r": r,
        "center_idx": center_idx,
        "z_points": z_points,
        "C_centers": C_centers,
        "summary": summary,
        "r_max": r_max,
    }
    return payload


def save_ldd_payload(save_path, geom_payload, ldd_payload):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    out = {
        "H": ldd_payload["H"].cpu().numpy().astype(np.float32),
        "r": ldd_payload["r"].cpu().numpy().astype(np.float32),
        "center_idx": ldd_payload["center_idx"].cpu().numpy().astype(np.int64),
    }

    if "Z" in geom_payload:
        out["z_mu"] = geom_payload["Z"].cpu().numpy().astype(np.float32)
    if "x" in geom_payload:
        out["x"] = geom_payload["x"].cpu().numpy().astype(np.float32)
    if "y" in geom_payload:
        out["y"] = geom_payload["y"].cpu().numpy().astype(np.int64)
    if "geom_file" in geom_payload:
        out["geom_file"] = np.array(geom_payload["geom_file"])
    if "chosen_idx_within_latent_file" in geom_payload:
        out["chosen_idx_within_latent_file"] = (
            geom_payload["chosen_idx_within_latent_file"].cpu().numpy().astype(np.int64)
        )
    if "dataset_indices_raw" in geom_payload:
        out["dataset_indices_raw"] = geom_payload["dataset_indices_raw"].cpu().numpy().astype(np.int64)
    if "canonical_order" in geom_payload:
        out["canonical_order"] = geom_payload["canonical_order"].cpu().numpy().astype(np.int64)

    np.savez_compressed(save_path, **out)


def save_summary_json(summary, tau, out_path, experiment_name):
    json_summary = {
        "experiment_name": experiment_name,
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


# --------------------------------------------------
# algorithm 1: phase 1 and 2
# --------------------------------------------------

def compute_M_distance(H_a, H_b, normalize=True):
    """
    Signature cost matrix M.

    M_ij = ||H_a[i] - H_b[j]||_2^2
    """
    M = torch.cdist(H_a, H_b, p=2) ** 2

    if normalize:
        max_val = M.max().item()
        if max_val > 0:
            M = M / max_val

    return M


def uniform_marginals(n_a, n_b):
    a = np.ones(n_a, dtype=np.float64) / n_a
    b = np.ones(n_b, dtype=np.float64) / n_b
    return a, b


def normalize_distance_matrix(C):
    C = C.clone()
    max_val = C.max().item()
    if max_val > 0:
        C = C / max_val
    return C

def init_sinkhorn(M, init_epsilon=0.1, num_itermax=5000):
    n_a, n_b = M.shape
    a, b = uniform_marginals(n_a, n_b)

    pi0 = ot.sinkhorn(
        a,
        b,
        M.detach().cpu().numpy(),
        reg=init_epsilon,
        numItermax=num_itermax,
    )

    return torch.from_numpy(pi0).float()


def solve_gw_coupling(C_a, C_b, epsilon, threshold, max_iter, pi0=None):
    n_a = C_a.shape[0]
    n_b = C_b.shape[0]
    p, q = uniform_marginals(n_a, n_b)

    P = ot.gromov.entropic_gromov_wasserstein(
        C1=C_a.detach().cpu().numpy(),
        C2=C_b.detach().cpu().numpy(),
        p=p,
        q=q,
        loss_fun="square_loss",
        epsilon=epsilon,
        G0=None if pi0 is None else pi0.detach().cpu().numpy(),
        max_iter=max_iter,
        tol=threshold,
        #solver="PPA"
    )

    return torch.from_numpy(P).float()


def coupling_euclidean_loss_target_to_source(Z_source, Z_target_mapped, P):
    """
    Algorithm 1 local Euclidean loss for map target -> source.

    Z_source:        (n_source, d), A-space
    Z_target_mapped: (n_target, d), mapped B -> A
    P:               (n_source, n_target)

    Loss:
        sum_ij P_ij || T(z_target_j) - z_source_i ||^2
    """
    diff = Z_source[:, None, :] - Z_target_mapped[None, :, :]
    sqdist = (diff ** 2).sum(dim=2)
    return (P * sqdist).sum()

def train_flow_target_to_source(
    Z_source,
    Z_target,
    P,
    transforms=4,
    hidden_features=(64, 64),
    bins=8,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=2000,
    device="cpu",
    verbose=False,
):
    """
    Train flow T: target -> source.

    In MNIST case:
        source = A
        target = B
        T      = B -> A
    """
    Z_source_t = torch.tensor(Z_source, dtype=torch.float32, device=device)
    Z_target_t = torch.tensor(Z_target, dtype=torch.float32, device=device)
    P_t = torch.tensor(P, dtype=torch.float32, device=device)

    model = ZukoFlowMap(
        latent_dim=Z_target_t.shape[1],
        transforms=transforms,
        hidden_features=hidden_features,
        bins=bins,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = None
    patience = 80
    min_delta = 1e-4
    wait = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        Z_target_mapped = model(Z_target_t)  # B -> A

        loss = coupling_euclidean_loss_target_to_source(
            Z_source=Z_source_t,
            Z_target_mapped=Z_target_mapped,
            P=P_t,
        )

        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            if verbose:
                print(f"[flow target->source] early stop at epoch={epoch} best={best_loss:.6f}")
            break

        if verbose and (epoch % 200 == 0):
            print(f"[flow target->source] epoch={epoch} loss={loss.item():.6f} best={best_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def compute_map_neural_GM(
    Z_a,
    Z_b,
    C_a,
    C_b,
    H_a,
    H_b,
    epsilon,
    lr,
    threshold,
    max_iter,
    is_s2=False,
):
    """
    Lige nu kun coupling:
    M, pi0, P
    """
    if H_a.shape[0] != C_a.shape[0]:
        raise ValueError(
            f"Mismatch on source side: H_a has shape {tuple(H_a.shape)} but C_a has shape {tuple(C_a.shape)}"
        )
    if H_b.shape[0] != C_b.shape[0]:
        raise ValueError(
            f"Mismatch on target side: H_b has shape {tuple(H_b.shape)} but C_b has shape {tuple(C_b.shape)}"
        )
    if Z_a.shape[0] != C_a.shape[0]:
        raise ValueError(
            f"Mismatch on source side: Z_a has shape {tuple(Z_a.shape)} but C_a has shape {tuple(C_a.shape)}"
        )
    if Z_b.shape[0] != C_b.shape[0]:
        raise ValueError(
            f"Mismatch on target side: Z_b has shape {tuple(Z_b.shape)} but C_b has shape {tuple(C_b.shape)}"
        )

    # phase 1
    M = compute_M_distance(H_a, H_b)
    pi0 = init_sinkhorn(M, init_epsilon=0.1, num_itermax=5000)

    # normalize geometry before GW
    C_a_gw = normalize_distance_matrix(C_a)
    C_b_gw = normalize_distance_matrix(C_b)

    # phase 2
    P = solve_gw_coupling(C_a_gw, C_b_gw, epsilon, threshold, max_iter, pi0=pi0)

    if not torch.isfinite(P).all() or P.sum().item() == 0.0:
        raise ValueError("GW coupling collapsed: P is non-finite or all zero.")

    # phase 3-4:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    flow_model = train_flow_target_to_source(
        Z_source=Z_a.cpu().numpy(),   # A, decoder/reference space
        Z_target=Z_b.cpu().numpy(),   # B, points to map
        P=P.cpu().numpy(),            # rows=A, cols=B
        transforms=4,
        hidden_features=(64, 64),
        bins=8,
        lr=lr,
        epochs=800,
        device=device,
        verbose=True,
    )

    return {
        "M": M,
        "pi0": pi0,
        "P": P,
        "flow_target_to_source": flow_model,
    }


# --------------------------------------------------
# saving / plotting
# --------------------------------------------------

def save_coupling_results(save_path, M, pi0, P):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez_compressed(
        save_path,
        M=M.detach().cpu().numpy().astype(np.float32),
        pi0=pi0.detach().cpu().numpy().astype(np.float32),
        P=P.detach().cpu().numpy().astype(np.float32),
    )


def save_metadata(save_path, metadata):
    with open(save_path, "w") as f:
        json.dump(metadata, f, indent=2)


def plot_P_matrix(P, save_path, title, log_scale=True, eps=1e-12):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    P_np = P.detach().cpu().numpy()

    if log_scale:
        P_np = np.log10(P_np + eps)

    cmap = color_segment()
    vmin = P_np.min()
    vmax = P_np.max()
 

    plt.figure(figsize=(5, 4))
    plt.imshow(P_np, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="log10(P)" if log_scale else "P")
    plt.xlabel("target")
    plt.ylabel("source")
    plt.title(title + (" (log scale)" if log_scale else ""))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def sort_coupling_for_plot(P, source_geom, target_geom):
    """
    Sort rows/cols of P by class labels only for visualization.
    Does not change the saved raw coupling.
    """
    if "y" not in source_geom or "y" not in target_geom:
        raise ValueError("Need labels y in both source and target payloads to sort P for plotting.")

    y_source = source_geom["y"]
    y_target = target_geom["y"]

    perm_source = torch.argsort(y_source)
    perm_target = torch.argsort(y_target)

    P_sorted = P[perm_source][:, perm_target]

    return {
        "P_sorted": P_sorted,
        "perm_source": perm_source,
        "perm_target": perm_target,
        "y_source_sorted": y_source[perm_source],
        "y_target_sorted": y_target[perm_target],
    }

def plot_P_matrix_sorted(P, source_geom, target_geom, save_path, title, log_scale=True, eps=1e-12):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sort_payload = sort_coupling_for_plot(P, source_geom, target_geom)
    P_np = sort_payload["P_sorted"].detach().cpu().numpy()

    if log_scale:
        P_np = np.log10(P_np + eps)

    cmap = color_segment()
    vmin = P_np.min()
    vmax = P_np.max()

    plt.figure(figsize=(5, 4))
    plt.imshow(P_np, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="log10(P)" if log_scale else "P")
    plt.xlabel("target (sorted by class)")
    plt.ylabel("source (sorted by class)")
    plt.title(title + (" (sorted by class, log scale)" if log_scale else " (sorted by class)"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def ldd_covariance_spectrum(H, eps=1e-12):
    """
    Compute normalized covariance eigenvalue spectrum of LDD signatures.
    """
    H = torch.as_tensor(H, dtype=torch.float32)
    Hc = H - H.mean(dim=0, keepdim=True)

    Cov = (Hc.T @ Hc) / max(H.shape[0] - 1, 1)

    eigvals = torch.linalg.eigvalsh(Cov)
    eigvals = torch.clamp(eigvals, min=0.0)
    eigvals = torch.flip(eigvals, dims=[0])

    if eigvals.sum() > eps:
        explained = eigvals / eigvals.sum()
    else:
        explained = eigvals

    return explained


def plot_ldd_spectrum(H, save_path, title):
    """
    Plot explained variance of LDD covariance eigenvalues.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    explained = ldd_covariance_spectrum(H)

    plt.figure(figsize=(5, 4))
    plt.plot(
        np.arange(1, len(explained) + 1),
        explained.numpy(),
        marker="o",
        markersize=3,
        linewidth=1.5,
        color=get_single_color(6),
    )

    plt.xlabel("Component")
    plt.ylabel("Explained variance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def darken_color(color, factor=0.55):
    rgb = np.array(to_rgb(color), dtype=float)
    return tuple(np.clip(factor * rgb, 0.0, 1.0))

def plot_source_and_mapped_2d(
    Z_source,
    Z_target,
    flow_model,
    source_geom,
    target_geom,
    save_path,
    device="cpu",
):
    if "y" not in source_geom or "y" not in target_geom:
        raise ValueError("Need y labels in both source_geom and target_geom for class-colored plot.")

    flow_model = flow_model.to(device)
    flow_model.eval()

    with torch.no_grad():
        Z_target_mapped = flow_model(Z_target.to(device)).cpu()

    A = Z_source.cpu().numpy()
    B = Z_target_mapped.cpu().numpy()

    y_source = source_geom["y"].cpu().numpy()
    y_target = target_geom["y"].cpu().numpy()

    classes = np.unique(np.concatenate([y_source, y_target]))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(5, 5))

    for c in classes:
        col_a = get_single_color(int(c + 4))
        col_b = darken_color(col_a, factor=0.55)

        idx_a = y_source == c
        idx_b = y_target == c

        plt.scatter(
            A[idx_a, 0],
            A[idx_a, 1],
            s=24,
            alpha=0.75,
            color=col_a,
            marker="o",
            label=f"A class {c}",
        )

        plt.scatter(
            B[idx_b, 0],
            B[idx_b, 1],
            s=32,
            alpha=0.95,
            color=col_b,
            marker="x",
            label=f"T(B) class {c}",
        )

    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("A latent space: source vs mapped target")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_source_and_mapped_3d_s2(
    Z_source,
    Z_target,
    flow_model,
    source_geom,
    target_geom,
    save_path,
    device="cpu",
):
    if Z_source.shape[1] != 3 or Z_target.shape[1] != 3:
        raise ValueError("S2 3D plot requires 3-dimensional points.")

    flow_model = flow_model.to(device)
    flow_model.eval()

    with torch.no_grad():
        Z_target_mapped = flow_model(Z_target.to(device)).cpu()

    A = Z_source.cpu().numpy()
    B = Z_target_mapped.cpu().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    has_labels = "y" in source_geom and "y" in target_geom

    if has_labels:
        y_source = source_geom["y"].cpu().numpy()
        y_target = target_geom["y"].cpu().numpy()
        classes = np.unique(np.concatenate([y_source, y_target]))

                # plot A first
        for c in classes:
            col_a = get_single_color(int(c + 4))
            idx_a = y_source == c

            ax.scatter(
                A[idx_a, 0], A[idx_a, 1], A[idx_a, 2],
                s=18,
                alpha=0.65,
                color=col_a,
                marker="o",
                depthshade=False,
                label=f"A {c}",
            )

        # plot mapped B as black/dark points instead of x
        for c in classes:
            col_a = get_single_color(int(c + 4))
            col_b = darken_color(col_a, factor=0.35)
            idx_b = y_target == c

            ax.scatter(
                B[idx_b, 0], B[idx_b, 1], B[idx_b, 2],
                s=26,
                alpha=1.0,
                color=col_b,
                marker=".",
                depthshade=False,
                label=f"T(B) {c}",
            )

    else:
        col_a = get_single_color(0)
        col_b = darken_color(col_a, factor=0.35)

        ax.scatter(
            A[:, 0], A[:, 1], A[:, 2],
            s=10,
            alpha=0.22,
            color=col_a,
            marker="o",
            depthshade=False,
            label="A",
        )

        ax.scatter(
            B[:, 0], B[:, 1], B[:, 2],
            s=85,
            alpha=1.0,
            color=col_b,
            marker="x",
            linewidths=2.5,
            depthshade=False,
            label="T(B)",
        )

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_zlabel("z3")
    ax.set_title("S2: source vs mapped target")

    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# --------------------------------------------------
# evaluating
# --------------------------------------------------

def evaluate_target_to_source_map(Z_source, Z_target, P, flow_model, device="cpu", eps=1e-12):
    """
    Evaluate learned map T: target -> source.

    source = A
    target = B
    P rows = source, cols = target
    """
    Z_source = Z_source.to(device)
    Z_target = Z_target.to(device)
    P = P.to(device)

    flow_model = flow_model.to(device)
    flow_model.eval()

    with torch.no_grad():
        Z_target_mapped = flow_model(Z_target)

        loss_identity_like = coupling_euclidean_loss_target_to_source(
            Z_source=Z_source,
            Z_target_mapped=Z_target,
            P=P,
        )

        loss_flow = coupling_euclidean_loss_target_to_source(
            Z_source=Z_source,
            Z_target_mapped=Z_target_mapped,
            P=P,
        )

        col_mass = P.sum(dim=0, keepdim=True).T
        Z_bary = (P.T @ Z_source) / (col_mass + eps)

        loss_bary = coupling_euclidean_loss_target_to_source(
            Z_source=Z_source,
            Z_target_mapped=Z_bary,
            P=P,
        )

        print("\n=== target -> source map evaluation ===")
        print("Z_source:", tuple(Z_source.shape))
        print("Z_target:", tuple(Z_target.shape))
        print("Z_target_mapped:", tuple(Z_target_mapped.shape))

        print("identity-like loss:", loss_identity_like.item())
        print("flow loss:", loss_flow.item())
        print("barycentric lower-bound loss:", loss_bary.item())
        print("flow / identity-like:", (loss_flow / loss_identity_like).item())
        print("flow / barycentric:", (loss_flow / loss_bary).item())

        print("source std mean:", Z_source.std(dim=0).mean().item())
        print("target std mean:", Z_target.std(dim=0).mean().item())
        print("mapped std mean:", Z_target_mapped.std(dim=0).mean().item())
        print("bary std mean:", Z_bary.std(dim=0).mean().item())

        print("source pairwise mean:", torch.cdist(Z_source, Z_source).mean().item())
        print("target pairwise mean:", torch.cdist(Z_target, Z_target).mean().item())
        print("mapped pairwise mean:", torch.cdist(Z_target_mapped, Z_target_mapped).mean().item())
        print("bary pairwise mean:", torch.cdist(Z_bary, Z_bary).mean().item())



# --------------------------------------------------
# main
# --------------------------------------------------

def main(args):
    # load geometry only; never modify these files
    source_geom = load_geom_file(args.source_geom_file)
    target_geom = load_geom_file(args.target_geom_file)

    source_model_folder = os.path.basename(os.path.dirname(args.source_geom_file))
    target_model_folder = os.path.basename(os.path.dirname(args.target_geom_file))

    if source_model_folder != target_model_folder:
        model_folder = f"{source_model_folder}__to__{target_model_folder}"
    else:
        model_folder = source_model_folder

    # compute LDD directly from the loaded geodesic files
    source_ldd = compute_ldd_from_geom_payload(
        source_geom,
        r_bins=args.r_bins,
        tau=args.tau,
        r_max=args.r_max,
        n_centers=args.n_centers,
    )
    target_ldd = compute_ldd_from_geom_payload(
        target_geom,
        r_bins=args.r_bins,
        tau=args.tau,
        r_max=args.r_max,
        n_centers=args.n_centers,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # save source LDD
    src_ldd_path, src_meta_path, src_summary_path, src_plot_path = make_ldd_paths(
        args.artifact_root,
        args.plot_root,
        args.experiment_name,
        model_folder,
        "source",
        args.source_geom_file,
        args.r_bins,
        args.tau,
    )

    src_spectrum_path = src_plot_path.replace(".png", "_spectrum.png")
    plot_ldd_spectrum(
        source_ldd["H"],
        save_path=src_spectrum_path,
        title="LDD spectrum (source)",
    )

    save_ldd_payload(src_ldd_path, source_geom, source_ldd)
    save_metadata(
        src_meta_path,
        {
            "experiment_name": args.experiment_name,
            "role": "source",
            "source_geodesic_file": args.source_geom_file,
            "n_points": int(source_geom["Z"].shape[0]),
            "n_centers": int(source_ldd["center_idx"].shape[0]),
            "r_bins": int(args.r_bins),
            "r_max": float(source_ldd["r_max"]),
            "tau": float(args.tau),
        },
    )
    save_summary_json(source_ldd["summary"], args.tau, src_summary_path, args.experiment_name)
    plot_ldd_signatures(
        source_ldd["r"],
        source_ldd["H"],
        source_ldd["summary"]["H_mean"],
        savepath=src_plot_path,
        title=f"LDD signatures (source)",
    )

    # save target LDD
    tgt_ldd_path, tgt_meta_path, tgt_summary_path, tgt_plot_path = make_ldd_paths(
        args.artifact_root,
        args.plot_root,
        args.experiment_name,
        model_folder,
        "target",
        args.target_geom_file,
        args.r_bins,
        args.tau,
    )

    tgt_spectrum_path = tgt_plot_path.replace(".png", "_spectrum.png")
    plot_ldd_spectrum(
        target_ldd["H"],
        save_path=tgt_spectrum_path,
        title="LDD spectrum (target)",
    )

    save_ldd_payload(tgt_ldd_path, target_geom, target_ldd)
    save_metadata(
        tgt_meta_path,
        {
            "experiment_name": args.experiment_name,
            "role": "target",
            "target_geodesic_file": args.target_geom_file,
            "n_points": int(target_geom["Z"].shape[0]),
            "n_centers": int(target_ldd["center_idx"].shape[0]),
            "r_bins": int(args.r_bins),
            "r_max": float(target_ldd["r_max"]),
            "tau": float(args.tau),
        },
    )
    save_summary_json(target_ldd["summary"], args.tau, tgt_summary_path, args.experiment_name)
    plot_ldd_signatures(
        target_ldd["r"],
        target_ldd["H"],
        target_ldd["summary"]["H_mean"],
        savepath=tgt_plot_path,
        title=f"LDD signatures (target)",
    )

    Z_a = source_geom["Z"]
    Z_b = target_geom["Z"]
    C_a = source_geom["C"]
    C_b = target_geom["C"]
    H_a = source_ldd["H"]
    H_b = target_ldd["H"]

    result = compute_map_neural_GM(
        Z_a=Z_a,
        Z_b=Z_b,
        C_a=C_a,
        C_b=C_b,
        H_a=H_a,
        H_b=H_b,
        epsilon=args.epsilon,
        lr=args.lr,
        threshold=args.threshold,
        max_iter=args.max_iter,
        is_s2=args.is_s2,
    )

    # print summary 
    evaluate_target_to_source_map(
    Z_source=Z_a,
    Z_target=Z_b,
    P=result["P"],
    flow_model=result["flow_target_to_source"],
    device=device,
    )

    # evaluating plot 2d
    if "y" in source_geom and "y" in target_geom:
        plot_source_and_mapped_2d(
            Z_source=Z_a,
            Z_target=Z_b,
            flow_model=result["flow_target_to_source"],
            source_geom=source_geom,
            target_geom=target_geom,
            save_path=os.path.join(
                args.plot_root,
                args.experiment_name,
                model_folder,
                f"mapped_target_to_source_2d_eps{str(args.epsilon).replace('.', '')}.png",
            ),
            device=device,
        )

    if args.is_s2:
        plot_source_and_mapped_3d_s2(
            Z_source=Z_a,
            Z_target=Z_b,
            flow_model=result["flow_target_to_source"],
            source_geom=source_geom,
            target_geom=target_geom,
            save_path=os.path.join(
                args.plot_root,
                args.experiment_name,
                model_folder,
                f"mapped_target_to_source_3d_s2_eps{str(args.epsilon).replace('.', '')}.png",
            ),
            device=device,
        )

    print("M min/max/std:", result["M"].min().item(), result["M"].max().item(), result["M"].std().item())
    print("pi0 min/max/std:", result["pi0"].min().item(), result["pi0"].max().item(), result["pi0"].std().item())
    print("P min/max/std:", result["P"].min().item(), result["P"].max().item(), result["P"].std().item())
    print("pi0 row sums min/max:", result["pi0"].sum(dim=1).min().item(), result["pi0"].sum(dim=1).max().item())
    print("P row sums min/max:", result["P"].sum(dim=1).min().item(), result["P"].sum(dim=1).max().item())
    print("P col sums min/max:", result["P"].sum(dim=0).min().item(), result["P"].sum(dim=0).max().item())

    map_npz_path, map_meta_path, map_plot_path = make_map_paths(
        args.artifact_root,
        args.plot_root,
        args.experiment_name,
        model_folder,
        args.source_geom_file,
        args.target_geom_file,
        args.epsilon,
        args.max_iter,
        args.threshold,
    )

    save_coupling_results(
        map_npz_path,
        result["M"],
        result["pi0"],
        result["P"],
    )

    metadata = {
        "experiment_name": args.experiment_name,
        "source_geom_file": args.source_geom_file,
        "target_geom_file": args.target_geom_file,
        "source_ldd_file": src_ldd_path,
        "target_ldd_file": tgt_ldd_path,
        "epsilon": args.epsilon,
        "lr": args.lr,
        "threshold": args.threshold,
        "max_iter": args.max_iter,
        "is_s2": args.is_s2,
        "source_num_points": int(Z_a.shape[0]),
        "target_num_points": int(Z_b.shape[0]),
        "M_shape": list(result["M"].shape),
        "P_shape": list(result["P"].shape),
    }
    save_metadata(map_meta_path, metadata)

    plot_P_matrix(
        result["P"],
        map_plot_path,
        title=f"P coupling: {args.experiment_name}",
    )

    sorted_plot_path = map_plot_path.replace(".png", "_sorted_by_class.png")

    if "y" in source_geom and "y" in target_geom:
        plot_P_matrix_sorted(
            result["P"],
            source_geom,
            target_geom,
            sorted_plot_path,
            title=f"P coupling: {args.experiment_name}",
        )

    print(f"Loaded source geodesics: {args.source_geom_file}")
    print(f"Loaded target geodesics: {args.target_geom_file}")
    print(f"Saved source LDD to: {src_ldd_path}")
    print(f"Saved target LDD to: {tgt_ldd_path}")
    print(f"Saved source LDD spectrum plot to: {src_spectrum_path}")
    print(f"Saved target LDD spectrum plot to: {tgt_spectrum_path}")
    print(f"Saved coupling arrays to: {map_npz_path}")
    print(f"Saved coupling metadata to: {map_meta_path}")
    print(f"Saved coupling plot to: {map_plot_path}")
    if "y" in source_geom and "y" in target_geom:
        print(f"Saved sorted coupling plot to: {sorted_plot_path}")
    print(f"M shape: {tuple(result['M'].shape)}")
    print(f"pi0 shape: {tuple(result['pi0'].shape)}")
    print(f"P shape: {tuple(result['P'].shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source-geom-file", type=str, required=True)
    parser.add_argument("--target-geom-file", type=str, required=True)

    parser.add_argument("--experiment-name", type=str, required=True)

    parser.add_argument("--artifact-root", type=str, default="artifacts/mnist")
    parser.add_argument("--plot-root", type=str, default="plots/mnist")

    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=1e-9)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--is-s2", action="store_true")

    parser.add_argument("--n-centers", type=int, default=None)
    parser.add_argument("--r-bins", type=int, default=200)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--tau", type=float, default=0.015)

    args = parser.parse_args()
    main(args)


# from A to B: A is source, B is target
# python map.py \
#   --source-geom-file artifacts/mnist_geodesics/split42_digits012/model_A_ld8_seed1/align_geodesics_n100_random_selseed0_quadratic_lr005_S20_steps200.npz \
#   --target-geom-file artifacts/mnist_geodesics/split42_digits012/model_B_ld8_seed12/align_geodesics_n100_random_selseed0_quadratic_lr005_S20_steps200.npz \
#   --experiment-name split42_digits012_A1_to_B12 \
#   --artifact-root artifacts/mnist \
#   --plot-root plots/mnist \
#   --epsilon 0.008 \
#   --threshold 1e-9 \
#   --max-iter 200 \
#   --r-bins 100 \
#   --tau 0.015


# Sphere2
# python map.py \
#   --source-geom-file artifacts/s2_geodesics/vmf_mixture/A_geodesics_n2000_seed1_rotseed123.npz \
#   --target-geom-file artifacts/s2_geodesics/vmf_mixture/B_geodesics_n2000_seed1_rotseed123.npz \
#   --experiment-name vmf_mixture_A_to_B_rotseed123 \
#   --artifact-root artifacts/s2_map \
#   --plot-root plots/s2 \
#   --epsilon 0.008 \
#   --threshold 1e-9 \
#   --max-iter 200 \
#   --r-bins 100 \
#   --tau 0.015 \
#   --is-s2