import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import ot


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
    }

    if "y" in data.files:
        payload["y"] = torch.from_numpy(data["y"]).long()

    if "center_idx" in data.files:
        payload["center_idx"] = torch.from_numpy(data["center_idx"]).long()

    return payload


def load_ldd_file(path):
    """
    Load saved LDD output.
    Must contain H and r.
    """
    data = load_npz_file(path)

    if "H" not in data.files:
        raise ValueError(f"Could not find H in: {path}")

    payload = {
        "H": torch.from_numpy(data["H"]).float(),
        "keys": list(data.files),
    }

    if "r" in data.files:
        payload["r"] = torch.from_numpy(data["r"]).float()

    if "center_idx" in data.files:
        payload["center_idx"] = torch.from_numpy(data["center_idx"]).long()

    return payload


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


def init_sinkhorn(M, epsilon):
    """
    Initialize pi^0 from the signature cost matrix M.
    """
    n_a, n_b = M.shape
    a, b = uniform_marginals(n_a, n_b)

    pi0 = ot.sinkhorn(
        a,
        b,
        M.detach().cpu().numpy(),
        reg=epsilon,
    )

    return torch.from_numpy(pi0).float()


def solve_gw_coupling(C_a, C_b, epsilon, threshold, max_iter, pi0=None):
    """
    Compute the entropic GW coupling P.
    """
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
    )

    return torch.from_numpy(P).float()


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
    input: Z_a, Z_b, C_a, C_b, H_a, H_b, epsilon, lr, threshold, max_iter
    output: for now only M, pi0, P

    algorithm 1: Neural GM alignment
    """

    # basic checks
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

    # Phase 1
    # -------------------
    # compute cost matrix M from the LDD signatures
    M = compute_M_distance(H_a, H_b)

    # initialize coupling pi^0 with Sinkhorn
    pi0 = init_sinkhorn(M, epsilon)

    # Phase 2
    # -------------------
    # compute the GW coupling P using pi^0 as warm start
    P = solve_gw_coupling(C_a, C_b, epsilon, threshold, max_iter, pi0=pi0)

    # Phase 3
    # -------------------
    # parameterize map T

    # Phase 4
    # -------------------
    # refine map T using P

    return {
        "M": M,
        "pi0": pi0,
        "P": P,
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


def plot_P_matrix(P, save_path, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    P_np = P.detach().cpu().numpy()

    plt.figure(figsize=(5, 4))
    plt.imshow(P_np, aspect="auto")
    plt.colorbar()
    plt.xlabel("target")
    plt.ylabel("source")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --------------------------------------------------
# main
# --------------------------------------------------

def main(args):
    # load saved geometry
    source_geom = load_geom_file(args.source_geom_file)
    target_geom = load_geom_file(args.target_geom_file)

    # load saved LDD
    source_ldd = load_ldd_file(args.source_ldd_file)
    target_ldd = load_ldd_file(args.target_ldd_file)

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

    artifact_dir = os.path.join(args.artifact_root, args.experiment_name)
    os.makedirs(artifact_dir, exist_ok=True)

    eps_tag = str(args.epsilon).replace(".", "")
    coupling_name = f"P_coupling_eps{eps_tag}"

    npz_path = os.path.join(artifact_dir, f"{coupling_name}.npz")
    json_path = os.path.join(artifact_dir, f"{coupling_name}_metadata.json")

    save_coupling_results(
        npz_path,
        result["M"],
        result["pi0"],
        result["P"],
    )

    metadata = {
        "experiment_name": args.experiment_name,
        "source_geom_file": args.source_geom_file,
        "target_geom_file": args.target_geom_file,
        "source_ldd_file": args.source_ldd_file,
        "target_ldd_file": args.target_ldd_file,
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
    save_metadata(json_path, metadata)

    plot_name = f"P_coupling_{args.experiment_name}_eps{eps_tag}.png"
    plot_path = os.path.join(args.plot_root, plot_name)
    plot_P_matrix(
        result["P"],
        plot_path,
        title=f"P coupling: {args.experiment_name}",
    )

    print(f"Saved coupling arrays to: {npz_path}")
    print(f"Saved metadata to: {json_path}")
    print(f"Saved plot to: {plot_path}")
    print(f"M shape: {tuple(result['M'].shape)}")
    print(f"pi0 shape: {tuple(result['pi0'].shape)}")
    print(f"P shape: {tuple(result['P'].shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source-geom-file", type=str, required=True)
    parser.add_argument("--target-geom-file", type=str, required=True)
    parser.add_argument("--source-ldd-file", type=str, required=True)
    parser.add_argument("--target-ldd-file", type=str, required=True)

    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--artifact-root", type=str, default="artifacts/maps")
    parser.add_argument("--plot-root", type=str, default="plots/maps")

    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=1e-9)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--is-s2", action="store_true")

    args = parser.parse_args()
    main(args)