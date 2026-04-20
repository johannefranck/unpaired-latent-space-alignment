import os
import json
import argparse
import numpy as np
import torch
import datetime

from vae import VAE
from geodesics import compute_geodesics


def select_indices(num_available, num_points, mode="first", seed=0):
    if num_points is None or num_points >= num_available:
        return torch.arange(num_available, dtype=torch.long)

    if mode == "first":
        return torch.arange(num_points, dtype=torch.long)

    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(num_available, generator=g)
        return perm[:num_points]

    raise ValueError(f"Unknown selection mode: {mode}")


def main(args):
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    model_name = f"{args.model_name}_seed{args.model_seed}"

    checkpoint_model_dir = os.path.join(
        args.checkpoint_root,
        args.experiment_name,
        model_name,
    )

    artifact_model_dir = os.path.join(
        args.artifact_root,
        args.experiment_name,
        model_name,
    )

    geodesic_model_dir = os.path.join(
        args.geodesic_root,
        args.experiment_name,
        model_name,
    )

    os.makedirs(geodesic_model_dir, exist_ok=True)

    latent_path = os.path.join(
        artifact_model_dir,
        f"{args.split_name}_latents.pt",
    )
    model_path = os.path.join(
        checkpoint_model_dir,
        args.model_file,
    )
    config_path = os.path.join(
        checkpoint_model_dir,
        "config.pt",
    )

    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Could not find latent file: {latent_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model file: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    latent_payload = torch.load(latent_path, map_location="cpu")
    model_config = torch.load(config_path, map_location="cpu")

    latent_dim = model_config["latent_dim"]

    model = VAE(latent_dim=latent_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    Z_all = latent_payload["z_mu"].float()
    X_all = latent_payload["x"].float()
    y_all = latent_payload["y"].long()
    dataset_indices_raw_all = latent_payload["dataset_indices_raw"].long()
    canonical_order_all = latent_payload["canonical_order"].long()

    n_available = Z_all.shape[0]
    chosen_idx = select_indices(
        num_available=n_available,
        num_points=args.num_points,
        mode=args.selection_mode,
        seed=args.selection_seed,
    )

    Z = Z_all[chosen_idx].to(device)
    X = X_all[chosen_idx]
    y = y_all[chosen_idx]
    dataset_indices_raw = dataset_indices_raw_all[chosen_idx]
    canonical_order = canonical_order_all[chosen_idx]

    cfg = {
        "curve_type": args.curve_type,
        "lr": args.lr,
        "steps": args.steps,
        "num_segments": args.num_segments,
        "ensemble": False,
        "print_every": args.print_every,
    }

    print(f"Computing geodesics for {Z.shape[0]} points...")
    D = compute_geodesics(Z, model, cfg)

    if args.num_points is None:
        num_points_tag = "all"
    else:
        num_points_tag = f"n{Z.shape[0]}"

    filename = (
        f"{args.split_name}_geodesics_"
        f"{num_points_tag}_"
        f"{args.curve_type}_"
        f"S{args.num_segments}_"
        f"steps{args.steps}.npz"
    )

    out_path = os.path.join(geodesic_model_dir, filename)

    # save numeric arrays in compressed npz
    np.savez_compressed(
        out_path,
        D_geodesic=D.cpu().numpy().astype(np.float32),
        x=X.cpu().numpy().astype(np.float32),
        y=y.cpu().numpy().astype(np.int64),
        z_mu=Z.cpu().numpy().astype(np.float32),
        chosen_idx_within_latent_file=chosen_idx.cpu().numpy().astype(np.int64),
        dataset_indices_raw=dataset_indices_raw.cpu().numpy().astype(np.int64),
        canonical_order=canonical_order.cpu().numpy().astype(np.int64),
    )

    # save non-array metadata separately
    metadata = {
        "experiment_name": args.experiment_name,
        "created_at": datetime.now().isoformat(),
        "model_name": args.model_name,
        "model_seed": args.model_seed,
        "model_file": args.model_file,
        "split_name": args.split_name,
        "latent_path": latent_path,
        "model_path": model_path,
        "config_path": config_path,
        "selection_mode": args.selection_mode,
        "selection_seed": args.selection_seed,
        "geodesic_cfg": cfg,
        "num_points": int(Z.shape[0]),
        "latent_dim": int(latent_dim),
        "distance_dtype": "float32",
        "npz_file": out_path,
    }

    metadata_path = out_path.replace(".npz", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved geodesics to: {out_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Distance matrix shape: {tuple(D.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--split-name", type=str, required=True, choices=["train", "val", "align", "test"])

    parser.add_argument("--checkpoint-root", type=str, default="checkpoints/mnist")
    parser.add_argument("--artifact-root", type=str, default="artifacts/mnist")
    parser.add_argument("--geodesic-root", type=str, default="artifacts/mnist_geodesics")

    parser.add_argument("--model-file", type=str, default="best_model.pt")

    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--selection-mode", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--selection-seed", type=int, default=0)

    parser.add_argument("--curve-type", type=str, default="quadratic", choices=["quadratic", "cubic"])
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-segments", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=0)

    args = parser.parse_args()
    main(args)


# python build_mnist_geodesics.py \
#   --device cuda \
#   --experiment-name split42_digits012_latent2_test \
#   --model-name model_A \
#   --model-seed 0 \
#   --split-name align \
#   --num-points 50 \
#   --selection-mode first \
#   --curve-type quadratic \
#   --steps 300 \
#   --num-segments 20