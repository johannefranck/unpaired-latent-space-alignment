import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from vae import VAE


@torch.no_grad()
def encode_dataset(model, loader, device):
    model.eval()

    xs = []
    ys = []
    mus = []
    logvars = []

    for x, y in loader:
        x = x.to(device)
        mu, logvar = model.encode(x)

        xs.append(x.cpu().float())
        ys.append(torch.as_tensor(y).cpu().long())
        mus.append(mu.cpu().float())
        logvars.append(logvar.cpu().float())

    return {
        "x": torch.cat(xs, dim=0),
        "y": torch.cat(ys, dim=0),
        "z_mu": torch.cat(mus, dim=0),
        "z_logvar": torch.cat(logvars, dim=0),
    }


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

    os.makedirs(artifact_model_dir, exist_ok=True)

    config_path = os.path.join(checkpoint_model_dir, "config.pt")
    model_path = os.path.join(checkpoint_model_dir, args.model_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model file: {model_path}")

    config = torch.load(config_path, map_location="cpu")
    latent_dim = config["latent_dim"]

    split_payload = torch.load(args.split_file, map_location="cpu")
    split_raw = split_payload["split_indices_raw"]

    if args.split_name not in split_raw:
        raise ValueError(
            f"Unknown split_name '{args.split_name}'. Available: {list(split_raw.keys())}"
        )

    split_indices_raw = split_raw[args.split_name].clone().long()

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root=args.data_root,
        train=True,
        download=False,
        transform=transform,
    )

    subset = Subset(dataset, split_indices_raw.tolist())

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,   # canonical order
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = VAE(latent_dim=latent_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    encoded = encode_dataset(model, loader, device)

    n = encoded["z_mu"].shape[0]
    canonical_order = torch.arange(n, dtype=torch.long)

    payload = {
        # experiment/model metadata
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "model_seed": args.model_seed,
        "model_file": args.model_file,
        "latent_dim": latent_dim,

        # path metadata
        "checkpoint_model_dir": checkpoint_model_dir,
        "model_path": model_path,
        "config_path": config_path,
        "split_file": args.split_file,
        "data_root": args.data_root,

        # split metadata
        "split_name": args.split_name,
        "dataset_name": split_payload.get("dataset_name", "MNIST"),
        "selected_digits": split_payload.get("selected_digits", None),
        "digit_to_label": split_payload.get("digit_to_label", None),

        # indexing metadata
        # raw indices into the original MNIST training set
        "dataset_indices_raw": split_indices_raw,
        # canonical order 
        "canonical_order": canonical_order,

        # encoded artifacts
        "x": encoded["x"],                 
        "y": encoded["y"],
        "z_mu": encoded["z_mu"],
        "z_logvar": encoded["z_logvar"],

        # convenience metadata
        "num_points": n,
        "x_shape": tuple(encoded["x"].shape),
        "z_shape": tuple(encoded["z_mu"].shape),
        "x_dtype": str(encoded["x"].dtype),
        "z_dtype": str(encoded["z_mu"].dtype),
    }

    out_path = os.path.join(
        artifact_model_dir,
        f"{args.split_name}_latents.pt",
    )
    torch.save(payload, out_path)

    print(f"Saved encoded split to: {out_path}")
    print(f"Split: {args.split_name}")
    print(f"Number of points: {n}")
    print(f"x shape: {tuple(encoded['x'].shape)}")
    print(f"latent shape: {tuple(encoded['z_mu'].shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--split-file", type=str, required=True)

    parser.add_argument("--experiment-name", type=str, required=True)

    parser.add_argument("--checkpoint-root", type=str, default="checkpoints/mnist")
    parser.add_argument("--artifact-root", type=str, default="artifacts/mnist")

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-seed", type=int, required=True)

    parser.add_argument(
        "--split-name",
        type=str,
        required=True,
        choices=["train", "val", "align", "test"],
    )

    parser.add_argument("--model-file", type=str, default="best_model.pt")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()
    main(args)



# python encode_split.py \
#   --device cuda \
#   --data-root data \
#   --split-file data/MNIST/splits/mnist_split_seed_42_digits_0_1_2.pt \
#   --experiment-name split42_digits012_latent2_test \
#   --model-name model_B \
#   --model-seed 1 \
#   --split-name align