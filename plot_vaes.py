import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from vae import VAE


def ensure_dir(path: str):
    if path != "" and not os.path.exists(path):
        os.makedirs(path)


def load_split_subset(data_root: str, split_file: str, split_name: str):
    """
    Load MNIST train set and select the requested split using raw indices
    stored in the split file.
    """
    split_payload = torch.load(split_file, map_location="cpu")
    split_raw = split_payload["split_indices_raw"]

    if split_name not in split_raw:
        raise ValueError(f"Unknown split_name '{split_name}'. Available: {list(split_raw.keys())}")

    indices = split_raw[split_name].long().tolist()

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=False,
        transform=transform,
    )

    subset = Subset(dataset, indices)
    return subset


def load_model(checkpoint_root, experiment_name, model_name, model_seed, latent_dim, device, model_file="best_model.pt"):
    """
    Load trained VAE from the expected checkpoint folder.
    """
    full_model_name = f"{model_name}_ld{latent_dim}_seed{model_seed}"
    model_dir = os.path.join(checkpoint_root, experiment_name, full_model_name)
    model_path = os.path.join(model_dir, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model file: {model_path}")

    model = VAE(latent_dim=latent_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, full_model_name, model_dir


@torch.no_grad()
def get_latent_mu(model, loader, device):
    """
    Deterministic latent extraction using encoder mean.
    """
    model.eval()
    z_list = []
    y_list = []

    for x, y in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        z_list.append(mu.cpu())
        y_list.append(y.cpu())

    z = torch.cat(z_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return z, y


def project_to_2d(zA_full, zB_full, latent_dim):
    """
    If latent_dim == 2, use the raw coordinates.
    Otherwise use shared PCA fitted on concatenated latents.
    """
    if latent_dim == 2:
        zA = zA_full.numpy()
        zB = zB_full.numpy()
        return zA, zB, "latent dim 1", "latent dim 2"

    Z = torch.cat([zA_full, zB_full], dim=0).numpy()
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)

    zA = Z_2d[:len(zA_full)]
    zB = Z_2d[len(zA_full):]
    return zA, zB, "PC 1", "PC 2"


def make_shared_legend(scatter_artist, labels):
    cmap = scatter_artist.cmap
    norm = scatter_artist.norm
    handles = [
        mpatches.Patch(color=cmap(norm(lbl)), label=str(int(lbl)))
        for lbl in labels
    ]
    return handles


def plot_vae_latent_alignment(
    checkpoint_root,
    plot_root,
    data_root,
    split_file,
    experiment_name,
    latent_dim,
    model_A_name,
    model_A_seed,
    model_B_name,
    model_B_seed,
    split_name="test",
    batch_size=256,
    device="cpu",
    model_file="best_model.pt",
    save_name=None,
):
    device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")

    subset = load_split_subset(
        data_root=data_root,
        split_file=split_file,
        split_name=split_name,
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    vaeA, full_A_name, model_A_dir = load_model(
        checkpoint_root=checkpoint_root,
        experiment_name=experiment_name,
        model_name=model_A_name,
        model_seed=model_A_seed,
        latent_dim=latent_dim,
        device=device,
        model_file=model_file,
    )

    vaeB, full_B_name, model_B_dir = load_model(
        checkpoint_root=checkpoint_root,
        experiment_name=experiment_name,
        model_name=model_B_name,
        model_seed=model_B_seed,
        latent_dim=latent_dim,
        device=device,
        model_file=model_file,
    )

    zA_full, yA = get_latent_mu(vaeA, loader, device)
    zB_full, yB = get_latent_mu(vaeB, loader, device)

    zA, zB, xlabel, ylabel = project_to_2d(zA_full, zB_full, latent_dim)

    yA = yA.numpy()
    yB = yB.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    sc1 = axes[0].scatter(
        zA[:, 0],
        zA[:, 1],
        c=yA,
        cmap="tab10",
        s=5,
        alpha=0.7,
    )
    axes[0].set_title(f"{full_A_name} ({split_name} set)")

    axes[1].scatter(
        zB[:, 0],
        zB[:, 1],
        c=yB,
        cmap="tab10",
        s=5,
        alpha=0.7,
    )
    axes[1].set_title(f"{full_B_name} ({split_name})")

    x_min = min(zA[:, 0].min(), zB[:, 0].min())
    x_max = max(zA[:, 0].max(), zB[:, 0].max())
    y_min = min(zA[:, 1].min(), zB[:, 1].min())
    y_max = max(zA[:, 1].max(), zB[:, 1].max())

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(xlabel)

    axes[0].set_ylabel(ylabel)

    labels = np.unique(np.concatenate([yA, yB]))
    handles = make_shared_legend(sc1, labels)

    for ax in axes:
        ax.legend(handles=handles, title="Digit")

    plt.tight_layout()

    plot_dir = os.path.join(plot_root, experiment_name)
    ensure_dir(plot_dir)

    if save_name is None:
        save_name = f"{split_name}_latent_alignment_ld{latent_dim}_{full_A_name}_vs_{full_B_name}.png"

    save_path = os.path.join(plot_dir, save_name)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved plot to: {save_path}")
    print(f"Loaded model A from: {model_A_dir}")
    print(f"Loaded model B from: {model_B_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--split-file", type=str, required=True)

    parser.add_argument("--checkpoint-root", type=str, default="checkpoints/mnist")
    parser.add_argument("--plot-root", type=str, default="plots/mnist")
    parser.add_argument("--experiment-name", type=str, required=True)

    parser.add_argument("--latent-dim", type=int, required=True)

    parser.add_argument("--model-A-name", type=str, default="model_A")
    parser.add_argument("--model-A-seed", type=int, required=True)

    parser.add_argument("--model-B-name", type=str, default="model_B")
    parser.add_argument("--model-B-seed", type=int, required=True)

    parser.add_argument("--split-name", type=str, default="test", choices=["train", "val", "align", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model-file", type=str, default="best_model.pt")
    parser.add_argument("--save-name", type=str, default=None)

    args = parser.parse_args()

    plot_vae_latent_alignment(
        checkpoint_root=args.checkpoint_root,
        plot_root=args.plot_root,
        data_root=args.data_root,
        split_file=args.split_file,
        experiment_name=args.experiment_name,
        latent_dim=args.latent_dim,
        model_A_name=args.model_A_name,
        model_A_seed=args.model_A_seed,
        model_B_name=args.model_B_name,
        model_B_seed=args.model_B_seed,
        split_name=args.split_name,
        batch_size=args.batch_size,
        device=args.device,
        model_file=args.model_file,
        save_name=args.save_name,
    )


# python plot_vaes.py \
#   --device cuda \
#   --data-root data \
#   --split-file data/MNIST/splits/mnist_split_seed_42_digits_0_1_2.pt \
#   --checkpoint-root checkpoints/mnist \
#   --plot-root plots/mnist \
#   --experiment-name split42_digits012 \
#   --latent-dim 8 \
#   --model-A-name model_A \
#   --model-A-seed 0 \
#   --model-B-name model_B \
#   --model-B-seed 12 \
#   --split-name test