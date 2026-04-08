import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from supervised_vae import SupervisedVAE


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_filtered_mnist_test(data_root: str):
    transform = transforms.ToTensor()
    test = datasets.MNIST(
        data_root,
        train=False,
        download=True,
        transform=transform,
    )

    mask = (test.targets == 1) | (test.targets == 2) | (test.targets == 3)
    idx = mask.nonzero(as_tuple=False).view(-1)
    return Subset(test, idx)


def get_latent_mu(model, loader, device):
    """
    Deterministic latent extraction (NO sampling).
    """
    model.eval()
    z_list = []
    y_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, _ = model.encoder(x)
            z_list.append(mu.cpu())
            y_list.append(y.cpu())

    z = torch.cat(z_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return z, y


# --------------------------------------------------
# Main plotting function (call from notebook)
# --------------------------------------------------

def plot_vae_latent_alignment(
    checkpoint_prefix="checkpoints/mnist_split",
    data_root="data/",
    latent_dim=8,
    batch_size=256,
    device="cpu",
    save_path=None,
    n_lines=200,
):
    """
    Plots:
    1) VAE A latent space
    2) VAE B latent space
    3) Overlay + correspondence lines

    Designed for debugging alignment / symmetry issues.
    """

    device = torch.device(device)

    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------
    test_subset = load_filtered_mnist_test(data_root)
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------
    vaeA = SupervisedVAE(latent_dim=latent_dim).to(device)
    vaeA.load_state_dict(
        torch.load(f"{checkpoint_prefix}_vaeA.pt", map_location=device)
    )

    vaeB = SupervisedVAE(latent_dim=latent_dim).to(device)
    vaeB.load_state_dict(
        torch.load(f"{checkpoint_prefix}_vaeB.pt", map_location=device)
    )

    # --------------------------------------------------
    # Extract deterministic latents
    # --------------------------------------------------
    zA_full, yA = get_latent_mu(vaeA, test_loader, device)
    zB_full, yB = get_latent_mu(vaeB, test_loader, device)

    # --------------------------------------------------
    # Shared PCA (CRITICAL)
    # --------------------------------------------------
    Z = torch.cat([zA_full, zB_full], dim=0).numpy()

    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)

    zA = Z_2d[:len(zA_full)]
    zB = Z_2d[len(zA_full):]

    yA = yA.numpy()
    yB = yB.numpy()

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # --- VAE A ---
    sc1 = axes[0].scatter(
        zA[:, 0], zA[:, 1],
        c=yA,
        cmap="tab10",
        s=5,
        alpha=0.7
    )
    axes[0].set_title("VAE A")

    # --- VAE B ---
    axes[1].scatter(
        zB[:, 0], zB[:, 1],
        c=yB,
        cmap="tab10",
        s=5,
        alpha=0.7
    )
    axes[1].set_title("VAE B")


    # --------------------------------------------------
    # Shared limits
    # --------------------------------------------------
    x_min = min(zA[:, 0].min(), zB[:, 0].min())
    x_max = max(zA[:, 0].max(), zB[:, 0].max())
    y_min = min(zA[:, 1].min(), zB[:, 1].min())
    y_max = max(zA[:, 1].max(), zB[:, 1].max())

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------
    labels = np.unique(np.concatenate([yA, yB]))
    cmap = sc1.cmap
    norm = sc1.norm

    handles = []
    for lbl in labels:
        handles.append(
            mpatches.Patch(color=cmap(norm(lbl)), label=str(int(lbl)))
        )

    for ax in axes:
        ax.legend(handles=handles, title="Digit")

    plt.tight_layout()

    # --------------------------------------------------
    # Save or show
    # --------------------------------------------------
    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"Saved to: {save_path}")
    else:
        plt.show()