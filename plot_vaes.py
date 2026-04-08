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


def load_filtered_mnist_test(data_root: str, digits):
    transform = transforms.ToTensor()
    test = datasets.MNIST(
        data_root,
        train=False,
        download=True,
        transform=transform,
    )

    digits = list(digits)
    mask = torch.zeros_like(test.targets, dtype=torch.bool)
    for d in digits:
        mask |= (test.targets == d)

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
# Main plotting function 
# --------------------------------------------------

def plot_vae_latent_alignment(
    checkpoint_prefix="checkpoints/mnist_split",
    data_root="data/",
    latent_dim=8,
    batch_size=256,
    device="cpu",
    save_path=None,
    digits=None,
):
    """
    Plots side-by-side latent spaces for VAE A and VAE B
    using whatever digits the models were trained on.
    """

    device = torch.device(device)

    # --------------------------------------------------
    # Infer digits if not provided
    # --------------------------------------------------
    if digits is None:
        splits_path = f"{checkpoint_prefix}_splits.pt"
        if os.path.exists(splits_path):
            split_info = torch.load(splits_path, map_location="cpu")
            digits = split_info["selected_digits"]
        else:
            raise ValueError(
                "digits=None, but no split file found. "
                "Pass digits explicitly or provide *_splits.pt."
            )

    digits = list(digits)

    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------
    test_subset = load_filtered_mnist_test(data_root, digits)
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------
    num_classes = len(digits)

    vaeA = SupervisedVAE(latent_dim=latent_dim, num_classes=num_classes).to(device)
    vaeA.load_state_dict(
        torch.load(f"{checkpoint_prefix}_vaeA.pt", map_location=device)
    )

    vaeB = SupervisedVAE(latent_dim=latent_dim, num_classes=num_classes).to(device)
    vaeB.load_state_dict(
        torch.load(f"{checkpoint_prefix}_vaeB.pt", map_location=device)
    )

    # --------------------------------------------------
    # Extract deterministic latents
    # --------------------------------------------------
    zA_full, yA = get_latent_mu(vaeA, test_loader, device)
    zB_full, yB = get_latent_mu(vaeB, test_loader, device)

    # --------------------------------------------------
    # Shared PCA
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

    sc1 = axes[0].scatter(
        zA[:, 0], zA[:, 1],
        c=yA,
        cmap="tab10",
        s=5,
        alpha=0.7
    )
    axes[0].set_title(f"VAE A (test digits {digits})")

    axes[1].scatter(
        zB[:, 0], zB[:, 1],
        c=yB,
        cmap="tab10",
        s=5,
        alpha=0.7
    )
    axes[1].set_title(f"VAE B (test digits {digits})")

    x_min = min(zA[:, 0].min(), zB[:, 0].min())
    x_max = max(zA[:, 0].max(), zB[:, 0].max())
    y_min = min(zA[:, 1].min(), zB[:, 1].min())
    y_max = max(zA[:, 1].max(), zB[:, 1].max())

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    labels = np.unique(np.concatenate([yA, yB]))
    cmap = sc1.cmap
    norm = sc1.norm

    handles = [
        mpatches.Patch(color=cmap(norm(lbl)), label=str(int(lbl)))
        for lbl in labels
    ]

    for ax in axes:
        ax.legend(handles=handles, title="Digit")

    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"Saved to: {save_path}")
    else:
        plt.show()