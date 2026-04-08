import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from supervised_vae import SupervisedVAE


class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, digit_to_label):
        self.subset = subset
        self.digit_to_label = digit_to_label

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        y = self.digit_to_label[int(y)]
        return x, y


def train_vae(model, loader, device, epochs, lr):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = total_recon = total_kl = total_cls = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss, recon, kl, cls = model(x, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            total_cls += cls.item()
            n_batches += 1

        print(f"[Epoch {epoch:3d}] "
              f"Loss={total_loss/n_batches:.4f} "
              f"Recon={total_recon/n_batches:.4f} "
              f"KL={total_kl/n_batches:.4f} "
              f"CLS={total_cls/n_batches:.4f}")

    return model


def collect_latents(model, loader, device):
    model.eval()
    zs, ys = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            zs.append(mu.cpu())
            ys.append(torch.as_tensor(y).clone())

    return torch.cat(zs, 0), torch.cat(ys, 0)


def train_two_vaes(
    device="cpu",
    epochs=15,
    batch_size=128,
    latent_dim=8,
    lr=1e-3,
    data_root="data/",
    prefix="checkpoints/mnist_same",
    seedA=0,
    seedB=1,
    num_classes=3,
    digits=None,
):
    """
    Train two VAEs on EXACT same data but with different seeds.

    Key property:
        zA[i] and zB[i] correspond to the SAME image.
    """

    device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seedA)

    # ---- Load MNIST and restrict to selected digits ----
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        data_root, train=True, download=True, transform=transform
    )

    if digits is None:
        selected_digits = list(range(num_classes))
    else:
        selected_digits = list(digits)

    assert len(selected_digits) == num_classes
    assert len(set(selected_digits)) == len(selected_digits)
    assert all(0 <= d <= 9 for d in selected_digits)

    targets = full_train.targets
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for d in selected_digits:
        mask |= (targets == d)

    idx = mask.nonzero(as_tuple=False).view(-1)
    full_train = Subset(full_train, idx)

    digit_to_label = {digit: i for i, digit in enumerate(selected_digits)}
    full_train = RemappedSubset(full_train, digit_to_label)

    # SAME dataset for both
    loaderA = DataLoader(full_train, batch_size=batch_size, shuffle=True)
    loaderB = DataLoader(full_train, batch_size=batch_size, shuffle=True)

    # ---- Train VAE A ----
    torch.manual_seed(seedA)
    vaeA = SupervisedVAE(latent_dim=latent_dim, num_classes=num_classes)
    print("Training VAE_A...")
    vaeA = train_vae(vaeA, loaderA, device, epochs, lr)
    torch.save(vaeA.state_dict(), f"{prefix}_vaeA.pt")

    # ---- Train VAE B (different seed) ----
    torch.manual_seed(seedB)
    vaeB = SupervisedVAE(latent_dim=latent_dim, num_classes=num_classes)
    print("Training VAE_B...")
    vaeB = train_vae(vaeB, loaderB, device, epochs, lr)
    torch.save(vaeB.state_dict(), f"{prefix}_vaeB.pt")

    # ---- Collect latents (deterministic order) ----
    loader_eval = DataLoader(full_train, batch_size=batch_size, shuffle=False)

    zA, yA = collect_latents(vaeA.to(device), loader_eval, device)
    zB, yB = collect_latents(vaeB.to(device), loader_eval, device)

    torch.save({"z": zA, "y": yA, "selected_digits": selected_digits}, f"{prefix}_zA.pt")
    torch.save({"z": zB, "y": yB, "selected_digits": selected_digits}, f"{prefix}_zB.pt")

    print("Saved models and latent representations.")

    return vaeA, vaeB, zA, zB, yA


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-root", type=str, default="data/")
    ap.add_argument("--prefix", type=str, default="checkpoints/mnist_same")
    ap.add_argument("--seedA", type=int, default=0)
    ap.add_argument("--seedB", type=int, default=1)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--digits", type=int, nargs="+", default=None)

    args = ap.parse_args()

    train_two_vaes(**vars(args))