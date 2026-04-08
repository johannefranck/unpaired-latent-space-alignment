import os
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


def stratified_four_way_split(labels, train=0.7, val=0.1, align=0.1, test=0.1, seed=0):
    """
    labels: 1D torch tensor of remapped class labels
    returns dict of index tensors into the FILTERED dataset
    """
    assert abs(train + val + align + test - 1.0) < 1e-8, "Split fractions must sum to 1."

    labels = torch.as_tensor(labels).clone()
    g = torch.Generator().manual_seed(seed)

    idx_train, idx_val, idx_align, idx_test = [], [], [], []

    for k in torch.unique(labels):
        idx = torch.where(labels == k)[0]
        perm = idx[torch.randperm(len(idx), generator=g)]

        n = len(perm)
        n_train = int(train * n)
        n_val = int(val * n)
        n_align = int(align * n)

        idx_train.append(perm[:n_train])
        idx_val.append(perm[n_train:n_train + n_val])
        idx_align.append(perm[n_train + n_val:n_train + n_val + n_align])
        idx_test.append(perm[n_train + n_val + n_align:])

    return {
        "train": torch.sort(torch.cat(idx_train))[0],
        "val": torch.sort(torch.cat(idx_val))[0],
        "align": torch.sort(torch.cat(idx_align))[0],
        "test": torch.sort(torch.cat(idx_test))[0],
    }



def train_vae(model, train_loader, device, epochs, lr, val_loader=None):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = total_recon = total_kl = total_cls = 0.0
        n_batches = 0

        for x, y in train_loader:
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

        msg = (
            f"[Epoch {epoch:3d}] "
            f"Train Loss={total_loss/n_batches:.4f} "
            f"Recon={total_recon/n_batches:.4f} "
            f"KL={total_kl/n_batches:.4f} "
            f"CLS={total_cls/n_batches:.4f}"
        )

        if val_loader is not None:
            model.eval()
            val_loss = val_recon = val_kl = val_cls = 0.0
            n_val = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    loss, recon, kl, cls = model(x, y)
                    val_loss += loss.item()
                    val_recon += recon.item()
                    val_kl += kl.item()
                    val_cls += cls.item()
                    n_val += 1
            msg += (
                f" | Val Loss={val_loss/n_val:.4f} "
                f"Recon={val_recon/n_val:.4f} "
                f"KL={val_kl/n_val:.4f} "
                f"CLS={val_cls/n_val:.4f}"
            )
            model.train()

        print(msg)

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


def collect_split_artifacts(model, loader, device):
    model.eval()
    xs, ys, zs, logvars = [], [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            xs.append(x.cpu())
            ys.append(torch.as_tensor(y).clone())
            zs.append(mu.cpu())
            logvars.append(logvar.cpu())

    return {
        "x": torch.cat(xs, 0),
        "y": torch.cat(ys, 0),
        "z": torch.cat(zs, 0),
        "logvar": torch.cat(logvars, 0),
    }



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
    split_seed=42,
    num_classes=3,
    digits=None,
    train_frac=0.7,
    val_frac=0.1,
    align_frac=0.1,
    test_frac=0.1,
):
    """
    Train two VAEs on EXACT same data but with different seeds.

    Key property:
        zA[i] and zB[i] correspond to the SAME image.
    """

    device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    save_dir = os.path.dirname(prefix)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

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
    filtered_labels = torch.tensor(
        [full_train[i][1] for i in range(len(full_train))],
        dtype=torch.long
    )

    split_idx = stratified_four_way_split(
        filtered_labels,
        train=train_frac,
        val=val_frac,
        align=align_frac,
        test=test_frac,
        seed=split_seed,
    )

    ds_train = Subset(full_train, split_idx["train"])
    ds_val = Subset(full_train, split_idx["val"])
    ds_align = Subset(full_train, split_idx["align"])
    ds_test = Subset(full_train, split_idx["test"])

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    # no shuffle here: preserves correspondence between A and B
    loader_train_eval = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    loader_val_eval = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    loader_align_eval = DataLoader(ds_align, batch_size=batch_size, shuffle=False)
    loader_test_eval = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # ---- Train VAE A ----
    torch.manual_seed(seedA)
    vaeA = SupervisedVAE(latent_dim=latent_dim, num_classes=num_classes)
    print("Training VAE_A...")
    vaeA = train_vae(vaeA, loader_train, device, epochs, lr, val_loader=loader_val)
    torch.save(vaeA.state_dict(), f"{prefix}_vaeA.pt")

    # ---- Train VAE B (different seed) ----
    torch.manual_seed(seedB)
    vaeB = SupervisedVAE(latent_dim=latent_dim, num_classes=num_classes)
    print("Training VAE_B...")
    vaeB = train_vae(vaeB, loader_train, device, epochs, lr, val_loader=loader_val)
    torch.save(vaeB.state_dict(), f"{prefix}_vaeB.pt")

    # ---- Save split metadata ----
    torch.save(
        {
            "selected_digits": selected_digits,
            "digit_to_label": digit_to_label,
            "split_seed": split_seed,
            "split_indices_filtered": split_idx,
            "fractions": {
                "train": train_frac,
                "val": val_frac,
                "align": align_frac,
                "test": test_frac,
            },
        },
        f"{prefix}_splits.pt"
    )

    # ---- Collect splitwise artifacts (deterministic order) ----
    vaeA = vaeA.to(device)
    vaeB = vaeB.to(device)

    artifacts_A = {
        "train": collect_split_artifacts(vaeA, loader_train_eval, device),
        "val": collect_split_artifacts(vaeA, loader_val_eval, device),
        "align": collect_split_artifacts(vaeA, loader_align_eval, device),
        "test": collect_split_artifacts(vaeA, loader_test_eval, device),
    }

    artifacts_B = {
        "train": collect_split_artifacts(vaeB, loader_train_eval, device),
        "val": collect_split_artifacts(vaeB, loader_val_eval, device),
        "align": collect_split_artifacts(vaeB, loader_align_eval, device),
        "test": collect_split_artifacts(vaeB, loader_test_eval, device),
    }

    for split_name in ["train", "val", "align", "test"]:
        payload_A = dict(artifacts_A[split_name])
        payload_B = dict(artifacts_B[split_name])

        payload_A["filtered_indices"] = split_idx[split_name].clone()
        payload_B["filtered_indices"] = split_idx[split_name].clone()

        torch.save(payload_A, f"{prefix}_A_{split_name}.pt")
        torch.save(payload_B, f"{prefix}_B_{split_name}.pt")

    return {
        "vaeA": vaeA,
        "vaeB": vaeB,
        "splits": split_idx,
        "A_align": artifacts_A["align"],
        "B_align": artifacts_B["align"],
        "A_test": artifacts_A["test"],
        "B_test": artifacts_B["test"],
    }


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
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--digits", type=int, nargs="+", default=None)

    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--align-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)

    args = ap.parse_args()

    train_two_vaes(**vars(args))