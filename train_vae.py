import os
import json
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from vae import VAE


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    n_batches = 0

    for x, _ in loader:
        x = x.to(device)

        optimizer.zero_grad()
        loss, recon, kl = model(x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
    }


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    n_batches = 0

    for x, _ in loader:
        x = x.to(device)
        loss, recon, kl = model(x)

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
    }


def save_history_plot(history, out_path):
    epochs = np.arange(1, len(history["train"]) + 1)

    train_loss = [x["loss"] for x in history["train"]]
    val_loss = [x["loss"] for x in history["val"]]

    train_recon = [x["recon"] for x in history["train"]]
    val_recon = [x["recon"] for x in history["val"]]

    train_kl = [x["kl"] for x in history["train"]]
    val_kl = [x["kl"] for x in history["val"]]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.plot(epochs, train_recon, label="train recon", linestyle="--")
    plt.plot(epochs, val_recon, label="val recon", linestyle="--")
    plt.plot(epochs, train_kl, label="train kl", linestyle=":")
    plt.plot(epochs, val_kl, label="val kl", linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_json_history(history, out_path):
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)


def main(args):
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    set_seed(args.model_seed)

    model_name = f"{args.model_name}_seed{args.model_seed}"
    model_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    split_payload = torch.load(args.split_file)
    split_raw = split_payload["split_indices_raw"]

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root=args.data_root,
        train=True,
        download=False,
        transform=transform,
    )

    ds_train = Subset(dataset, split_raw["train"])
    ds_val = Subset(dataset, split_raw["val"])

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train": [], "val": []}

    best_val = float("inf")
    best_epoch = -1
    best_state = None

    config = {
        "model_name": args.model_name,
        "model_seed": args.model_seed,
        "latent_dim": args.latent_dim,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "split_file": args.split_file,
        "data_root": args.data_root,
        "num_workers": args.num_workers,
        "save_every": args.save_every,
    }
    torch.save(config, os.path.join(model_dir, "config.pt"))

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = eval_one_epoch(model, val_loader, device)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"[Epoch {epoch:04d}] "
            f"train loss={train_metrics['loss']:.4f} recon={train_metrics['recon']:.4f} kl={train_metrics['kl']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} recon={val_metrics['recon']:.4f} kl={val_metrics['kl']:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(model_dir, "last_model.pt"))
        torch.save(history, os.path.join(model_dir, "history.pt"))
        save_json_history(history, os.path.join(model_dir, "history.json"))
        save_history_plot(history, os.path.join(model_dir, "training_curve.png"))

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(model_dir, "best_model.pt"))

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_val": best_val,
                "best_epoch": best_epoch,
            }
            torch.save(ckpt, os.path.join(model_dir, f"checkpoint_epoch_{epoch:04d}.pt"))

    summary = {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "n_train": len(ds_train),
        "n_val": len(ds_val),
    }
    torch.save(summary, os.path.join(model_dir, "summary.pt"))
    print(f"Saved model artifacts to: {model_dir}")
    print(f"Best val loss: {best_val:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--split-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="checkpoints/mnist")
    parser.add_argument("--model-name", type=str, default="vaeA")
    parser.add_argument("--model-seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50)

    args = parser.parse_args()
    main(args)