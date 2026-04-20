import os
import argparse
import torch
from torchvision import datasets, transforms


def stratified_four_way_split(labels, train=0.7, val=0.1, align=0.1, test=0.1, seed=0):
    assert abs(train + val + align + test - 1.0) < 1e-8, "Fractions must sum to 1."

    labels = torch.as_tensor(labels, dtype=torch.long)
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

    split = {
        "train": torch.sort(torch.cat(idx_train))[0],
        "val": torch.sort(torch.cat(idx_val))[0],
        "align": torch.sort(torch.cat(idx_align))[0],
        "test": torch.sort(torch.cat(idx_test))[0],
    }
    return split


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = datasets.MNIST(
        root=args.data_root,
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )

    raw_targets = dataset.targets.clone()

    if args.digits is None:
        selected_digits = list(range(10))
    else:
        selected_digits = sorted(list(set(args.digits)))

    mask = torch.zeros_like(raw_targets, dtype=torch.bool)
    for d in selected_digits:
        mask |= (raw_targets == d)

    raw_indices = torch.where(mask)[0]
    filtered_targets_raw = raw_targets[raw_indices]

    digit_to_label = {digit: i for i, digit in enumerate(selected_digits)}
    filtered_labels_remapped = torch.tensor(
        [digit_to_label[int(y)] for y in filtered_targets_raw],
        dtype=torch.long
    )

    split_filtered = stratified_four_way_split(
        filtered_labels_remapped,
        train=args.train_frac,
        val=args.val_frac,
        align=args.align_frac,
        test=args.test_frac,
        seed=args.split_seed,
    )

    split_raw = {
        split_name: raw_indices[idx_filtered]
        for split_name, idx_filtered in split_filtered.items()
    }

    if args.digits is None:
        digits_tag = "all_digits"
    else:
        digits_tag = "digits_" + "_".join(map(str, selected_digits))

    out_path = os.path.join(
        args.out_dir,
        f"mnist_split_seed_{args.split_seed}_{digits_tag}.pt"
    )

    payload = {
        "dataset_name": "MNIST",
        "data_root": args.data_root,
        "split_seed": args.split_seed,
        "selected_digits": selected_digits,
        "digit_to_label": digit_to_label,
        "fractions": {
            "train": args.train_frac,
            "val": args.val_frac,
            "align": args.align_frac,
            "test": args.test_frac,
        },
        "raw_indices": raw_indices,
        "filtered_labels_remapped": filtered_labels_remapped,
        "filtered_labels_raw_digits": filtered_targets_raw,
        "split_indices_filtered": split_filtered,
        "split_indices_raw": split_raw,
    }

    torch.save(payload, out_path)

    print(f"Saved split to: {out_path}")
    print(f"Selected digits: {selected_digits}")
    for k, v in split_raw.items():
        print(f"{k:>5}: {len(v)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="data/MNIST/splits")
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--align-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)

    parser.add_argument("--digits", type=int, nargs="+", default=None)

    args = parser.parse_args()
    main(args)