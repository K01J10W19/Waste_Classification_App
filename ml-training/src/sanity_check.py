"""
Phase 2 sanity-check script.

1. Loads one training batch through the full tf.data pipeline
   (resize → ResNet50 preprocess → augment).
2. Reverses normalisation for display, saves a sample-batch grid to
   outputs/sanity_check_batch.png.
3. Reads the split counts from disk and saves a class-distribution bar
   chart to outputs/class_distribution.png.
4. Prints a dataset summary table to stdout.

Usage:
    python ml-training/src/sanity_check.py
    python ml-training/src/sanity_check.py --splits-dir path/to/splits --batch-size 16
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is on the path regardless of working directory
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset_pipeline import build_train_pipeline

REPO_ROOT   = Path(__file__).resolve().parents[2]
SPLITS_DIR  = REPO_ROOT / "ml-training" / "data" / "splits"
OUTPUTS_DIR = REPO_ROOT / "ml-training" / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _denormalize_resnet(images: np.ndarray) -> np.ndarray:
    """
    Reverse ResNet-50 preprocess_input for human-readable display.

    preprocess_input converts RGB → BGR then subtracts ImageNet channel means
    [103.939, 116.779, 123.68].  We invert that here.
    """
    imgs = images.copy().astype("float32")
    # Add back the channel means (BGR order)
    imgs[..., 0] += 103.939
    imgs[..., 1] += 116.779
    imgs[..., 2] += 123.680
    # Flip BGR → RGB
    imgs = imgs[..., ::-1]
    return np.clip(imgs, 0, 255).astype("uint8")


def visualize_batch(
    splits_dir: Path,
    outputs_dir: Path,
    batch_size: int = 32,
    n_show: int = 16,
    seed: int = 42,
) -> None:
    """Load one training batch, denormalise, save grid to outputs/."""
    print("Loading one training batch …")
    train_ds, class_names = build_train_pipeline(
        splits_dir, batch_size=batch_size, seed=seed
    )

    for images, labels in train_ds.take(1):
        images = images.numpy()
        labels = labels.numpy()

    n_show = min(n_show, len(images))
    images_rgb = _denormalize_resnet(images[:n_show])

    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for i, ax in enumerate(np.array(axes).flatten()):
        if i < n_show:
            ax.imshow(images_rgb[i])
            ax.set_title(class_names[labels[i]], fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"Training batch — after ResNet50 preprocess + augmentation\n"
        f"({n_show} of {batch_size} samples shown)",
        fontsize=11,
    )
    plt.tight_layout()

    out = outputs_dir / "sanity_check_batch.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def count_splits(splits_dir: Path, class_names: list[str]) -> dict:
    """Count images per class per split by walking the directory tree."""
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for cls in class_names:
            cls_dir = splits_dir / split / cls
            n = sum(1 for f in cls_dir.glob("*") if f.is_file()) if cls_dir.exists() else 0
            counts[split][cls] = n
    return counts


def print_stats_table(counts: dict, class_names: list[str]) -> None:
    header = f"{'Class':<12} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}"
    print("\n" + header)
    print("-" * len(header))
    grand = {"train": 0, "val": 0, "test": 0}
    for cls in class_names:
        row = {s: counts[s].get(cls, 0) for s in ("train", "val", "test")}
        total = sum(row.values())
        print(f"{cls:<12} {row['train']:>6} {row['val']:>6} {row['test']:>6} {total:>7}")
        for s in grand:
            grand[s] += row[s]
    print("-" * len(header))
    gt = sum(grand.values())
    print(f"{'TOTAL':<12} {grand['train']:>6} {grand['val']:>6} {grand['test']:>6} {gt:>7}\n")


def plot_class_distribution(
    counts: dict,
    class_names: list[str],
    outputs_dir: Path,
) -> None:
    splits = ("train", "val", "test")
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, split in enumerate(splits):
        vals = [counts[split].get(c, 0) for c in class_names]
        bars = ax.bar(x + i * width, vals, width, label=split)
        ax.bar_label(bars, padding=2, fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.set_ylabel("Image count")
    ax.set_title("Class distribution across train / val / test splits")
    ax.legend()
    plt.tight_layout()

    out = outputs_dir / "class_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 sanity check.")
    parser.add_argument("--splits-dir",  type=Path, default=SPLITS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--batch-size",  type=int,  default=32)
    parser.add_argument("--n-show",      type=int,  default=16,
                        help="Number of images to show in the batch grid (max batch_size)")
    parser.add_argument("--seed",        type=int,  default=42)
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    if not (args.splits_dir / "train").exists():
        print(
            "[ERROR] Splits directory not found. Run data_preprocessing.py first:\n"
            "  python ml-training/src/data_preprocessing.py"
        )
        raise SystemExit(1)

    # 1. Visualize batch
    visualize_batch(
        args.splits_dir, args.outputs_dir,
        batch_size=args.batch_size, n_show=args.n_show, seed=args.seed,
    )

    # Need class names from the dataset to count splits
    _, class_names = build_train_pipeline(
        args.splits_dir, batch_size=1, seed=args.seed
    )

    # 2. Count splits and print table
    counts = count_splits(args.splits_dir, class_names)
    print_stats_table(counts, class_names)

    # 3. Save class distribution chart
    plot_class_distribution(counts, class_names, args.outputs_dir)

    # 4. Save summary JSON (mirrors dataset_stats.json but from actual split dirs)
    summary = {
        "class_names": class_names,
        "counts_per_split": counts,
        "totals": {
            cls: sum(counts[s].get(cls, 0) for s in ("train", "val", "test"))
            for cls in class_names
        },
        "grand_total": sum(
            counts[s].get(cls, 0)
            for s in ("train", "val", "test")
            for cls in class_names
        ),
    }
    stats_path = args.outputs_dir / "split_summary.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {stats_path}")
    print("\nSanity check complete.")


if __name__ == "__main__":
    main()
