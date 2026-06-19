"""
Maps the 30 Kaggle dataset categories to 6 major waste classes and
splits images into train/val/test sets (80/10/10 stratified by class).

Dataset: Recyclable and Household Waste Classification
Source:  https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification

Expected raw layout (after Kaggle download + unzip):
    ml-training/data/raw/images/<Category Name>/default/*.jpg

Output layout:
    ml-training/data/splits/train/<class>/
    ml-training/data/splits/val/<class>/
    ml-training/data/splits/test/<class>/

Usage:
    python ml-training/src/data_preprocessing.py
    python ml-training/src/data_preprocessing.py --raw-dir path/to/raw --seed 42
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Category → major class mapping
# Keys are exact Kaggle folder names (case-insensitive match in code below).
# Categories absent from this mapping are skipped with a warning.
# ---------------------------------------------------------------------------
CATEGORY_TO_CLASS: dict[str, str] = {
    # Plastic (14 sub-categories)
    "disposable_plastic_cutlery": "plastic",
    "plastic_cup_lids": "plastic",
    "plastic_detergent_bottles": "plastic",
    "plastic_food_containers": "plastic",
    "plastic_shopping_bags": "plastic",
    "plastic_soda_bottles": "plastic",
    "plastic_straws": "plastic",
    "plastic_trash_bags": "plastic",
    "plastic_water_bottles": "plastic",
    "styrofoam_cups": "plastic",
    "styrofoam_food_containers": "plastic",
    # Paper (5 sub-categories)
    "magazines": "paper",
    "newspaper": "paper",
    "office_paper": "paper",
    "paper_cups": "paper",
    # Glass (3 sub-categories)
    "glass_beverage_bottles": "glass",
    "glass_cosmetic_containers": "glass",
    "glass_food_jars": "glass",
    # Metal (4 sub-categories)
    "aerosol_cans": "metal",
    "aluminum_food_cans": "metal",
    "aluminum_soda_cans": "metal",
    "steel_food_cans": "metal",
    # Cardboard (2 sub-categories)
    "cardboard_boxes": "cardboard",
    "cardboard_packaging": "cardboard",
    # Organic (4 sub-categories)
    "coffee_grounds": "organic",
    "eggshells": "organic",
    "food_waste": "organic",
    "tea_bags": "organic",
    # Deliberately excluded: clothing, shoes (out of scope), real_world (mixed meta-folder)
}

SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(raw_dir: Path) -> dict[str, list[Path]]:
    """
    Walk raw_dir and group image paths by major class.

    The Kaggle dataset has two known layouts:
        raw/images/<Category>/default/*.jpg   (nested)
        raw/<Category>/*.jpg                  (flat)
    Both are handled by recursing into every subdirectory.
    """
    # Build a lookup: lower-case category name → major class
    lookup = {k.lower(): v for k, v in CATEGORY_TO_CLASS.items()}

    class_images: dict[str, list[Path]] = defaultdict(list)
    skipped_dirs: set[str] = set()

    # Find all category-level directories: any directory whose name matches a
    # known category (case-insensitive).
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_dir():
            continue
        major_class = lookup.get(path.name.lower())
        if major_class is None:
            skipped_dirs.add(path.name)
            continue
        # Collect image files directly inside this directory (and subfolders
        # like the "default/" sub-layer).
        for img in path.rglob("*"):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                class_images[major_class].append(img)

    unknown = skipped_dirs - {"images", "raw", "default", "splits", "processed"}
    if unknown:
        print(f"[WARNING] Skipped unrecognized category directories: {sorted(unknown)}")

    return dict(class_images)


def split_and_copy(
    class_images: dict[str, list[Path]],
    splits_dir: Path,
    seed: int,
) -> None:
    """Stratified split per class, then copy files into splits_dir."""
    rng = random.Random(seed)
    stats: dict[str, dict[str, int]] = {}

    for major_class, images in sorted(class_images.items()):
        shuffled = images[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        # Remainder goes to test to avoid rounding loss
        buckets = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }
        stats[major_class] = {}

        for split_name, files in buckets.items():
            dest_dir = splits_dir / split_name / major_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                # Preserve original filename; add class prefix to avoid
                # collisions when multiple Kaggle sub-categories map to same class.
                dest = dest_dir / src.name
                if dest.exists():
                    # Disambiguate by including the immediate parent folder name.
                    dest = dest_dir / f"{src.parent.name}__{src.name}"
                shutil.copy2(src, dest)
            stats[major_class][split_name] = len(files)

    return stats


def print_summary(stats: dict[str, dict[str, int]]) -> None:
    header = f"{'Class':<12} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}"
    print("\n" + header)
    print("-" * len(header))
    grand = {"train": 0, "val": 0, "test": 0}
    for cls, counts in sorted(stats.items()):
        total = sum(counts.values())
        print(
            f"{cls:<12} {counts['train']:>6} {counts['val']:>6} {counts['test']:>6} {total:>7}"
        )
        for k in grand:
            grand[k] += counts[k]
    print("-" * len(header))
    gt = sum(grand.values())
    print(f"{'TOTAL':<12} {grand['train']:>6} {grand['val']:>6} {grand['test']:>6} {gt:>7}\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_raw = repo_root / "ml-training" / "data" / "raw"
    default_splits = repo_root / "ml-training" / "data" / "splits"

    parser = argparse.ArgumentParser(description="Preprocess Kaggle waste dataset.")
    parser.add_argument("--raw-dir", type=Path, default=default_raw)
    parser.add_argument("--splits-dir", type=Path, default=default_splits)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Raw data dir : {args.raw_dir}")
    print(f"Splits dir   : {args.splits_dir}")

    if not args.raw_dir.exists() or not any(args.raw_dir.iterdir()):
        print(
            "\n[ERROR] Raw data directory is empty or does not exist.\n"
            "Download the dataset from Kaggle first:\n"
            "  kaggle datasets download -d alistairking/recyclable-and-household-waste-classification\n"
            "Then unzip into ml-training/data/raw/\n"
        )
        raise SystemExit(1)

    print("\nScanning images …")
    class_images = collect_images(args.raw_dir)

    if not class_images:
        print("[ERROR] No images found. Check that the raw directory layout matches expectations.")
        raise SystemExit(1)

    found_classes = sorted(class_images.keys())
    total_images = sum(len(v) for v in class_images.values())
    print(f"Found {total_images} images across {len(found_classes)} classes: {found_classes}")

    print("\nSplitting and copying …")
    stats = split_and_copy(class_images, args.splits_dir, seed=args.seed)
    print_summary(stats)
    print(f"Done. Splits written to: {args.splits_dir}")


if __name__ == "__main__":
    main()
