"""
Phase 2 — Data preprocessing pipeline for the Kaggle waste dataset.

Maps 30 Kaggle categories → 7 major waste classes, performs data cleaning,
and builds stratified 70/15/15 train/val/test splits.

Dataset layout (after Kaggle download):
    data/raw/images/images/<category>/<default|real_world>/<Image_N.png>

    - <category>   : one of 30 folder names (snake_case)
    - default      : studio / controlled-background images (250 per category)
    - real_world   : in-situ / environmental images (250 per category)

Output:
    data/splits/<train|val|test>/<class>/<variant>__<filename>

    Both variants are present in every split so models learn from
    both controlled and real-world appearances.

Stats saved to:
    outputs/dataset_stats.json

Usage:
    python ml-training/src/data_preprocessing.py
    python ml-training/src/data_preprocessing.py --raw-dir path/to/raw --seed 42 --no-clean
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# 7-class mapping — all 30 Kaggle categories included, nothing excluded
# ---------------------------------------------------------------------------
CATEGORY_TO_CLASS: dict[str, str] = {
    # Plastic (11 sub-categories)
    "disposable_plastic_cutlery": "plastic",
    "plastic_cup_lids":           "plastic",
    "plastic_detergent_bottles":  "plastic",
    "plastic_food_containers":    "plastic",
    "plastic_shopping_bags":      "plastic",
    "plastic_soda_bottles":       "plastic",
    "plastic_straws":             "plastic",
    "plastic_trash_bags":         "plastic",
    "plastic_water_bottles":      "plastic",
    "styrofoam_cups":             "plastic",
    "styrofoam_food_containers":  "plastic",
    # Paper (4 sub-categories)
    "magazines":    "paper",
    "newspaper":    "paper",
    "office_paper": "paper",
    "paper_cups":   "paper",
    # Glass (3 sub-categories)
    "glass_beverage_bottles":   "glass",
    "glass_cosmetic_containers":"glass",
    "glass_food_jars":          "glass",
    # Metal (4 sub-categories)
    "aerosol_cans":       "metal",
    "aluminum_food_cans": "metal",
    "aluminum_soda_cans": "metal",
    "steel_food_cans":    "metal",
    # Cardboard (2 sub-categories)
    "cardboard_boxes":      "cardboard",
    "cardboard_packaging":  "cardboard",
    # Organic (4 sub-categories)
    "coffee_grounds": "organic",
    "eggshells":      "organic",
    "food_waste":     "organic",
    "tea_bags":       "organic",
    # Textiles (2 sub-categories)
    "clothing": "textiles",
    "shoes":    "textiles",
}

SPLITS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGE_VARIANTS: set[str] = {"default", "real_world"}


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def _is_valid_image(path: Path) -> bool:
    """Return True if PIL can fully decode the image (catches truncated files)."""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_images(
    raw_dir: Path,
    clean: bool = True,
) -> tuple[dict[str, dict[str, list[Path]]], int, list[Path]]:
    """
    Walk raw_dir, group image paths by (major_class, variant).

    Returns:
        class_images : {major_class: {"default": [...], "real_world": [...]}}
        n_corrupt    : number of files that failed PIL decode and were skipped
        corrupt_list : list of corrupt file paths
    """
    if not PIL_AVAILABLE and clean:
        print("[WARNING] Pillow not installed — skipping data cleaning (--no-clean implied).")
        clean = False

    lookup = {k.lower(): v for k, v in CATEGORY_TO_CLASS.items()}
    class_images: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    corrupt_list: list[Path] = []

    for cat_dir in sorted(raw_dir.rglob("*")):
        if not cat_dir.is_dir():
            continue
        major_class = lookup.get(cat_dir.name.lower())
        if major_class is None:
            continue

        # Collect from each image-variant subfolder (default / real_world)
        variant_dirs = sorted(
            d for d in cat_dir.iterdir()
            if d.is_dir() and d.name in IMAGE_VARIANTS
        )

        if not variant_dirs:
            # Flat layout: images sit directly in the category folder
            variant_dirs_iter = [(cat_dir, "default")]
        else:
            variant_dirs_iter = [(v, v.name) for v in variant_dirs]

        for variant_path, variant_name in variant_dirs_iter:
            for img in sorted(variant_path.iterdir()):
                if not (img.is_file() and img.suffix.lower() in IMAGE_EXTS):
                    continue
                if clean and not _is_valid_image(img):
                    corrupt_list.append(img)
                    continue
                class_images[major_class][variant_name].append(img)

    n_corrupt = len(corrupt_list)
    return dict(class_images), n_corrupt, corrupt_list


# ---------------------------------------------------------------------------
# Split & copy
# ---------------------------------------------------------------------------

def split_and_copy(
    class_images: dict[str, dict[str, list[Path]]],
    splits_dir: Path,
    seed: int,
) -> dict:
    """
    Stratified 70/15/15 split per (major_class × variant), then copy into splits_dir.

    Splitting within each variant independently ensures train/val/test each
    receive a proportional mix of studio (default) and real-world images.
    """
    rng = random.Random(seed)
    stats: dict[str, dict] = {}

    for major_class, variants in sorted(class_images.items()):
        stats[major_class] = {s: 0 for s in SPLITS}
        stats[major_class]["total"] = 0
        stats[major_class]["by_variant"] = {}

        for variant_name, images in sorted(variants.items()):
            shuffled = images[:]
            rng.shuffle(shuffled)

            n = len(shuffled)
            n_train = int(n * SPLITS["train"])
            n_val = int(n * SPLITS["val"])
            # remainder → test to avoid rounding loss

            buckets: dict[str, list[Path]] = {
                "train": shuffled[:n_train],
                "val":   shuffled[n_train : n_train + n_val],
                "test":  shuffled[n_train + n_val :],
            }

            stats[major_class]["by_variant"][variant_name] = {
                s: len(b) for s, b in buckets.items()
            }

            for split_name, files in buckets.items():
                dest_dir = splits_dir / split_name / major_class
                dest_dir.mkdir(parents=True, exist_ok=True)
                for src in files:
                    # Multiple Kaggle sub-categories (e.g. 11 → plastic) all share
                    # the same base filenames (Image_1.png … Image_N.png).
                    # Include the Kaggle category folder name to guarantee uniqueness:
                    #   <variant>__<kaggle_category>__<original_filename>
                    kaggle_cat = src.parent.parent.name  # e.g. "plastic_water_bottles"
                    dest = dest_dir / f"{variant_name}__{kaggle_cat}__{src.name}"
                    shutil.copy2(src, dest)
                stats[major_class][split_name] += len(files)

        stats[major_class]["total"] = sum(stats[major_class][s] for s in SPLITS)

    return stats


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_summary(stats: dict, n_corrupt: int) -> None:
    classes = sorted(stats.keys())
    header = f"{'Class':<12} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}"
    print("\n" + header)
    print("-" * len(header))
    grand = {s: 0 for s in SPLITS}
    for cls in classes:
        row = stats[cls]
        print(
            f"{cls:<12} {row['train']:>6} {row['val']:>6} {row['test']:>6} {row['total']:>7}"
        )
        for s in SPLITS:
            grand[s] += row[s]
    print("-" * len(header))
    gt = sum(grand.values())
    print(f"{'TOTAL':<12} {grand['train']:>6} {grand['val']:>6} {grand['test']:>6} {gt:>7}")
    if n_corrupt:
        print(f"\n[INFO] Corrupt files removed during cleaning: {n_corrupt}")
    print()


def save_stats(stats: dict, n_corrupt: int, corrupt_list: list[Path], outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_ratios": SPLITS,
        "n_corrupt_removed": n_corrupt,
        "corrupt_files": [str(p) for p in corrupt_list],
        "classes": sorted(stats.keys()),
        "per_class": stats,
    }
    out = outputs_dir / "dataset_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Stats saved to: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_raw = repo_root / "ml-training" / "data" / "raw"
    default_splits = repo_root / "ml-training" / "data" / "splits"
    default_outputs = repo_root / "ml-training" / "outputs"

    parser = argparse.ArgumentParser(description="Preprocess Kaggle waste dataset.")
    parser.add_argument("--raw-dir",    type=Path, default=default_raw)
    parser.add_argument("--splits-dir", type=Path, default=default_splits)
    parser.add_argument("--outputs-dir",type=Path, default=default_outputs)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--no-clean",   action="store_true",
                        help="Skip PIL data-cleaning pass (faster, unsafe)")
    args = parser.parse_args()

    print(f"Raw data dir  : {args.raw_dir}")
    print(f"Splits dir    : {args.splits_dir}")
    print(f"Data cleaning : {'OFF' if args.no_clean else 'ON (PIL decode check)'}")

    if not args.raw_dir.exists() or not any(args.raw_dir.iterdir()):
        print(
            "\n[ERROR] Raw data directory is empty or does not exist.\n"
            "Download the dataset first:\n"
            "  kaggle datasets download "
            "-d alistairking/recyclable-and-household-waste-classification "
            "-p ml-training/data/raw --unzip\n"
        )
        raise SystemExit(1)

    print("\nScanning and cleaning images …")
    class_images, n_corrupt, corrupt_list = collect_images(
        args.raw_dir, clean=not args.no_clean
    )

    if not class_images:
        print("[ERROR] No images found — check raw directory layout.")
        raise SystemExit(1)

    total = sum(
        len(imgs)
        for variants in class_images.values()
        for imgs in variants.values()
    )
    print(f"Found {total} valid images across {len(class_images)} classes "
          f"({n_corrupt} corrupt files removed)")

    print("\nSplitting and copying …")
    stats = split_and_copy(class_images, args.splits_dir, seed=args.seed)
    print_summary(stats, n_corrupt)
    save_stats(stats, n_corrupt, corrupt_list, args.outputs_dir)
    print(f"Done. Splits written to: {args.splits_dir}")


if __name__ == "__main__":
    main()
