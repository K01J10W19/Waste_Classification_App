"""
tf.data input pipelines for train / val / test splits.

Pipeline order (training):
    load + resize → ResNet50 preprocess → [cache] → shuffle
    → batch → augment → prefetch

Pipeline order (val / test):
    load + resize → ResNet50 preprocess → batch → prefetch

Usage:
    from dataset_pipeline import build_pipelines

    train_ds, val_ds, test_ds, class_names = build_pipelines(
        splits_dir="ml-training/data/splits"
    )
    # train_ds, val_ds, test_ds yield (images, labels) batches.
    # images: float32, ResNet50-normalised, shape (B, 256, 256, 3)
    # labels: int32 index into class_names (sorted alphabetically)
"""

import sys
from pathlib import Path

# Allow running this file directly from any working directory
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import tensorflow as tf
from augmentation import build_augmentation_layer

AUTOTUNE = tf.data.AUTOTUNE
DEFAULT_IMG_SIZE = 256
DEFAULT_BATCH_SIZE = 32


def _resnet_preprocess(image: tf.Tensor, label: tf.Tensor):
    """Apply ResNet-50 channel-mean subtraction to a single image."""
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label


def build_train_pipeline(
    splits_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: int = DEFAULT_IMG_SIZE,
    seed: int = 42,
    cache: bool = False,
    cache_path: str = "",
) -> tuple[tf.data.Dataset, list[str]]:
    """
    Build the training tf.data pipeline with augmentation.

    Args:
        splits_dir:  Root directory containing train/ val/ test/ sub-folders.
        batch_size:  Number of samples per batch.
        img_size:    Resize target (square). Must match ResNet-50 input.
        seed:        RNG seed for shuffle and augmentation.
        cache:       Cache preprocessed images to speed up subsequent epochs.
        cache_path:  File path for on-disk cache (empty = in-memory cache).
                     In-memory cache is only viable if the dataset fits in RAM.

    Returns:
        (dataset, class_names)
        class_names is sorted alphabetically and determines label→name mapping.
    """
    train_dir = Path(splits_dir) / "train"

    # Unbatched so we can shuffle individual samples before batching
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=None,
        shuffle=False,          # manual shuffle below for full control
        seed=seed,
        label_mode="int",
    )
    class_names: list[str] = raw_ds.class_names

    ds = raw_ds.map(_resnet_preprocess, num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache(cache_path)

    # Shuffle individual samples, then batch; this gives better randomness
    # than shuffling batches.
    ds = (
        ds
        .shuffle(buffer_size=2_000, seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
    )

    # Augmentation applied per-batch, training=True activates random ops
    augment = build_augmentation_layer(seed=seed)
    ds = ds.map(
        lambda imgs, lbls: (augment(imgs, training=True), lbls),
        num_parallel_calls=AUTOTUNE,
    )

    return ds.prefetch(AUTOTUNE), class_names


def build_val_pipeline(
    splits_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: int = DEFAULT_IMG_SIZE,
    cache: bool = False,
    cache_path: str = "",
) -> tuple[tf.data.Dataset, list[str]]:
    """
    Build the validation tf.data pipeline (no augmentation, no shuffle).

    Returns:
        (dataset, class_names)
    """
    val_dir = Path(splits_dir) / "val"

    raw_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(img_size, img_size),
        batch_size=None,
        shuffle=False,
        label_mode="int",
    )
    class_names: list[str] = raw_ds.class_names

    ds = raw_ds.map(_resnet_preprocess, num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache(cache_path)

    return ds.batch(batch_size).prefetch(AUTOTUNE), class_names


def build_test_pipeline(
    splits_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: int = DEFAULT_IMG_SIZE,
) -> tuple[tf.data.Dataset, list[str]]:
    """
    Build the test tf.data pipeline (no augmentation, no shuffle, no cache).

    Returns:
        (dataset, class_names)
    """
    test_dir = Path(splits_dir) / "test"

    raw_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_size, img_size),
        batch_size=None,
        shuffle=False,
        label_mode="int",
    )
    class_names: list[str] = raw_ds.class_names

    ds = raw_ds.map(_resnet_preprocess, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE), class_names


def build_pipelines(
    splits_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: int = DEFAULT_IMG_SIZE,
    seed: int = 42,
    cache_train: bool = False,
    cache_val: bool = False,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Convenience wrapper — returns all three pipelines and shared class_names.

    Args:
        splits_dir:   Root splits directory.
        batch_size:   Batch size for all three pipelines.
        img_size:     Image resize target (square pixels).
        seed:         RNG seed for training shuffle and augmentation.
        cache_train:  Cache train pipeline in memory (needs ~11 GB RAM for
                      full 15 k dataset at 256×256 float32 — use with caution).
        cache_val:    Cache val pipeline in memory (needs ~1.7 GB).

    Returns:
        (train_ds, val_ds, test_ds, class_names)
    """
    train_ds, class_names = build_train_pipeline(
        splits_dir, batch_size, img_size, seed, cache=cache_train
    )
    val_ds, _ = build_val_pipeline(
        splits_dir, batch_size, img_size, cache=cache_val
    )
    test_ds, _ = build_test_pipeline(splits_dir, batch_size, img_size)
    return train_ds, val_ds, test_ds, class_names
