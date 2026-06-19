"""
Image augmentation pipeline for training set.

Returns a Keras Sequential model of preprocessing layers that can be
embedded directly in the model graph (runs on GPU, zero extra I/O).

Used by train.py:
    from augmentation import build_augmentation_layer
    augment = build_augmentation_layer()
    ...
    x = augment(x, training=True)

All parameters are tuned for household waste images (varied lighting,
angles, distances) while staying within realistic distortion ranges.
"""

import tensorflow as tf


def build_augmentation_layer(
    flip: str = "horizontal_and_vertical",
    rotation_factor: float = 0.15,
    zoom_range: float = 0.15,
    brightness_delta: float = 0.2,
    contrast_range: float = 0.2,
    translation_factor: float = 0.1,
    seed: int = 42,
) -> tf.keras.Sequential:
    """
    Build and return a Keras augmentation pipeline.

    Args:
        flip:               "horizontal", "vertical", or "horizontal_and_vertical".
        rotation_factor:    Max rotation as a fraction of 2π (0.15 ≈ ±54°).
        zoom_range:         Fraction for random zoom in/out.
        brightness_delta:   Max absolute brightness shift (0–1 scale).
        contrast_range:     Max contrast factor shift.
        translation_factor: Max translation as a fraction of image dimension.
        seed:               Random seed for reproducibility.

    Returns:
        A tf.keras.Sequential model; apply with training=True during training,
        training=False (or omit) during validation/inference.
    """
    layers = [
        tf.keras.layers.RandomFlip(mode=flip, seed=seed),
        tf.keras.layers.RandomRotation(factor=rotation_factor, seed=seed),
        tf.keras.layers.RandomZoom(
            height_factor=(-zoom_range, zoom_range),
            width_factor=(-zoom_range, zoom_range),
            seed=seed,
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=translation_factor,
            width_factor=translation_factor,
            seed=seed,
        ),
        tf.keras.layers.RandomBrightness(factor=brightness_delta, seed=seed),
        tf.keras.layers.RandomContrast(factor=contrast_range, seed=seed),
    ]
    return tf.keras.Sequential(layers, name="augmentation")


def build_preprocessing_layer(img_size: int = 256) -> tf.keras.Sequential:
    """
    Resize + ResNet-50 normalisation layer.

    Combines resizing and channel-mean subtraction into one reusable layer
    so the same transform is applied identically during training and inference.

    Args:
        img_size: Target height/width (square). Default 256 per IR scope.

    Returns:
        A tf.keras.Sequential model that outputs float32 tensors normalised
        for ResNet-50 (ImageNet mean/std via tf.keras.applications.resnet50.preprocess_input).
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(img_size, img_size),
            tf.keras.layers.Lambda(
                tf.keras.applications.resnet50.preprocess_input,
                name="resnet50_preprocess",
            ),
        ],
        name="preprocessing",
    )
