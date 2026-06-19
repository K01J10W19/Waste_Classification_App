"""
Image augmentation and preprocessing layers for the waste classification model.

Two distinct concerns are separated here:

  build_augmentation_layer()   — random transforms applied to TRAINING SET ONLY.
                                 Pass training=True when calling; pass False or
                                 omit during validation / inference.

  build_preprocessing_layer()  — deterministic resize + ResNet-50 normalisation
                                 applied to ALL splits (train, val, test) and
                                 at inference time.
                                 Also embedded at the top of the Keras model so
                                 the exported .h5 / SavedModel handles raw
                                 [0-255] pixel input without a separate step.

Both are used by dataset_pipeline.py to build the tf.data input pipelines.
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
    Build a Keras augmentation pipeline (TRAIN SET ONLY).

    Runs inside the model graph on GPU; zero extra I/O cost.
    Apply with training=True so ops are no-ops at eval time.

    Args:
        flip:               "horizontal", "vertical", or "horizontal_and_vertical".
        rotation_factor:    Max rotation as fraction of 2π (0.15 ≈ ±54°).
        zoom_range:         Fraction for random zoom in/out.
        brightness_delta:   Max absolute brightness shift on [0-1] scale.
        contrast_range:     Max contrast factor shift.
        translation_factor: Max translation as fraction of image dimension.
        seed:               Fixed seed for reproducibility.

    Returns:
        tf.keras.Sequential that accepts a batch of float32 images.
    """
    return tf.keras.Sequential(
        [
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
        ],
        name="augmentation",
    )


def build_preprocessing_layer(img_size: int = 256) -> tf.keras.Sequential:
    """
    Build a deterministic preprocessing pipeline (ALL splits + inference).

    Steps:
      1. Resize to (img_size × img_size) — matches IR scope of 256 px.
      2. ResNet-50 channel-mean subtraction (ImageNet statistics, BGR order)
         via tf.keras.applications.resnet50.preprocess_input.

    Embed this at the top of the Keras model so the exported artifact
    accepts raw uint8 images without a separate preprocessing step.

    Args:
        img_size: Target height = width in pixels (default 256).

    Returns:
        tf.keras.Sequential returning float32 tensors in ResNet-50 range.
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
