"""
ResNet-50 transfer learning training script for waste classification.

Two-phase training strategy:
  Phase 1 — head-only (frozen base): train the GAP + Dropout + Dense head
             with the ResNet-50 base fully frozen. Uses a high learning rate
             to quickly converge the randomly-initialised head without
             destroying the ImageNet weights.

  Phase 2 — fine-tuning (partial unfreeze): unfreeze the top ~30 layers of
             the ResNet-50 base and retrain with a very low learning rate to
             adapt the high-level feature detectors to waste imagery.

Usage (from repo root):
    python ml-training/src/train.py
    python ml-training/src/train.py --epochs-phase1 15 --epochs-phase2 20
    python ml-training/src/train.py --batch-size 16 --no-class-weights
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dataset_pipeline import build_pipelines

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SPLITS_DIR   = Path(__file__).resolve().parents[2] / "ml-training" / "data" / "splits"
OUTPUTS_DIR  = Path(__file__).resolve().parents[2] / "ml-training" / "outputs"
MODELS_DIR   = Path(__file__).resolve().parents[2] / "ml-training" / "outputs" / "checkpoints"

IMG_SIZE     = 256
BATCH_SIZE   = 32
SEED         = 42
NUM_CLASSES  = 7

# Number of ResNet-50 layers to unfreeze for fine-tuning (counted from the end).
# ResNet-50 has 175 layers total; the last ~30 is the conv5_block layers.
UNFREEZE_FROM = -30


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE) -> tf.keras.Model:
    """
    Build a ResNet-50 transfer learning model.

    Architecture:
        Input(img_size, img_size, 3)
        -> ResNet50 base (imagenet weights, no top)
        -> GlobalAveragePooling2D
        -> Dropout(0.5)
        -> Dense(256, relu) + BatchNormalization
        -> Dropout(0.3)
        -> Dense(num_classes, softmax)

    The input is expected to be ResNet-50 preprocessed (channel-mean
    subtracted, BGR) — the dataset_pipeline handles this before batching.
    """
    # Use input_shape (not input_tensor) so Keras 3 keeps ResNet50 as a
    # named sub-model — required for model.get_layer("resnet50") in Phase 2.
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        pooling=None,
    )
    base.trainable = False  # freeze for Phase 1

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image_input")
    x = base(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.5, seed=SEED, name="drop1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="fc1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Dropout(0.3, seed=SEED, name="drop2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs, outputs, name="waste_classifier_resnet50")


# ---------------------------------------------------------------------------
# Class weights (handle plastic imbalance)
# ---------------------------------------------------------------------------

def compute_class_weights(splits_dir: Path, class_names: list[str]) -> dict[int, float]:
    """
    Compute balanced class weights from the training split file counts.

    Uses sklearn-style balanced formula:
        weight[i] = n_samples / (n_classes * count[i])
    """
    train_dir = splits_dir / "train"
    counts = np.array(
        [len(list((train_dir / c).iterdir())) for c in class_names],
        dtype=np.float32,
    )
    n_samples = counts.sum()
    n_classes = len(class_names)
    weights = n_samples / (n_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def make_callbacks(
    outputs_dir: Path,
    models_dir: Path,
    phase: int,
) -> list[tf.keras.callbacks.Callback]:
    """Return a standard callback set for a training phase."""
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = models_dir / f"phase{phase}_best.keras"

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(outputs_dir / f"training_history_phase{phase}.csv"),
            append=False,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(outputs_dir / f"logs" / f"phase{phase}"),
            histogram_freq=0,
            update_freq="epoch",
        ),
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    splits_dir  = Path(args.splits_dir)
    outputs_dir = Path(args.outputs_dir)
    models_dir  = outputs_dir / "checkpoints"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pipelines --------------------------------------------------------
    print("\n[1/5] Building tf.data pipelines …")
    train_ds, val_ds, _, class_names = build_pipelines(
        splits_dir,
        batch_size=args.batch_size,
        img_size=IMG_SIZE,
        seed=SEED,
    )
    print(f"      Classes ({len(class_names)}): {class_names}")

    # Save class names so evaluate / export / backend can use them without
    # rebuilding the full pipeline.
    class_names_path = outputs_dir / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"      class_names saved -> {class_names_path}")

    # ---- Class weights ----------------------------------------------------
    class_weights = None
    if args.class_weights:
        class_weights = compute_class_weights(splits_dir, class_names)
        print(f"      Class weights: { {class_names[i]: f'{w:.3f}' for i, w in class_weights.items()} }")

    # ---- Build model ------------------------------------------------------
    print("\n[2/5] Building model …")
    model = build_model(num_classes=len(class_names), img_size=IMG_SIZE)
    model.summary(line_length=100)

    # ---- Phase 1: train head ----------------------------------------------
    print(f"\n[3/5] Phase 1 — head training (base frozen, lr=1e-3, max {args.epochs_phase1} epochs) …")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_phase1,
        class_weight=class_weights,
        callbacks=make_callbacks(outputs_dir, models_dir, phase=1),
        verbose=1,
    )
    print(f"      Phase 1 best val_accuracy: {max(history1.history['val_accuracy']):.4f}")

    # ---- Phase 2: fine-tune -----------------------------------------------
    print(f"\n[4/5] Phase 2 — fine-tuning (top {abs(UNFREEZE_FROM)} layers unfrozen, lr=1e-5, max {args.epochs_phase2} epochs) …")
    base = model.get_layer("resnet50")
    base.trainable = True
    for layer in base.layers[:UNFREEZE_FROM]:
        layer.trainable = False
    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"      Trainable layers: {trainable_count}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_phase2,
        class_weight=class_weights,
        callbacks=make_callbacks(outputs_dir, models_dir, phase=2),
        verbose=1,
    )
    print(f"      Phase 2 best val_accuracy: {max(history2.history['val_accuracy']):.4f}")

    # ---- Save final model -------------------------------------------------
    print("\n[5/5] Saving final model …")
    final_path = models_dir / "waste_classifier_final.keras"
    model.save(str(final_path))
    print(f"      Final model saved -> {final_path}")
    print("\nTraining complete. Run evaluate.py to produce metrics and charts.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 waste classifier (Phase 3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--splits-dir",
        default=str(repo_root / "ml-training" / "data" / "splits"),
        help="Root directory containing train/ val/ test/ sub-folders.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(repo_root / "ml-training" / "outputs"),
        help="Directory for checkpoints, CSV history, and class_names.json.",
    )
    parser.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs-phase1",  type=int, default=15,
                        help="Max epochs for Phase 1 (head only).")
    parser.add_argument("--epochs-phase2",  type=int, default=20,
                        help="Max epochs for Phase 2 (fine-tuning).")
    parser.add_argument(
        "--no-class-weights",
        dest="class_weights",
        action="store_false",
        default=True,
        help="Disable class-weight balancing (not recommended — plastic is 37%% of data).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args)
