"""
Phase 3 — Export trained model to the backend.

Loads the best checkpoint produced by train.py and writes:
  backend/models/waste_classifier.h5      — legacy HDF5 (Keras 2 compatible)
  backend/models/waste_classifier.keras   — native Keras format (preferred)
  backend/models/class_names.json         — ordered class name list

The backend's preprocess.py is responsible for applying ResNet-50
preprocessing (channel-mean subtraction) before calling model.predict().
The model weights were trained on ResNet-50-preprocessed inputs, so raw
[0-255] uint8 images must pass through the same preprocessing at inference
time.

Usage:
    python ml-training/src/export_model.py
    python ml-training/src/export_model.py --model-path ml-training/outputs/checkpoints/phase2_best.keras
"""

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(args: argparse.Namespace) -> None:
    outputs_dir = Path(args.outputs_dir)
    backend_models_dir = Path(args.backend_models_dir)
    backend_models_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resolve model path -----------------------------------------------
    model_path = Path(args.model_path)
    if not model_path.exists():
        candidates = [
            outputs_dir / "checkpoints" / "phase2_best.keras",
            outputs_dir / "checkpoints" / "waste_classifier_final.keras",
            outputs_dir / "checkpoints" / "phase1_best.keras",
        ]
        for c in candidates:
            if c.exists():
                model_path = c
                break
        else:
            raise FileNotFoundError(
                f"No model checkpoint found. Searched:\n"
                + "\n".join(f"  {c}" for c in candidates)
                + "\nRun train.py first."
            )

    print(f"\n[1/4] Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    model.summary(line_length=100)

    # ---- Load class names -----------------------------------------------
    class_names_src = outputs_dir / "class_names.json"
    if class_names_src.exists():
        with open(class_names_src) as f:
            class_names: list[str] = json.load(f)
    else:
        raise FileNotFoundError(
            f"class_names.json not found at {class_names_src}. "
            "Run train.py to generate it."
        )
    print(f"\n[2/4] Classes ({len(class_names)}): {class_names}")

    # ---- Export .keras (native format) ------------------------------------
    keras_out = backend_models_dir / "waste_classifier.keras"
    model.save(str(keras_out))
    print(f"\n[3/4] Saved .keras -> {keras_out}")

    # ---- Export .h5 (legacy HDF5) -----------------------------------------
    h5_out = backend_models_dir / "waste_classifier.h5"
    model.save(str(h5_out), save_format="h5")
    print(f"      Saved .h5   -> {h5_out}")

    # ---- Export class_names.json ------------------------------------------
    class_names_out = backend_models_dir / "class_names.json"
    with open(class_names_out, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"      Saved class_names -> {class_names_out}")

    # ---- Smoke test -------------------------------------------------------
    print("\n[4/4] Smoke test — single random batch …")
    import numpy as np
    dummy = np.random.randn(1, 256, 256, 3).astype(np.float32)
    # Apply ResNet-50 preprocessing so the input matches training distribution
    dummy_prep = tf.keras.applications.resnet50.preprocess_input(dummy)
    preds = model.predict(dummy_prep, verbose=0)
    predicted_idx = int(np.argmax(preds[0]))
    print(f"      Predicted class: {class_names[predicted_idx]} "
          f"(index {predicted_idx}, confidence {preds[0][predicted_idx]:.4f})")
    print(f"      All class probabilities: "
          + ", ".join(f"{class_names[i]}={preds[0][i]:.3f}" for i in range(len(class_names))))

    print("\nExport complete. Backend is ready to load:")
    print(f"  model    : {keras_out}")
    print(f"  classes  : {class_names_out}")
    print("\nReminder: backend/app/ml/preprocess.py must apply")
    print("  tf.keras.applications.resnet50.preprocess_input(image)")
    print("  before calling model.predict() at inference time.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_outputs      = repo_root / "ml-training" / "outputs"
    default_model        = default_outputs / "checkpoints" / "phase2_best.keras"
    default_backend_dir  = repo_root / "backend" / "models"

    parser = argparse.ArgumentParser(
        description="Export trained waste classifier to backend/models/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default=str(default_model),
        help="Path to the trained model checkpoint (.keras or .h5).",
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(default_outputs),
        help="Directory containing class_names.json (produced by train.py).",
    )
    parser.add_argument(
        "--backend-models-dir",
        default=str(default_backend_dir),
        help="Destination directory inside the backend.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(args)
