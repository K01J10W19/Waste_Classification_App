"""
Phase 3 — Model evaluation: confusion matrix, accuracy/loss curves,
classification report.

Loads the best Phase 2 checkpoint (or a specified model path), runs inference
on the test split, and saves all evaluation artefacts to outputs/.

Outputs written to ml-training/outputs/:
    confusion_matrix.png          — normalised heatmap
    confusion_matrix_raw.png      — raw counts heatmap
    training_history.png          — accuracy & loss curves (both phases combined)
    classification_report.txt     — per-class precision/recall/F1
    evaluation_results.json       — summary dict for programmatic use

Usage:
    python ml-training/src/evaluate.py
    python ml-training/src/evaluate.py --model-path ml-training/outputs/checkpoints/phase2_best.keras
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in Colab and headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import tensorflow as tf
from dataset_pipeline import build_test_pipeline

try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    *,
    normalise: bool = True,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Save a confusion matrix heatmap to out_path."""
    if normalise:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=figsize)

    if SNS_AVAILABLE:
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            linewidths=0.5,
        )
    else:
        im = ax.imshow(
            cm_plot,
            interpolation="nearest",
            cmap="Blues",
            **({} if vmin is None else {"vmin": vmin, "vmax": vmax}),
        )
        fig.colorbar(im, ax=ax)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)
        thresh = cm_plot.max() / 2.0
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                val = f"{cm_plot[i, j]:{fmt}}"
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if cm_plot[i, j] > thresh else "black",
                        fontsize=9)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Training history curves
# ---------------------------------------------------------------------------

def _load_csv_history(csv_path: Path) -> dict[str, list[float]]:
    """Load a CSVLogger file into {column: [values]} dict."""
    import csv
    history: dict[str, list[float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                history.setdefault(k, []).append(float(v) if v != "" else float("nan"))
    return history


def plot_training_history(outputs_dir: Path, out_path: Path) -> None:
    """
    Combine Phase 1 and Phase 2 CSVLogger files and plot accuracy + loss curves.
    Each phase is drawn with a distinct colour; a vertical dashed line marks the
    boundary between phases.
    """
    csv1 = outputs_dir / "training_history_phase1.csv"
    csv2 = outputs_dir / "training_history_phase2.csv"

    histories: list[tuple[str, dict]] = []
    if csv1.exists():
        histories.append(("Phase 1", _load_csv_history(csv1)))
    if csv2.exists():
        histories.append(("Phase 2", _load_csv_history(csv2)))

    if not histories:
        print("      [SKIP] No CSV history files found — skipping curves plot.")
        return

    # Concatenate phases
    all_acc, all_val_acc = [], []
    all_loss, all_val_loss = [], []
    phase_boundaries: list[int] = []
    offset = 0

    for phase_name, h in histories:
        epochs = len(h.get("accuracy", h.get("loss", [])))
        all_acc.extend(h.get("accuracy", [float("nan")] * epochs))
        all_val_acc.extend(h.get("val_accuracy", [float("nan")] * epochs))
        all_loss.extend(h.get("loss", [float("nan")] * epochs))
        all_val_loss.extend(h.get("val_loss", [float("nan")] * epochs))
        offset += epochs
        phase_boundaries.append(offset)

    x = np.arange(1, len(all_acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colours = ["#2196F3", "#FF9800"]

    # Draw per-phase segments with distinct colours
    starts = [0] + phase_boundaries[:-1]
    for idx, (phase_name, _) in enumerate(histories):
        s = starts[idx]
        e = phase_boundaries[idx]
        seg_x = x[s:e]
        c = colours[idx % len(colours)]
        ax1.plot(seg_x, all_acc[s:e],     color=c, label=f"{phase_name} train")
        ax1.plot(seg_x, all_val_acc[s:e], color=c, linestyle="--",
                label=f"{phase_name} val")
        ax2.plot(seg_x, all_loss[s:e],    color=c, label=f"{phase_name} train")
        ax2.plot(seg_x, all_val_loss[s:e],color=c, linestyle="--",
                label=f"{phase_name} val")

    # Phase boundary marker
    for boundary in phase_boundaries[:-1]:
        ax1.axvline(x=boundary + 0.5, color="gray", linestyle=":", linewidth=1)
        ax2.axvline(x=boundary + 0.5, color="gray", linestyle=":", linewidth=1)

    ax1.set_title("Model Accuracy", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax1.grid(alpha=0.3)

    ax2.set_title("Model Loss", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("ResNet-50 Transfer Learning — Training History", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

def _sklearn_report(y_true, y_pred, class_names) -> str:
    """Build a classification report using sklearn if available, else manual."""
    try:
        from sklearn.metrics import classification_report
    except ImportError:
        pass
    else:
        return str(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Manual fallback
    lines = [f"{'Class':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}"]
    lines.append("-" * 55)
    for i, name in enumerate(class_names):
        mask_true = y_true == i
        mask_pred = y_pred == i
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sup  = int(mask_true.sum())
        lines.append(f"{name:<14} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {sup:>9}")
    lines.append("-" * 55)
    acc = float((y_true == y_pred).mean())
    lines.append(f"\nOverall accuracy: {acc:.4f}")
    return "\n".join(lines)


def _confusion_matrix_manual(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Compute confusion matrix without sklearn."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load class names -------------------------------------------------
    class_names_path = outputs_dir / "class_names.json"
    if class_names_path.exists():
        with open(class_names_path) as f:
            class_names: list[str] = json.load(f)
    else:
        # Derive from splits directory as fallback
        test_dir = Path(args.splits_dir) / "test"
        class_names = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
        print(f"      [WARN] class_names.json not found; derived from splits: {class_names}")

    print(f"\n[1/5] Classes: {class_names}")

    # ---- Load model -------------------------------------------------------
    model_path = Path(args.model_path)
    if not model_path.exists():
        # Try common fallback paths
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
                f"No model found at {args.model_path}. Run train.py first."
            )

    print(f"\n[2/5] Loading model from {model_path} …")
    model = tf.keras.models.load_model(str(model_path))
    print(f"      Model loaded. Input shape: {model.input_shape}")

    # ---- Build test pipeline ----------------------------------------------
    print("\n[3/5] Running inference on test set …")
    test_ds, _ = build_test_pipeline(
        splits_dir=Path(args.splits_dir),
        batch_size=args.batch_size,
    )

    # Collect ground truth and predictions
    y_true_list, y_pred_list = [], []
    for images, labels in test_ds:
        preds = model.predict_on_batch(images)
        y_true_list.append(labels.numpy())
        y_pred_list.append(np.argmax(preds, axis=-1))

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    accuracy = float((y_true == y_pred).mean())
    print(f"      Test accuracy: {accuracy:.4f}  ({int((y_true == y_pred).sum())}/{len(y_true)})")

    # ---- Confusion matrix ------------------------------------------------
    print("\n[4/5] Plotting confusion matrices …")
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
    except ImportError:
        cm = _confusion_matrix_manual(y_true, y_pred, len(class_names))

    plot_confusion_matrix(
        cm, class_names,
        out_path=outputs_dir / "confusion_matrix.png",
        normalise=True,
        title="Confusion Matrix (Normalised) — ResNet-50",
    )
    plot_confusion_matrix(
        cm, class_names,
        out_path=outputs_dir / "confusion_matrix_raw.png",
        normalise=False,
        title="Confusion Matrix (Raw Counts) — ResNet-50",
    )

    # ---- Training curves -------------------------------------------------
    print("\n      Plotting training history curves …")
    plot_training_history(outputs_dir, outputs_dir / "training_history.png")

    # ---- Classification report -------------------------------------------
    print("\n[5/5] Saving classification report …")
    report_str = _sklearn_report(y_true, y_pred, class_names)
    report_path = outputs_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_str)
    print(f"      Saved -> {report_path}")
    print("\n" + report_str)

    # ---- Summary JSON ----------------------------------------------------
    summary = {
        "model_path": str(model_path),
        "test_accuracy": round(accuracy, 6),
        "n_test_samples": int(len(y_true)),
        "class_names": class_names,
    }
    summary_path = outputs_dir / "evaluation_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n      Summary saved -> {summary_path}")
    print("\nEvaluation complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_outputs = repo_root / "ml-training" / "outputs"
    default_model   = default_outputs / "checkpoints" / "phase2_best.keras"

    parser = argparse.ArgumentParser(
        description="Evaluate waste classifier — confusion matrix, curves, report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default=str(default_model),
        help="Path to a saved Keras model (.keras or .h5).",
    )
    parser.add_argument(
        "--splits-dir",
        default=str(repo_root / "ml-training" / "data" / "splits"),
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(default_outputs),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(args)
