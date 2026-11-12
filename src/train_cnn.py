"""
src/train_cnn.py  —  Aryan's CNN training script
Run on Google Colab (free GPU) after Akhil's data is in data/processed/.

Usage:
    python src/train_cnn.py
    python src/train_cnn.py --epochs 30 --batch-size 64

Produces:
    models/cnn.keras
    results/cnn_metrics.json
    results/cnn_confusion.png
    results/cnn_training_history.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data" / "processed"
MODEL_DIR  = ROOT / "models"
RESULT_DIR = ROOT / "results"

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
          "SITTING", "STANDING", "LAYING"]
NUM_CLASSES = len(LABELS)


# ── Model definition ───────────────────────────────────────────────────────────

def build_cnn(input_shape: tuple[int, int], num_classes: int = NUM_CLASSES):
    """
    1-D CNN for time-series HAR.
    input_shape: (timesteps, channels) = (128, 6)
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape, name="input")

    # Block 1
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # Block 2
    x = layers.Conv1D(128, kernel_size=5, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dropout(0.4, name="drop3")(x)

    # Classifier head
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="drop4")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="CNN_HAR")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_history(history, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[train_cnn] Training history saved → {out_path}")


def plot_confusion(cm: list[list[int]], out_path: Path) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_arr, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES), LABELS, rotation=40, ha="right")
    ax.set_yticks(range(NUM_CLASSES), LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("CNN Confusion Matrix")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[train_cnn] Confusion matrix saved → {out_path}")


# ── Main training routine ──────────────────────────────────────────────────────

def train(epochs: int = 25, batch_size: int = 64) -> None:
    # 1. Load data
    print("[train_cnn] Loading data...")
    X_train = np.load(DATA_DIR / "X_train.npy")   # (7352, 128, 6)
    X_test  = np.load(DATA_DIR / "X_test.npy")    # (2947, 128, 6)
    y_train = np.load(DATA_DIR / "y_train.npy")   # (7352,) int64, labels 0–5
    y_test  = np.load(DATA_DIR / "y_test.npy")    # (2947,) int64, labels 0–5

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")

    # 2. Build model
    model = build_cnn(input_shape=(128, 6))
    model.summary()

    # 3. Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(MODEL_DIR / "cnn.keras"), monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

    # 4. Train
    print(f"\n[train_cnn] Training for up to {epochs} epochs, batch size {batch_size}...")
    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.perf_counter() - t0
    print(f"[train_cnn] Training finished in {train_time:.1f}s")

    # 5. Evaluate
    print("[train_cnn] Evaluating on test set...")
    # Time inference on a sample batch
    sample = X_test[: min(200, len(X_test))]
    model.predict(sample, verbose=0)                       # warm-up
    t1 = time.perf_counter()
    model.predict(sample, verbose=0)
    infer_ms = (time.perf_counter() - t1) * 1000 / len(sample)

    raw_preds = model.predict(X_test, verbose=0)
    y_pred = np.argmax(raw_preds, axis=1)

    # 6. Save metrics via Dhairya's shared helper
    from src.evaluate import evaluate_predictions
    metrics = evaluate_predictions(
        model_name="CNN",
        y_true=y_test,
        y_pred=y_pred,
        labels=list(range(NUM_CLASSES)),
        output_path=RESULT_DIR / "cnn_metrics.json",
    )
    # Patch in human-readable label names (same pattern as Dhairya's train_classical_models)
    metrics["labels"] = LABELS
    metrics["train_time_s"] = round(train_time, 2)
    metrics["infer_ms_per_sample"] = round(infer_ms, 4)
    (RESULT_DIR / "cnn_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n[train_cnn] ── Results ──────────────────────────────")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1 macro  : {metrics['f1_macro']:.4f}")
    print(f"  Train time: {train_time:.1f}s  |  Infer: {infer_ms:.3f}ms/sample")
    print(f"  Model     → {MODEL_DIR / 'cnn.keras'}")
    print(f"  Metrics   → {RESULT_DIR / 'cnn_metrics.json'}")

    # 7. Plots
    plot_history(history, RESULT_DIR / "cnn_training_history.png")
    plot_confusion(metrics["confusion_matrix"], RESULT_DIR / "cnn_confusion.png")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",     type=int, default=25, help="Max training epochs (default 25)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default 64)")
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
