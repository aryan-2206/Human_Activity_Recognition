"""Train and evaluate Indraneel's LSTM on the shared UCI HAR data contract.

Run from the repository root:
    python src/train_lstm.py
    python src/train_lstm.py --epochs 40 --batch-size 128

The script prepares the public UCI HAR data if needed, holds out training
subjects for validation, saves the best restored model, and evaluates once on
the official test split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
INPUT_SHAPE = (128, 6)
DEFAULT_SEED = 42


def build_lstm(input_shape: tuple[int, int], num_classes: int):
    """Build the two-layer LSTM specified for the HAR project."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape, name="sensor_window")
    x = layers.LSTM(64, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.LSTM(64, name="lstm_2")(x)
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="activity")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LSTM_HAR")
    model.compile(
        optimizer=keras.optimizers.Adam(clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history: Any, output_path: Path) -> None:
    """Save training and validation accuracy/loss curves."""
    fig, (accuracy_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 4))

    accuracy_ax.plot(history.history["accuracy"], label="Train")
    accuracy_ax.plot(history.history["val_accuracy"], label="Validation")
    accuracy_ax.set_title("Accuracy")
    accuracy_ax.set_xlabel("Epoch")
    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.grid(alpha=0.3)
    accuracy_ax.legend()

    loss_ax.plot(history.history["loss"], label="Train")
    loss_ax.plot(history.history["val_loss"], label="Validation")
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(alpha=0.3)
    loss_ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[train_lstm] Curves saved: {output_path}")


def _load_label_names() -> list[str]:
    label_path = DATA_DIR / "label_names.json"
    if label_path.is_file():
        return json.loads(label_path.read_text(encoding="utf-8"))

    # `src.data` owns the canonical order and can provide it when an older
    # local data cache predates label_names.json.
    from src.data import LABELS

    return list(LABELS)


def _validate_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    subjects_train: np.ndarray,
    label_names: list[str],
) -> None:
    if X_train.ndim != 3 or X_train.shape[1:] != INPUT_SHAPE:
        raise ValueError(f"Expected X_train shape (N, 128, 6), got {X_train.shape}")
    if X_test.ndim != 3 or X_test.shape[1:] != INPUT_SHAPE:
        raise ValueError(f"Expected X_test shape (N, 128, 6), got {X_test.shape}")
    if y_train.reshape(-1).size != X_train.shape[0]:
        raise ValueError("X_train and y_train sample counts differ")
    if y_test.reshape(-1).size != X_test.shape[0]:
        raise ValueError("X_test and y_test sample counts differ")
    if subjects_train.reshape(-1).size != X_train.shape[0]:
        raise ValueError("X_train and subjects_train sample counts differ")
    if not label_names:
        raise ValueError("No activity label names were found")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Feature arrays contain NaN or infinite values")

    valid_labels = np.arange(len(label_names))
    train_labels = np.unique(y_train)
    test_labels = np.unique(y_test)
    if not np.isin(train_labels, valid_labels).all() or not np.isin(test_labels, valid_labels).all():
        raise ValueError("Labels must be zero-based indices matching label_names.json")


def train(
    *,
    epochs: int = 40,
    batch_size: int = 128,
    validation_fraction: float = 0.15,
    seed: int = DEFAULT_SEED,
    early_stopping_patience: int = 8,
) -> dict[str, Any]:
    """Prepare data, fit the LSTM, save artifacts, and return test metrics."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1")

    from src.data import prepare_data

    print("[train_lstm] Preparing/loading UCI HAR data...")
    X_train, X_test, y_train, y_test, subjects_train, _subjects_test = prepare_data()
    y_train = np.asarray(y_train).reshape(-1).astype(np.int64)
    y_test = np.asarray(y_test).reshape(-1).astype(np.int64)
    subjects_train = np.asarray(subjects_train).reshape(-1)
    label_names = _load_label_names()
    _validate_data(X_train, X_test, y_train, y_test, subjects_train, label_names)

    unique_subjects = np.unique(subjects_train)
    if unique_subjects.size < 2:
        raise ValueError("At least two training subjects are required for validation")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=seed,
    )
    fit_indices, validation_indices = next(
        splitter.split(X_train, y_train, groups=subjects_train)
    )
    X_fit, y_fit = X_train[fit_indices], y_train[fit_indices]
    X_validation, y_validation = X_train[validation_indices], y_train[validation_indices]

    fit_subjects = np.unique(subjects_train[fit_indices])
    validation_subjects = np.unique(subjects_train[validation_indices])
    overlap = np.intersect1d(fit_subjects, validation_subjects)
    if overlap.size:
        raise RuntimeError(f"Subject leakage across train/validation split: {overlap.tolist()}")

    missing_validation_classes = np.setdiff1d(np.unique(y_train), np.unique(y_validation))
    if missing_validation_classes.size:
        raise ValueError(
            "The validation subjects omit classes "
            f"{missing_validation_classes.tolist()}; choose a different validation split."
        )

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    tf.keras.utils.set_random_seed(seed)
    keras = tf.keras
    keras.backend.clear_session()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = build_lstm(INPUT_SHAPE, num_classes=len(label_names))
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, early_stopping_patience // 2),
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print(
        f"[train_lstm] Fitting {len(fit_indices)} windows from "
        f"{len(fit_subjects)} subjects; validating on {len(validation_indices)} "
        f"windows from {len(validation_subjects)} held-out subjects."
    )
    train_started = time.perf_counter()
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )
    train_time_s = time.perf_counter() - train_started

    model_path = MODEL_DIR / "lstm.keras"
    model.save(model_path)
    print(f"[train_lstm] Model saved: {model_path}")

    benchmark = X_test[: min(256, len(X_test))]
    model.predict(benchmark[: min(16, len(benchmark))], verbose=0)
    inference_started = time.perf_counter()
    model.predict(benchmark, batch_size=batch_size, verbose=0)
    infer_ms_per_sample = (time.perf_counter() - inference_started) * 1000 / len(benchmark)

    probabilities = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    from src.evaluate import evaluate_and_save

    metrics = evaluate_and_save(
        "lstm",
        y_test,
        y_pred,
        label_names,
        train_time_s,
        infer_ms_per_sample,
    )

    # Preserve the current evaluator fields and add the aliases required by
    # the original shared prompt without editing Dhairya's shared helper.
    metrics["model"] = "LSTM"
    metrics["macro_f1"] = float(metrics["f1_macro"])
    metrics["per_class_report"] = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    metrics_path = RESULTS_DIR / "lstm_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    curves_path = RESULTS_DIR / "lstm_curves.png"
    plot_history(history, curves_path)

    print(f"[train_lstm] Test accuracy: {metrics['accuracy']:.4f}")
    print(f"[train_lstm] Test macro-F1: {metrics['macro_f1']:.4f}")
    print(f"[train_lstm] Training time: {train_time_s:.1f}s")
    print(f"[train_lstm] Inference: {infer_ms_per_sample:.4f} ms/sample")
    print(f"[train_lstm] Metrics saved: {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Fraction of training subjects held out for validation",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Early stopping patience on validation loss",
    )
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == "__main__":
    main()
