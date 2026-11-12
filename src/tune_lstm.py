"""Run a small, subject-wise LSTM tuning sweep within a time budget.

Run from the repository root (recommended in Colab with a GPU):
    python src/tune_lstm.py --time-budget-minutes 30

Configurations are selected using validation macro-F1 only. The official test
split is evaluated once, after selecting the best validation configuration.
The winner replaces models/lstm.keras and refreshes its metrics and curves.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_lstm import (  # noqa: E402
    DATA_DIR,
    DEFAULT_SEED,
    INPUT_SHAPE,
    MODEL_DIR,
    RESULTS_DIR,
    _load_label_names,
    _validate_data,
    build_lstm,
    plot_history,
)

CONFIGURATIONS = (
    {"name": "baseline_u64_d03", "hidden_units": 64, "dropout_rate": 0.3, "bidirectional": False},
    {"name": "unidirectional_u32_d03", "hidden_units": 32, "dropout_rate": 0.3, "bidirectional": False},
    {"name": "unidirectional_u64_d02", "hidden_units": 64, "dropout_rate": 0.2, "bidirectional": False},
    {"name": "unidirectional_u64_d05", "hidden_units": 64, "dropout_rate": 0.5, "bidirectional": False},
    {"name": "bidirectional_u32_d03", "hidden_units": 32, "dropout_rate": 0.3, "bidirectional": True},
)
TABLE_FIELDS = (
    "config",
    "architecture",
    "hidden_units",
    "dropout_rate",
    "bidirectional",
    "validation_accuracy",
    "validation_macro_f1",
    "macro_f1_delta_vs_baseline",
    "best_validation_loss",
    "epochs_completed",
    "train_time_s",
    "selected",
    "status",
)


class _StopAtDeadline:
    """Keras callback that stops the current fit at the global time deadline."""

    def __init__(self, deadline: float, callback_base: type):
        self._callback = callback_base
        self._deadline = deadline

    def as_callback(self):
        callback_base = self._callback
        deadline = self._deadline

        class DeadlineCallback(callback_base):
            def on_epoch_end(self, epoch, logs=None):
                if time.perf_counter() >= deadline:
                    print("[tune_lstm] Time budget reached; ending this fit after the current epoch.")
                    self.model.stop_training = True

        return DeadlineCallback()


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _metric_row(config: dict[str, Any], *, architecture: str) -> dict[str, Any]:
    return {
        "config": config["name"],
        "architecture": architecture,
        "hidden_units": config["hidden_units"],
        "dropout_rate": config["dropout_rate"],
        "bidirectional": config["bidirectional"],
        "validation_accuracy": "",
        "validation_macro_f1": "",
        "macro_f1_delta_vs_baseline": "",
        "best_validation_loss": "",
        "epochs_completed": "",
        "train_time_s": "",
        "selected": False,
        "status": "not_run_time_budget",
    }


def tune(
    *,
    epochs: int = 20,
    batch_size: int = 128,
    validation_fraction: float = 0.15,
    seed: int = DEFAULT_SEED,
    early_stopping_patience: int = 4,
    time_budget_minutes: float = 30.0,
) -> dict[str, Any]:
    """Train variants, pick by validation macro-F1, and evaluate the winner."""
    if epochs < 1 or batch_size < 1 or early_stopping_patience < 1:
        raise ValueError("epochs, batch_size, and early_stopping_patience must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if time_budget_minutes <= 0:
        raise ValueError("time_budget_minutes must be positive")

    from src.data import prepare_data

    print("[tune_lstm] Preparing/loading UCI HAR data...")
    X_train, X_test, y_train, y_test, subjects_train, _subjects_test = prepare_data()
    y_train = np.asarray(y_train).reshape(-1).astype(np.int64)
    y_test = np.asarray(y_test).reshape(-1).astype(np.int64)
    subjects_train = np.asarray(subjects_train).reshape(-1)
    label_names = _load_label_names()
    _validate_data(X_train, X_test, y_train, y_test, subjects_train, label_names)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=seed,
    )
    fit_indices, validation_indices = next(
        splitter.split(X_train, y_train, groups=subjects_train)
    )
    fit_subjects = np.unique(subjects_train[fit_indices])
    validation_subjects = np.unique(subjects_train[validation_indices])
    if np.intersect1d(fit_subjects, validation_subjects).size:
        raise RuntimeError("Subject leakage detected in the validation split")

    X_fit, y_fit = X_train[fit_indices], y_train[fit_indices]
    X_validation, y_validation = X_train[validation_indices], y_train[validation_indices]
    missing_classes = np.setdiff1d(np.unique(y_train), np.unique(y_validation))
    if missing_classes.size:
        raise ValueError(
            "Validation subjects omit classes "
            f"{missing_classes.tolist()}; change the split seed before tuning."
        )

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table_path = RESULTS_DIR / "lstm_tuning.csv"
    rows: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    best_weights: list[np.ndarray] | None = None
    best_history: dict[str, list[float]] | None = None
    best_row: dict[str, Any] | None = None
    best_score: tuple[float, float, float] | None = None
    best_train_time_s = 0.0
    total_train_time_s = 0.0
    started_total = time.perf_counter()
    deadline = started_total + time_budget_minutes * 60

    for index, config in enumerate(CONFIGURATIONS):
        if time.perf_counter() >= deadline:
            rows.extend(
                _metric_row(remaining, architecture="BiLSTM" if remaining["bidirectional"] else "LSTM")
                for remaining in CONFIGURATIONS[index:]
            )
            _write_table(rows, table_path)
            break

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        architecture = "BiLSTM" if config["bidirectional"] else "LSTM"
        model = build_lstm(
            INPUT_SHAPE,
            num_classes=len(label_names),
            hidden_units=config["hidden_units"],
            dropout_rate=config["dropout_rate"],
            bidirectional=config["bidirectional"],
        )
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(2, early_stopping_patience // 2),
                min_lr=1e-5,
                verbose=0,
            ),
            _StopAtDeadline(deadline, tf.keras.callbacks.Callback).as_callback(),
        ]

        print(
            f"[tune_lstm] {index + 1}/{len(CONFIGURATIONS)} {config['name']}: "
            f"{architecture}, {config['hidden_units']} units, dropout={config['dropout_rate']}"
        )
        fit_started = time.perf_counter()
        history = model.fit(
            X_fit,
            y_fit,
            validation_data=(X_validation, y_validation),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=0,
        )
        fit_time_s = time.perf_counter() - fit_started
        total_train_time_s += fit_time_s

        validation_probabilities = model.predict(
            X_validation,
            batch_size=batch_size,
            verbose=0,
        )
        validation_predictions = np.argmax(validation_probabilities, axis=1)
        val_accuracy = float(accuracy_score(y_validation, validation_predictions))
        val_macro_f1 = float(
            f1_score(y_validation, validation_predictions, average="macro", zero_division=0)
        )
        best_val_loss = float(np.min(history.history["val_loss"]))
        epochs_completed = len(history.history["loss"])
        row = {
            **_metric_row(config, architecture=architecture),
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_macro_f1,
            "best_validation_loss": best_val_loss,
            "epochs_completed": epochs_completed,
            "train_time_s": round(fit_time_s, 3),
            "status": "completed",
        }
        score = (val_macro_f1, val_accuracy, -best_val_loss)
        if best_score is None or score > best_score:
            best_score = score
            best_config = dict(config)
            best_weights = [np.array(weight, copy=True) for weight in model.get_weights()]
            best_history = {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            }
            best_train_time_s = fit_time_s
            best_row = row
        row["selected"] = config["name"] == (best_config or {}).get("name")
        rows.append(row)
        if best_row is not None:
            for existing_row in rows:
                existing_row["selected"] = existing_row["config"] == best_row["config"]
        baseline_row = next(
            (existing for existing in rows if existing["config"] == "baseline_u64_d03" and existing["status"] == "completed"),
            None,
        )
        if baseline_row is not None:
            baseline_f1 = float(baseline_row["validation_macro_f1"])
            for existing_row in rows:
                if existing_row["status"] == "completed":
                    existing_row["macro_f1_delta_vs_baseline"] = round(
                        float(existing_row["validation_macro_f1"]) - baseline_f1,
                        6,
                    )
        _write_table(rows, table_path)
        print(
            f"[tune_lstm] validation accuracy={val_accuracy:.4f}, "
            f"macro-F1={val_macro_f1:.4f}, time={fit_time_s:.1f}s"
        )

        del model, history, validation_probabilities, validation_predictions
        gc.collect()
        if time.perf_counter() >= deadline:
            rows.extend(
                _metric_row(remaining, architecture="BiLSTM" if remaining["bidirectional"] else "LSTM")
                for remaining in CONFIGURATIONS[index + 1:]
            )
            _write_table(rows, table_path)
            break

    if best_config is None or best_weights is None or best_history is None or best_row is None:
        raise RuntimeError("No LSTM configuration completed within the time budget")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    best_model = build_lstm(
        INPUT_SHAPE,
        num_classes=len(label_names),
        hidden_units=best_config["hidden_units"],
        dropout_rate=best_config["dropout_rate"],
        bidirectional=best_config["bidirectional"],
    )
    best_model.set_weights(best_weights)
    model_path = MODEL_DIR / "lstm.keras"
    best_model.save(model_path)

    benchmark = X_test[: min(256, len(X_test))]
    best_model.predict(benchmark[: min(16, len(benchmark))], verbose=0)
    inference_started = time.perf_counter()
    best_model.predict(benchmark, batch_size=batch_size, verbose=0)
    infer_ms_per_sample = (time.perf_counter() - inference_started) * 1000 / len(benchmark)
    test_probabilities = best_model.predict(X_test, batch_size=batch_size, verbose=0)
    test_predictions = np.argmax(test_probabilities, axis=1)

    from src.evaluate import evaluate_and_save

    metrics = evaluate_and_save(
        "lstm",
        y_test,
        test_predictions,
        label_names,
        best_train_time_s,
        infer_ms_per_sample,
    )
    metrics["model"] = "LSTM"
    metrics["macro_f1"] = float(metrics["f1_macro"])
    metrics["per_class_report"] = classification_report(
        y_test,
        test_predictions,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["selected_tuning_config"] = {
        "name": best_config["name"],
        "architecture": "BiLSTM" if best_config["bidirectional"] else "LSTM",
        "hidden_units": best_config["hidden_units"],
        "dropout_rate": best_config["dropout_rate"],
        "bidirectional": best_config["bidirectional"],
        "validation_accuracy": best_row["validation_accuracy"],
        "validation_macro_f1": best_row["validation_macro_f1"],
    }
    metrics["tuning_total_train_time_s"] = round(total_train_time_s, 3)
    metrics_path = RESULTS_DIR / "lstm_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_history(SimpleNamespace(history=best_history), RESULTS_DIR / "lstm_curves.png")

    total_elapsed_s = time.perf_counter() - started_total
    print(f"[tune_lstm] Selected: {best_config['name']}")
    print(f"[tune_lstm] Test accuracy: {metrics['accuracy']:.4f}")
    print(f"[tune_lstm] Test macro-F1: {metrics['macro_f1']:.4f}")
    print(f"[tune_lstm] Model saved: {model_path}")
    print(f"[tune_lstm] Results table: {table_path}")
    print(f"[tune_lstm] Total fit time: {total_train_time_s / 60:.2f} min; elapsed: {total_elapsed_s / 60:.2f} min")
    if total_train_time_s > time_budget_minutes * 60:
        print("[tune_lstm] WARNING: one training run crossed the configured fit-time budget.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs per configuration")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
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
        default=4,
        help="Early stopping patience for each configuration",
    )
    parser.add_argument(
        "--time-budget-minutes",
        type=float,
        default=30.0,
        help="Total training fit-time budget across configurations",
    )
    args = parser.parse_args()
    tune(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        time_budget_minutes=args.time_budget_minutes,
    )


if __name__ == "__main__":
    main()
