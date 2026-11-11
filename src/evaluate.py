"""Shared evaluation and classical-model training for the HAR project.

Data contract: ``data/processed/{X,y}_{train,test}.npy``. Features may be
2-D or windowed 3-D arrays; scikit-learn models receive flattened features.
Metrics are written as ``results/<model>_metrics.json`` using one schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def _json_value(value: Any) -> Any:
    """Convert NumPy values recursively into standard JSON values."""
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score predictions, optionally write the shared metrics JSON, return it.

    Every model result contains ``model``, ``accuracy``, macro/weighted
    precision, recall and F1, ``labels``, and a row-normalized-independent
    integer ``confusion_matrix`` (rows are true labels, columns predictions).
    """
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if true.size == 0:
        raise ValueError("Cannot evaluate an empty test set")
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred lengths differ: {true.size} vs {pred.size}")

    classes = list(labels) if labels is not None else list(np.unique(np.concatenate((true, pred))))
    result = {
        "model": str(model_name),
        "accuracy": accuracy_score(true, pred),
        "precision_macro": precision_score(true, pred, labels=classes, average="macro", zero_division=0),
        "recall_macro": recall_score(true, pred, labels=classes, average="macro", zero_division=0),
        "f1_macro": f1_score(true, pred, labels=classes, average="macro", zero_division=0),
        "precision_weighted": precision_score(true, pred, labels=classes, average="weighted", zero_division=0),
        "recall_weighted": recall_score(true, pred, labels=classes, average="weighted", zero_division=0),
        "f1_weighted": f1_score(true, pred, labels=classes, average="weighted", zero_division=0),
        "labels": classes,
        "confusion_matrix": confusion_matrix(true, pred, labels=classes).tolist(),
        "n_samples": int(true.size),
    }
    result = _json_value(result)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def evaluate_and_save(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[Any] | None = None,
    train_time_s: float | None = None,
    infer_ms_per_sample: float | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Team-contract helper: evaluate and save metrics including runtime measurements."""
    if output_path is None:
        output_path = RESULTS_DIR / f"{name.lower()}_metrics.json"
    result = evaluate_predictions(name.upper(), y_true, y_pred, labels=labels, output_path=None)
    if train_time_s is not None:
        result["train_time_s"] = float(train_time_s)
    if infer_ms_per_sample is not None:
        result["infer_ms_per_sample"] = float(infer_ms_per_sample)
    result = _json_value(result)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _load_array(name: str) -> np.ndarray:
    path = DATA_DIR / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Required processed data is missing: {path}")
    return np.load(path, allow_pickle=False)


def train_classical_models() -> dict[str, dict[str, Any]]:
    """Train Dhairya's SVM and Random Forest, saving models and metrics."""
    X_train, X_test = _load_array("X_train"), _load_array("X_test")
    y_train, y_test = _load_array("y_train").reshape(-1), _load_array("y_test").reshape(-1)
    if X_train.shape[0] != y_train.size or X_test.shape[0] != y_test.size:
        raise ValueError("Feature and label sample counts do not match")
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Ensure string/object labels are handled consistently by all sklearn models.
    encoder = LabelEncoder().fit(y_train)
    y_train_encoded = encoder.transform(y_train)
    try:
        y_test_encoded = encoder.transform(y_test)
    except ValueError as exc:
        raise ValueError("Test labels contain classes absent from the training set") from exc
    class_names = encoder.classes_.tolist()

    models = {
        "svm": SVC(),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, model in models.items():
        model.fit(X_train, y_train_encoded)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        pred = model.predict(X_test)
        outputs[name] = evaluate_predictions(
            name.upper(), y_test_encoded, pred, labels=list(range(len(class_names))),
            output_path=RESULTS_DIR / f"{name}_metrics.json",
        )
        # Store original label names in metrics while keeping matrix order aligned.
        outputs[name]["labels"] = _json_value(class_names)
        (RESULTS_DIR / f"{name}_metrics.json").write_text(
            json.dumps(outputs[name], indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name.upper()}: accuracy={outputs[name]['accuracy']:.4f}; saved model and metrics")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-classical", action="store_true", help="Train SVM and Random Forest")
    args = parser.parse_args()
    if args.train_classical:
        train_classical_models()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
