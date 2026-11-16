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
from joblib import parallel_backend
from threadpoolctl import threadpool_limits
import numpy as np
import csv
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC

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
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[Any],
    train_time_s: float | None = None,
    infer_ms_per_sample: float | None = None,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for team trainers, including ``train_knn``.

    ``labels`` are display names ordered by encoded class ID (0..N-1), as in
    ``data/label_names.json``. Timing fields are retained when provided.
    """
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    result = evaluate_predictions(
        model_name,
        true,
        pred,
        labels=list(range(len(labels))),
        output_path=None,
    )
    result["labels"] = _json_value(labels)
    if train_time_s is not None:
        result["train_time_s"] = float(train_time_s)
    if infer_ms_per_sample is not None:
        result["infer_ms_per_sample"] = float(infer_ms_per_sample)
    if output_path is None:
        slug = "".join(c.lower() if c.isalnum() else "_" for c in str(model_name)).strip("_")
        output_path = RESULTS_DIR / f"{slug}_metrics.json"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _load_array(name: str) -> np.ndarray:
    path = DATA_DIR / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Required processed data is missing: {path}")
    return np.load(path, allow_pickle=False)


def train_classical_models(model_names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Train selected classical models, saving fitted models and metrics."""
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

    groups_path = DATA_DIR / "subjects_train.npy"
    groups = np.load(groups_path, allow_pickle=False) if groups_path.is_file() else None
    if groups is not None and groups.reshape(-1).size != X_train.shape[0]:
        raise ValueError(
            "subjects_train.npy must contain exactly one subject ID per training sample "
            f"({groups.reshape(-1).size} IDs for {X_train.shape[0]} samples)"
        )
    if groups is not None:
        groups = groups.reshape(-1)
    cv = GroupKFold(n_splits=3) if groups is not None else 3
    searches = {
        "svm": (
            make_pipeline(StandardScaler(), SVC()),
            [
                {
                    "svc": [LinearSVC(dual="auto", max_iter=10000, random_state=42)],
                    "svc__C": [0.1],
                },
                {
                    "svc": [SVC()],
                    "svc__C": [1, 10],
                    "svc__kernel": ["rbf"],
                    "svc__gamma": ["scale"],
                },
                {
                    "svc": [SVC()],
                    "svc__C": [10],
                    "svc__kernel": ["rbf"],
                    "svc__gamma": [0.001],
                },
            ],
        ),
        "rf": (
            RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1),
            {"max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 2]},
        ),
    }
    selected = set(model_names or searches)
    unknown = selected - searches.keys()
    if unknown:
        raise ValueError(f"Unknown classical model(s): {', '.join(sorted(unknown))}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, (estimator, param_grid) in searches.items():
        if name not in selected:
            continue
        model_cv = cv
        search = GridSearchCV(
            estimator,
            param_grid,
            cv=model_cv,
            scoring="accuracy",
            n_jobs=2,
            pre_dispatch=2,
            refit=True,
            verbose=2,
        )
        started = time.perf_counter()
        # Threads share input arrays and avoid joblib worker memmap files,
        # which can fail in restricted Windows environments.
        with parallel_backend("threading"), threadpool_limits(limits=1):
            search.fit(X_train, y_train_encoded, groups=groups)
        train_time_s = time.perf_counter() - started
        model = search.best_estimator_
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        best_params = {
            key: value for key, value in search.best_params_.items() if key != "svc"
        }
        best_params["estimator"] = type(model.named_steps["svc"]).__name__
        (RESULTS_DIR / f"{name}_best_params.json").write_text(
            json.dumps({"best_params": best_params, "best_cv_accuracy": float(search.best_score_)}, indent=2) + "\n",
            encoding="utf-8",
        )
        started = time.perf_counter()
        pred = model.predict(X_test)
        batch_infer_ms_per_sample = (time.perf_counter() - started) * 1000 / len(X_test)
        # Measure interactive single-window latency separately from batch throughput.
        latency_samples = X_test[: min(100, len(X_test))]
        model.predict(latency_samples[:1])  # warm up any lazy runtime initialization
        started = time.perf_counter()
        for sample in latency_samples:
            model.predict(sample.reshape(1, -1))
        infer_ms_per_sample = (time.perf_counter() - started) * 1000 / len(latency_samples)
        outputs[name] = evaluate_predictions(
            name.upper(), y_test_encoded, pred, labels=list(range(len(class_names))),
            output_path=RESULTS_DIR / f"{name}_metrics.json",
        )
        # Store original label names in metrics while keeping matrix order aligned.
        outputs[name]["labels"] = _json_value(class_names)
        outputs[name]["train_time_s"] = float(train_time_s)
        outputs[name]["infer_ms_per_sample"] = float(infer_ms_per_sample)
        outputs[name]["batch_infer_ms_per_sample"] = float(batch_infer_ms_per_sample)
        outputs[name]["inference_timing_method"] = (
            f"single-window predict latency averaged over {len(latency_samples)} warmed samples"
        )
        (RESULTS_DIR / f"{name}_metrics.json").write_text(
            json.dumps(outputs[name], indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name.upper()}: accuracy={outputs[name]['accuracy']:.4f}; saved model and metrics")
        if name == "rf":
            importance = model.feature_importances_
            # Flattened window position features use feature order [timestep, channel].
            # Summing over timesteps gives a readable six-sensor-channel ranking.
            if X_test.ndim == 2 and X_train.shape[1] % 6 == 0:
                channel_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
                grouped = importance.reshape(-1, 6).sum(axis=0)
                ranked = sorted(zip(channel_names, grouped.tolist()), key=lambda item: item[1], reverse=True)
                importance_rows = [{"feature": n, "importance": float(v)} for n, v in ranked]
            else:
                importance_rows = [
                    {"feature": f"feature_{i}", "importance": float(value)}
                    for i, value in sorted(enumerate(importance), key=lambda item: item[1], reverse=True)[:30]
                ]
            with (RESULTS_DIR / "rf_feature_importance.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["feature", "importance"])
                writer.writeheader()
                writer.writerows(importance_rows)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-classical", action="store_true", help="Train SVM and Random Forest")
    parser.add_argument(
        "--models", nargs="+", choices=("svm", "rf"), default=None,
        help="Optionally train only selected models (for example: --models svm)",
    )
    args = parser.parse_args()
    if args.train_classical:
        train_classical_models(args.models)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
