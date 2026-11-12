"""
src/predict.py  —  Aryan's deliverable
Loads all five trained models and returns predictions for a given input.

Called by Indraneel's app/app.py to run inference.

Usage (smoke-test from project root):
    python src/predict.py

Input contract (from Akhil's data.py):
    X shape: (N, 128, 6) — N windows, 128 timesteps, 6 inertial channels
    Labels : 0-5 integers mapping to LABELS list below

Output:
    dict[model_name -> np.ndarray of int class predictions, shape (N,)]
"""

from __future__ import annotations

# -- Runtime compatibility shim -- must precede tensorflow import ----------
import sys
from pathlib import Path as _Path

_SRC_ROOT = _Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import src.runtime_compat  # noqa: E402

# -- Standard / third-party imports ----------------------------------------
import os
import time
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_DIR  = ROOT / "data" / "processed"

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
          "SITTING", "STANDING", "LAYING"]

# Model file paths
_PATHS = {
    "CNN":  MODEL_DIR / "cnn.keras",
    "LSTM": MODEL_DIR / "lstm.keras",
    "SVM":  MODEL_DIR / "svm.joblib",
    "RF":   MODEL_DIR / "rf.joblib",
    "KNN":  MODEL_DIR / "knn.joblib",
}


def load_models(verbose: bool = True) -> dict:
    """
    Load all available trained models. Skips any whose file doesn't exist yet.
    Returns a dict keyed by model name.
    """
    import joblib

    models: dict = {}

    # Deep learning models
    for name in ("CNN", "LSTM"):
        path = _PATHS[name]
        if path.exists():
            from tensorflow import keras
            models[name] = keras.models.load_model(str(path))
            if verbose:
                print(f"[predict] {name} loaded <- {path.name}")
        else:
            if verbose:
                print(f"[predict] {name} not found at {path} — skipping")

    # Sklearn models
    for name in ("SVM", "RF", "KNN"):
        path = _PATHS[name]
        if path.exists():
            models[name] = joblib.load(path)
            if verbose:
                print(f"[predict] {name} loaded <- {path.name}")
        else:
            if verbose:
                print(f"[predict] {name} not found at {path} — skipping")

    return models


def predict_all(X: np.ndarray, verbose: bool = False) -> dict[str, np.ndarray]:
    """
    Run all available models on X and return class predictions.

    Args:
        X      : np.ndarray of shape (N, 128, 6) — the windowed inertial data
                 produced by Akhil's data.py (already standardized).
        verbose: print timing per model.

    Returns:
        dict mapping model_name -> np.ndarray of int predictions, shape (N,)
        Predictions are integer indices 0–5 matching LABELS.
    """
    if X.ndim != 3 or X.shape[1:] != (128, 6):
        raise ValueError(f"Expected X shape (N, 128, 6), got {X.shape}")

    models = load_models(verbose=False)
    if not models:
        raise RuntimeError("No models found in models/. Train or download them first.")

    # Flattened view for SVM/RF: (N, 128*6=768)
    X_flat = X.reshape(X.shape[0], -1)

    # KNN uses 85 engineered features, not raw flat windows
    _knn_features = None
    if "KNN" in models:
        try:
            from src.features import extract_batch as _extract_batch
            _knn_features = _extract_batch(X)
            scaler_path = DATA_DIR / "feature_scaler.joblib"
            if scaler_path.exists():
                import joblib
                _scaler = joblib.load(scaler_path)
                _knn_features = _scaler.transform(_knn_features).astype(np.float32)
        except Exception as _knn_feat_err:
            import warnings
            warnings.warn(
                f"[predict] KNN feature extraction failed ({_knn_feat_err}); "
                "falling back to flat features."
            )
            _knn_features = X_flat

    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        if name in ("CNN", "LSTM"):
            raw = model.predict(X, verbose=0)        # (N, 6) softmax
            predictions[name] = np.argmax(raw, axis=1)
        elif name == "KNN":
            predictions[name] = model.predict(_knn_features)
        else:  # SVM, RF
            predictions[name] = model.predict(X_flat)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if verbose:
            print(f"[predict] {name}: {elapsed_ms:.1f}ms for {len(X)} samples")

    return predictions


def predict_single(window: np.ndarray) -> dict[str, int]:
    """
    Convenience wrapper: predict a single window (128, 6) with all models.
    Returns dict[model_name -> int class index].
    """
    if window.shape != (128, 6):
        raise ValueError(f"Expected single window shape (128, 6), got {window.shape}")
    preds = predict_all(window[np.newaxis, ...])
    return {name: int(p[0]) for name, p in preds.items()}


def label_name(class_idx: int) -> str:
    """Convert integer class index to human-readable activity label."""
    return LABELS[class_idx]


# ── Smoke-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    x_path = DATA_DIR / "X_test.npy"
    y_path = DATA_DIR / "y_test.npy"

    if not x_path.exists():
        print("[predict] X_test.npy not found — run Akhil's data.py first.")
    else:
        X_test = np.load(x_path)
        y_test = np.load(y_path)
        print(f"[predict] Test set: {X_test.shape}")

        all_preds = predict_all(X_test, verbose=True)
        print("\n── Results ──────────────────────────────")
        for model_name, preds in all_preds.items():
            acc = float((preds == y_test).mean())
            print(f"  {model_name:5s}: accuracy = {acc:.4f}")
