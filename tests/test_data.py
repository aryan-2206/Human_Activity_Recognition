"""
tests/test_data.py
Verifies that the processed dataset and results files are valid.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Processed data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def processed_data():
    for name in ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"):
        if not (DATA_DIR / name).exists():
            pytest.skip(f"{name} not found — run `python src/data.py` first.")
    return {
        "X_train": np.load(DATA_DIR / "X_train.npy"),
        "X_test":  np.load(DATA_DIR / "X_test.npy"),
        "y_train": np.load(DATA_DIR / "y_train.npy"),
        "y_test":  np.load(DATA_DIR / "y_test.npy"),
    }


def test_window_shape(processed_data):
    assert processed_data["X_train"].shape[1:] == (128, 6), \
        f"Expected windows (N,128,6), got {processed_data['X_train'].shape}"
    assert processed_data["X_test"].shape[1:] == (128, 6)


def test_label_range(processed_data):
    for key in ("y_train", "y_test"):
        y = processed_data[key]
        assert y.min() >= 0 and y.max() <= 5, \
            f"{key} labels out of range 0-5: min={y.min()}, max={y.max()}"


def test_no_nan_in_windows(processed_data):
    assert not np.isnan(processed_data["X_train"]).any(), "NaN in X_train"
    assert not np.isnan(processed_data["X_test"]).any(), "NaN in X_test"


def test_train_test_sizes_consistent(processed_data):
    assert len(processed_data["X_train"]) == len(processed_data["y_train"])
    assert len(processed_data["X_test"]) == len(processed_data["y_test"])


def test_scaler_exists():
    assert (DATA_DIR / "scaler.joblib").exists(), \
        "scaler.joblib not found in data/processed/"


# ---------------------------------------------------------------------------
# Results / metrics files
# ---------------------------------------------------------------------------

METRIC_FILES = ["cnn_metrics.json", "lstm_metrics.json",
                "svm_metrics.json",  "rf_metrics.json", "knn_metrics.json"]

@pytest.mark.parametrize("fname", METRIC_FILES)
def test_metrics_json_valid(fname):
    path = RESULTS_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not present yet.")
    with open(path) as f:
        data = json.load(f)
    assert "accuracy" in data, f"{fname} missing 'accuracy' key"
    assert 0.0 < data["accuracy"] <= 1.0, \
        f"{fname} accuracy out of range: {data['accuracy']}"


def test_comparison_csv_exists():
    path = RESULTS_DIR / "comparison.csv"
    if not path.exists():
        pytest.skip("comparison.csv not present — run `python src/compare.py`.")
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1, "comparison.csv is empty"
    assert "accuracy" in rows[0], "comparison.csv missing 'accuracy' column"
