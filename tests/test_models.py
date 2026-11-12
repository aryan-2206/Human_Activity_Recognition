"""
tests/test_models.py
Verifies that all five trained model files load correctly,
produce the right output shapes, and return real (non-random) predictions.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_windows():
    """10 real test windows from the processed dataset."""
    x_path = DATA_DIR / "X_test.npy"
    y_path = DATA_DIR / "y_test.npy"
    if not x_path.exists() or not y_path.exists():
        pytest.skip("Processed data not found — run `python src/data.py` first.")
    X = np.load(x_path).astype(np.float32)[:10]
    y = np.load(y_path)[:10]
    return X, y


@pytest.fixture(scope="module")
def all_models():
    """Load all five models once for the whole test module."""
    import src.runtime_compat  # noqa: F401
    from src.predict import load_models
    return load_models(verbose=False)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def test_cnn_model_loads(all_models):
    assert "CNN" in all_models, "CNN model failed to load"


def test_lstm_model_loads(all_models):
    assert "LSTM" in all_models, "LSTM model failed to load"


def test_svm_model_loads(all_models):
    assert "SVM" in all_models, "SVM model failed to load"


def test_rf_model_loads(all_models):
    assert "RF" in all_models, "RF model failed to load"


def test_knn_model_loads(all_models):
    assert "KNN" in all_models, "KNN model failed to load"


# ---------------------------------------------------------------------------
# Output shape & class range
# ---------------------------------------------------------------------------

def test_cnn_output_shape(all_models, sample_windows):
    X, _ = sample_windows
    preds = np.argmax(all_models["CNN"].predict(X, verbose=0), axis=1)
    assert preds.shape == (10,), f"Expected (10,), got {preds.shape}"
    assert preds.min() >= 0 and preds.max() <= 5


def test_lstm_output_shape(all_models, sample_windows):
    X, _ = sample_windows
    preds = np.argmax(all_models["LSTM"].predict(X, verbose=0), axis=1)
    assert preds.shape == (10,), f"Expected (10,), got {preds.shape}"
    assert preds.min() >= 0 and preds.max() <= 5


def test_svm_output_shape(all_models, sample_windows):
    X, _ = sample_windows
    preds = all_models["SVM"].predict(X.reshape(10, -1))
    assert preds.shape == (10,)
    assert preds.min() >= 0 and preds.max() <= 5


def test_rf_output_shape(all_models, sample_windows):
    X, _ = sample_windows
    preds = all_models["RF"].predict(X.reshape(10, -1))
    assert preds.shape == (10,)
    assert preds.min() >= 0 and preds.max() <= 5


def test_knn_output_shape(all_models, sample_windows):
    from src.features import extract_batch
    X, _ = sample_windows
    preds = all_models["KNN"].predict(extract_batch(X))
    assert preds.shape == (10,)
    assert preds.min() >= 0 and preds.max() <= 5


# ---------------------------------------------------------------------------
# Softmax probabilities sum to 1 (CNN & LSTM)
# ---------------------------------------------------------------------------

def test_cnn_probabilities_sum_to_one(all_models, sample_windows):
    X, _ = sample_windows
    probs = all_models["CNN"].predict(X, verbose=0)
    assert probs.shape == (10, 6)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), atol=1e-5)


def test_lstm_probabilities_sum_to_one(all_models, sample_windows):
    X, _ = sample_windows
    probs = all_models["LSTM"].predict(X, verbose=0)
    assert probs.shape == (10, 6)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), atol=1e-5)


# ---------------------------------------------------------------------------
# Accuracy sanity check on full test set
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_test_data():
    x_path = DATA_DIR / "X_test.npy"
    y_path = DATA_DIR / "y_test.npy"
    if not x_path.exists():
        pytest.skip("Processed data not found.")
    return np.load(x_path).astype(np.float32), np.load(y_path)


def test_cnn_accuracy_above_threshold(all_models, full_test_data):
    X, y = full_test_data
    preds = np.argmax(all_models["CNN"].predict(X, verbose=0), axis=1)
    acc = (preds == y).mean()
    assert acc > 0.85, f"CNN accuracy {acc:.3f} is below expected 85%"


def test_lstm_accuracy_above_threshold(all_models, full_test_data):
    X, y = full_test_data
    preds = np.argmax(all_models["LSTM"].predict(X, verbose=0), axis=1)
    acc = (preds == y).mean()
    assert acc > 0.80, f"LSTM accuracy {acc:.3f} is below expected 80%"


def test_svm_accuracy_above_threshold(all_models, full_test_data):
    X, y = full_test_data
    preds = all_models["SVM"].predict(X.reshape(len(X), -1))
    acc = (preds == y).mean()
    assert acc > 0.80, f"SVM accuracy {acc:.3f} is below expected 80%"


def test_rf_accuracy_above_threshold(all_models, full_test_data):
    X, y = full_test_data
    preds = all_models["RF"].predict(X.reshape(len(X), -1))
    acc = (preds == y).mean()
    assert acc > 0.80, f"RF accuracy {acc:.3f} is below expected 80%"


def test_knn_accuracy_above_threshold(all_models, full_test_data):
    from src.predict import predict_all
    X, y = full_test_data
    preds = predict_all(X)["KNN"]
    acc = (preds == y).mean()
    assert acc > 0.80, f"KNN accuracy {acc:.3f} is below expected 80%"


def test_predict_single_sample(all_models, sample_windows):
    from src.predict import predict_single, label_name
    X, _ = sample_windows
    single_preds = predict_single(X[0])
    assert set(single_preds.keys()) == {"CNN", "LSTM", "SVM", "RF", "KNN"}
    for model_name, class_idx in single_preds.items():
        assert 0 <= class_idx <= 5
        assert isinstance(label_name(class_idx), str)
