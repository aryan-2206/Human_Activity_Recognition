"""
src/predict.py  —  Aryan's deliverable
Loads all five trained models and returns predictions for a given input.

Usage (from project root):
    python src/predict.py

Depends on:
    - models/cnn.keras      (Aryan)
    - models/lstm.keras     (Indraneel)
    - models/svm.joblib     (Dhairya)
    - models/rf.joblib      (Dhairya)
    - models/knn.joblib     (Akhil)
    - data/processed/X_test.npy and y_test.npy (Akhil)
"""

import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

CNN_PATH  = os.path.join(MODEL_DIR, "cnn.keras")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm.keras")
SVM_PATH  = os.path.join(MODEL_DIR, "svm.joblib")
RF_PATH   = os.path.join(MODEL_DIR, "rf.joblib")
KNN_PATH  = os.path.join(MODEL_DIR, "knn.joblib")

# Activity labels — update once Akhil confirms the class names
ACTIVITY_LABELS = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
]


def load_models():
    """Load all five trained models. Returns a dict keyed by model name."""
    import joblib

    models = {}

    if os.path.exists(CNN_PATH):
        from tensorflow import keras
        models["CNN"] = keras.models.load_model(CNN_PATH)
        print("[predict] CNN loaded.")
    else:
        print(f"[predict] WARNING: CNN model not found at {CNN_PATH}")

    if os.path.exists(LSTM_PATH):
        from tensorflow import keras
        models["LSTM"] = keras.models.load_model(LSTM_PATH)
        print("[predict] LSTM loaded.")
    else:
        print(f"[predict] WARNING: LSTM model not found at {LSTM_PATH}")

    for name, path in [("SVM", SVM_PATH), ("RF", RF_PATH), ("KNN", KNN_PATH)]:
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"[predict] {name} loaded.")
        else:
            print(f"[predict] WARNING: {name} model not found at {path}")

    return models


def predict_all(X: np.ndarray) -> dict:
    """
    Run all available models on X and return a dict of predictions.

    Args:
        X: numpy array of shape (n_samples, timesteps, features)
           — the windowed/processed format Akhil produces.

    Returns:
        dict mapping model_name -> np.ndarray of integer class predictions
    """
    models = load_models()
    predictions = {}

    for name, model in models.items():
        if name in ("CNN", "LSTM"):
            # Deep learning models expect 3-D input (samples, timesteps, features)
            raw = model.predict(X, verbose=0)
            predictions[name] = np.argmax(raw, axis=1)
        else:
            # Sklearn models expect 2-D input (samples, features)
            X_flat = X.reshape(X.shape[0], -1)
            predictions[name] = model.predict(X_flat)

    return predictions


# ── Quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_test_path = os.path.join(DATA_DIR, "X_test.npy")
    y_test_path = os.path.join(DATA_DIR, "y_test.npy")

    if not os.path.exists(X_test_path):
        print("X_test.npy not found — waiting on Akhil's processed data.")
    else:
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        print(f"Loaded test set: {X_test.shape}")

        preds = predict_all(X_test)
        for model_name, p in preds.items():
            acc = (p == y_test).mean()
            print(f"  {model_name}: accuracy = {acc:.4f}")
