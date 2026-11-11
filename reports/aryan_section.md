# Deep Learning (1D-CNN) and Unified Inference Layer — Aryan Doshi

## 1. 1D-CNN Architecture

The 1D Convolutional Neural Network (CNN) operates directly on raw, multichannel time-series windows of shape `(128, 6)` (128 time samples across 3-axis accelerometer and 3-axis gyroscope signals) without requiring hand-crafted feature engineering. The architecture is designed to capture hierarchical temporal dynamics while maintaining high inference speed.

| Layer | Type | Specifications | Output Shape | Parameters |
|---|---|---|---|---|
| Input | InputLayer | Multichannel inertial signals | `(128, 6)` | 0 |
| Block 1 | Conv1D + BN + ReLU | 64 filters, kernel=5, padding="same" | `(128, 64)` | 1,984 |
| Pool 1 | MaxPooling1D | pool_size=2 | `(64, 64)` | 0 |
| Reg 1 | Dropout | rate=0.25 | `(64, 64)` | 0 |
| Block 2 | Conv1D + BN + ReLU | 128 filters, kernel=5, padding="same" | `(64, 128)` | 41,216 |
| Pool 2 | MaxPooling1D | pool_size=2 | `(32, 128)` | 0 |
| Reg 2 | Dropout | rate=0.25 | `(32, 128)` | 0 |
| Block 3 | Conv1D + BN + ReLU | 256 filters, kernel=3, padding="same" | `(32, 256)` | 98,816 |
| Pool 3 | GlobalAveragePooling1D | Temporal average over 32 steps | `(256)` | 0 |
| Reg 3 | Dropout | rate=0.40 | `(256)` | 0 |
| Dense Head | Dense + ReLU | Fully connected feature projection | `(128)` | 32,896 |
| Reg 4 | Dropout | rate=0.30 | `(128)` | 0 |
| Output | Dense (Softmax) | 6 activity classes | `(6)` | 774 |

**Total Parameters:** ~175,686 (Trainable: ~174,800).

---

## 2. Training Setup & Validation Strategy

- **Dataset:** UCI HAR official split: 7,352 training windows and 2,947 test windows.
- **Validation Split:** 10% subject-stratified holdout from the training set.
- **Optimization:** Adam optimizer with an initial learning rate of `1e-3` and sparse categorical cross-entropy loss.
- **Regularization & Callbacks:**
  - `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`: Prevents overfitting on static postures.
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)`: Dynamically halves learning rate when validation loss plateaus.
  - `ModelCheckpoint('models/cnn.keras', monitor='val_accuracy', save_best_only=True)`: Retains the best epoch weights.
- **Hardware & Runtime:** Google Colab GPU (T4). Total training duration was **36.3 seconds** for 25 epochs.

---

## 3. Results & Curve Interpretation

- **Test Accuracy:** **91.45%** (highest on the team leaderboard)
- **Macro F1-Score:** **0.9135** (Weighted F1: 0.9130)
- **Inference Latency:** **0.40 ms per sample** (~2,500 predictions/second)

### Confusion Matrix Breakdown:
- **Dynamic Activities:** `LAYING` (100% precision/recall), `WALKING` (95.2% recall), and `WALKING_DOWNSTAIRS` (100% precision) showed minimal confusion.
- **Static Activity Confusion:** The primary source of misclassifications occurred between `SITTING` and `STANDING` (51 standing windows misclassified as sitting, and 123 sitting windows misclassified as standing). This is a well-documented physical limitation of waist-worn accelerometers where gravitational orientation is nearly identical between seated and upright postures.

---

## 4. Unified Inference Layer (`src/predict.py`)

To bridge the gap between model development and application deployment, `src/predict.py` provides an abstracted inference engine:
- **`load_models()`**: Dynamically detects and caches loaded models (`CNN`, `LSTM`, `SVM`, `RF`, `KNN`) from `models/`.
- **Dimensionality Adapters**: Seamlessly handles shape differences between deep learning models (expecting `(N, 128, 6)`) and classical scikit-learn models (expecting flattened `(N, 768)`).
- **`predict_single(window)` & `predict_all(X)`**: Standardized API consumed directly by Indraneel's Streamlit application (`app/app.py`), returning human-readable activity labels, integer class codes, and predicted class probabilities.
