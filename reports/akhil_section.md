# Data Preprocessing, Feature Engineering, and KNN — Akhil

## 1. Dataset & Preprocessing Pipeline (`src/data.py`)

The primary dataset used across all models is the official **UCI Human Activity Recognition (HAR) Using Smartphones** dataset. It consists of inertial sensor recordings collected from 30 volunteers aged 19–48 wearing a waist-mounted smartphone (Samsung Galaxy S II) capturing 3-axis linear acceleration and 3-axis angular velocity at 50 Hz.

### Data Contract & Windowing:
- **Raw Signal Windows:** Continuous signals are partitioned into fixed-width sliding windows of **128 samples (2.56 seconds)** with a 50% overlap (64 samples).
- **Channels (6):** `total_acc_x`, `total_acc_y`, `total_acc_z`, `body_gyro_x`, `body_gyro_y`, `body_gyro_z`.
- **Split:** Subject-wise partition separating the 30 participants into **21 training subjects** (7,352 windows) and **9 test subjects** (2,947 windows). This ensures zero subject leakage across splits.
- **Normalization:** A channel-wise `StandardScaler` is fit **strictly on the training split** and applied to both training and test arrays, preserving test set independence. The fitted scaler is saved to `data/processed/scaler.joblib`.
- **Labels (6 Classes):** `WALKING` (0), `WALKING_UPSTAIRS` (1), `WALKING_DOWNSTAIRS` (2), `SITTING` (3), `STANDING` (4), `LAYING` (5).

---

## 2. Exploratory Data Analysis (EDA) (`src/eda.py`)

Key exploratory findings derived from the dataset:
- **Class Balance:** Class distributions in both the train (~1,200 windows/class) and test sets (~490 windows/class) are balanced, with no single majority class dominating.
- **Static vs. Dynamic Separation:**
  - **Dynamic activities** (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`) exhibit large periodic oscillations in accelerometer and gyroscope signals, with standard deviations and kinetic energy significantly higher than static postures.
  - **Static activities** (`SITTING`, `STANDING`, `LAYING`) exhibit near-zero signal variance across windows, governed almost entirely by the constant 1g gravitational component.
- **Static Posture Overlap:** While `LAYING` is easily isolated due to gravitational acceleration shifting from the Y/Z axis to the X axis, `SITTING` and `STANDING` share nearly identical waist tilt angles, leading to natural sensor overlap.

---

## 3. Time- and Frequency-Domain Feature Engineering (`src/features.py`)

While deep models consume raw `(128, 6)` arrays, classical machine learning algorithms require engineered tabular representations. An **85-dimensional feature vector** is extracted per window, supporting both batch processing and real-time single-window inference for the Streamlit app:

1. **Per-Channel Statistical Features (12 stats × 6 channels = 72 features):**
   - *Time-domain:* Mean, standard deviation, minimum, maximum, median, median absolute deviation (MAD), interquartile range (IQR), signal energy ($x^2$), skewness, and kurtosis.
   - *Frequency-domain (FFT):* Discrete Fourier Transform (via `np.fft.rfft`) computed on centered signals to extract the dominant non-DC peak frequency (`dominant_fft_hz`) and overall spectral energy.
2. **Euclidean Magnitude Statistics (5 stats × 2 sensor types = 10 features):**
   - Norm computed across 3 axes for total acceleration ($\sqrt{a_x^2 + a_y^2 + a_z^2}$) and angular velocity ($\sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}$).
   - Summary statistics: mean, standard deviation, minimum, maximum, and energy.
3. **Cross-Axis Correlation (3 features):**
   - Pearson correlation coefficients across accelerometer axis pairs: $(a_x, a_y)$, $(a_x, a_z)$, and $(a_y, a_z)$.

All features are normalized using a `StandardScaler` fit on the training feature matrix `F_train` and saved to `data/processed/feature_scaler.joblib`.

---

## 4. K-Nearest Neighbors (KNN) Baseline & Tuning (`src/train_knn.py`)

- **Cross-Validation Setup:** To prevent data leakage between repeated windows from the same individual, hyperparameter selection uses `GroupKFold(n_splits=3)` grouped by `subjects_train`.
- **Search Space:**
  - `n_neighbors`: $[3, 5, 7, 9, 11]$
  - `weights`: `['uniform', 'distance']`
  - `metric`: `['euclidean', 'manhattan']`
- **Tuning Observations:** Distance-weighted voting consistently outperformed uniform voting across all values of $k$. Manhattan distance achieved marginally tighter neighborhood bounds than Euclidean distance.
- **Selected Best Parameters:** `metric='manhattan'`, `n_neighbors=9`, `weights='distance'`.
- **Model Checkpoint:** Saved to `models/knn.joblib`.

---

## 5. Performance & Findings

- **Test Accuracy:** **88.60%**
- **Macro F1-Score:** **0.8836** (Weighted F1: 0.8857)
- **Inference Latency:** **0.07 ms per sample** (Fastest model in the benchmark; ~14,000 samples/sec)
- **Class-Level Observations:**
  - `LAYING` achieved **100% precision and 100% recall**.
  - `WALKING` achieved strong discrimination (93.8% recall).
  - Consistent with the EDA, the primary confusion occurred between `SITTING` (79.8% recall) and `STANDING` (89.5% recall), with 56 standing windows misclassified as sitting and 97 sitting windows misclassified as standing.
