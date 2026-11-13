# 🏃 Human Activity Recognition

A collaborative machine-learning project that trains and compares five models — **CNN**, **LSTM**, **SVM**, **Random Forest**, and **KNN** — on the [UCI HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and exposes the results through an interactive **Streamlit** dashboard.

---

## 📊 Model Results (UCI HAR test set, 2 947 windows)

| Model | Accuracy | F1 (macro) | F1 (weighted) | Infer (ms/sample) |
|-------|----------|------------|---------------|-------------------|
| **CNN** | **91.4 %** | **91.4 %** | **91.3 %** | 0.40 |
| KNN | 88.6 % | 88.4 % | 88.6 % | 0.07 |
| SVM | 88.5 % | 88.3 % | 88.5 % | 22.6 |
| LSTM | 87.0 % | 86.9 % | 87.0 % | 1.40 |
| RF | 83.8 % | 83.6 % | 83.8 % | 0.17 |

---

## 👥 Team & Contributions

| Member | Branch | Deliverables |
|--------|--------|--------------|
| **Akhil** | `akhil-data-knn` | `src/data.py`, `src/features.py`, `src/train_knn.py`, `models/knn.joblib`, `results/knn_*` |
| **Dhairya** | `dhairya-classical` | `src/evaluate.py`, `src/compare.py`, `src/eda.py`, `models/svm.joblib`, `models/rf.joblib`, `results/svm_*`, `results/rf_*`, `results/comparison.*` |
| **Aryan** | `aryan-cnn` | `src/predict.py`, `src/runtime_compat.py`, `src/train_cnn.py`, `models/cnn.keras`, `results/cnn_*` |
| **Indraneel** | `indraneel-app` | `app/app.py`, `src/train_lstm.py`, `src/tune_lstm.py`, `models/lstm.keras`, `results/lstm_*` |

---

## 🗂️ Project Structure

```
Human_Activity_Recognition/
├── app/
│   └── app.py                  # Streamlit dashboard (Indraneel)
├── data/
│   ├── raw/                    # UCI HAR zip + extracted dataset (git-ignored)
│   └── processed/              # NumPy arrays + scaler (git-ignored, see below)
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── scaler.joblib
│       └── label_names.json
├── models/                     # Saved model files (tracked in git, all < 50 MB)
│   ├── cnn.keras
│   ├── lstm.keras
│   ├── svm.joblib
│   ├── rf.joblib
│   └── knn.joblib
├── notebooks/
│   ├── cnn_training_colab.ipynb
│   └── lstm_training_colab.ipynb
├── reports/                    # Individual member write-ups
│   ├── akhil_section.md
│   ├── aryan_section.md
│   ├── dhairya_section.md
│   └── indraneel_section.md
├── results/                    # Metrics JSON, confusion matrices, plots
│   ├── comparison.csv
│   ├── comparison_metrics.png
│   ├── cnn_metrics.json  / cnn_confusion.png  / cnn_training_history.png
│   ├── lstm_metrics.json / lstm_curves.png    / lstm_tuning.csv
│   ├── svm_metrics.json  / svm_best_params.json
│   ├── rf_metrics.json   / rf_best_params.json / rf_feature_importance.csv
│   └── knn_metrics.json  / knn_confusion.png  / knn_k_curve.png
├── src/
│   ├── data.py                 # Dataset download, windowing, preprocessing (Akhil)
│   ├── eda.py                  # Exploratory data analysis (Dhairya)
│   ├── features.py             # 85 engineered features for KNN (Akhil)
│   ├── evaluate.py             # Shared evaluation + metrics JSON writer (Dhairya)
│   ├── compare.py              # Cross-model comparison CSV + chart (Dhairya)
│   ├── predict.py              # Unified inference for all 5 models (Aryan)
│   ├── runtime_compat.py       # Windows DLL-order shim + protobuf stub (Aryan)
│   ├── train_cnn.py            # CNN training script (Aryan)
│   ├── train_knn.py            # KNN training script (Akhil)
│   ├── train_lstm.py           # LSTM training script (Indraneel)
│   └── tune_lstm.py            # LSTM hyperparameter search (Indraneel)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment

| Property | Value |
|----------|-------|
| Python | 3.12.4 (64-bit, Windows 11) |
| TensorFlow | 2.20.0 |
| Keras | 3.11.2 |
| NumPy | 1.26.4 |
| SciPy | 1.15.2 |
| Scikit-learn | 1.3.2 |
| Pandas | 2.3.1 |
| Streamlit | 1.67.0 |
| Plotly | 7.1.0 |
| protobuf | 4.25.8 (**pinned** — do not upgrade to ≥ 5) |

> [!IMPORTANT]
> `protobuf` must stay at `4.x`. MediaPipe 0.10.x requires `protobuf<5`. TensorFlow 2.20 expects `google.protobuf.runtime_version` (added in protobuf 5.26+); a compatibility stub in `src/runtime_compat.py` bridges the gap.

---

## 🚀 Setup

### 1. Clone the repository

```powershell
git clone <repo-url>
cd Human_Activity_Recognition
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

The processed `.npy` arrays are **not stored in git** (they are large binary files).  
Run the data preparation script once; it downloads the UCI HAR dataset automatically:

```powershell
python src/data.py
```

This creates `data/processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `scaler.joblib`, and `label_names.json`.

---

## ▶️ Running the App

```powershell
# From the project root with the venv active:
streamlit run app/app.py
```

The app opens at `http://localhost:8501` and shows:
- Dataset overview & EDA
- Per-model confusion matrices and classification reports  
- Five-model comparison charts (Plotly)
- Live inference on test windows

> [!NOTE]
> `src/runtime_compat.py` is imported as the **very first** statement in `app/app.py`. This resolves the Windows C-runtime DLL conflict between TensorFlow and OpenBLAS/OpenMP. Do not move or remove that import.

---

## 🤖 Model Details

### CNN (`models/cnn.keras`)
- **Input:** `(N, 128, 6)` — 128-timestep windows, 6 inertial channels
- **Architecture:** 2× Conv1D → GlobalAvgPool → Dense → Softmax (6 classes)
- **Training:** Google Colab (GPU), 50 epochs, Adam, cross-entropy
- **Test accuracy:** 91.4 %

### LSTM (`models/lstm.keras`)
- **Input:** `(N, 128, 6)`
- **Architecture:** Unidirectional LSTM (64 units) → Dropout(0.5) → Dense → Softmax
- **Training:** Colab (GPU), selected via hyperparameter tuning (`src/tune_lstm.py`)
- **Test accuracy:** 87.0 %

### SVM (`models/svm.joblib`)
- **Input:** `(N, 768)` — raw flat windows (128 × 6)
- **Kernel:** RBF, tuned with `GridSearchCV`
- **Test accuracy:** 88.5 %

### Random Forest (`models/rf.joblib`)
- **Input:** `(N, 768)` — raw flat windows
- **Trees:** 200, tuned with `RandomizedSearchCV`
- **Test accuracy:** 83.8 %

### KNN (`models/knn.joblib`)
- **Input:** `(N, 85)` — engineered time/frequency-domain features (`src/features.py`)
- **k = 5**, distance-weighted
- **Test accuracy:** 88.6 %

> [!NOTE]
> KNN uses **85 engineered features** (not raw flat windows). `src/predict.py` calls `src.features.extract_batch(X)` automatically before passing data to KNN.

---

## 🔧 Training Scripts

Re-train individual models from processed data (requires `data/processed/` to be populated):

```powershell
# CNN  (best run on GPU / Colab)
python src/train_cnn.py

# LSTM — hyperparameter search then final training
python src/tune_lstm.py
python src/train_lstm.py

# Classical models (runs locally, SVM is slow ~20 min)
python src/evaluate.py --train-classical

# KNN
python src/train_knn.py
```

After `cnn_metrics.json`, `knn_metrics.json`, `svm_metrics.json`, `rf_metrics.json`,
and `lstm_metrics.json` are present, regenerate the five-model comparison with:

```powershell
python src/compare.py
```

---

## 🧪 Testing

```powershell
# Quick smoke-test of all models on the test set
python src/predict.py

# Streamlit app syntax check (no browser)
python -c "import ast; ast.parse(open('app/app.py').read()); print('app.py OK')"
```

---

## 🐛 Troubleshooting

### `DLL load failed while importing _pywrap_tensorflow_internal`

**Cause:** On Windows, NumPy/SciPy/Scikit-learn lock the OpenBLAS C-runtime before TensorFlow's native DLL initialises.

**Fix (already applied):** `app/app.py` and `src/predict.py` both import `src.runtime_compat` as their very first import. The shim pre-initialises TensorFlow's native wrapper before any other library loads. **Do not add `import numpy` or `import sklearn` above the shim line.**

### `cannot import name 'runtime_version' from 'google.protobuf'`

**Cause:** `protobuf 4.x` (required by MediaPipe 0.10.x) is missing the `runtime_version` sub-module that TensorFlow 2.20 generated code tries to import.

**Fix (already applied):** `src/runtime_compat.py` injects a stub `google.protobuf.runtime_version` module into `sys.modules` before TensorFlow is imported.

**Do not upgrade protobuf to ≥ 5** without first checking MediaPipe compatibility.

### `InconsistentVersionWarning` for sklearn models

The `.joblib` model files were saved with scikit-learn 1.9.x; the installed version is 1.3.2. The models load and predict correctly despite the warning. To silence it, upgrade scikit-learn to 1.9.x (verify NumPy/SciPy compatibility first).

### `Demo-only dummy predictions` shown in the app

This means TensorFlow failed to load. Check that:
1. `src.runtime_compat` is the first import in the entry-point file.
2. `protobuf==4.25.8` is installed (`pip show protobuf`).
3. You are using Python 3.12 (64-bit).

### Dataset not found

Run `python src/data.py` from the project root. The script downloads the UCI HAR zip (~60 MB) automatically and writes the processed arrays.

---

## 📁 Large Files

Model files and processed data above ~50 MB are **not** stored in this repository.  
📁 **Google Drive folder:** *(add link here)*

Files shared via Drive:
- `data/processed/` — NumPy arrays (download once, then `python src/data.py` populates them)
- Any `.keras` or `.joblib` file above 50 MB (current files are all well below this limit)

---

## 📄 License

For academic / lab use only. UCI HAR dataset © UCI Machine Learning Repository.
