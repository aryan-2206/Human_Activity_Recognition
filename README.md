# Human Activity Recognition

A collaborative ML project comparing CNN, LSTM, SVM, Random Forest, and KNN models on the UCI HAR dataset.

## Team & Branches

| Person | Branch | Runs On | Produces |
|---|---|---|---|
| Akhil | `akhil-data` | Laptop / Colab | `data/processed/*.npy`, `models/knn.joblib`, `results/knn_*` |
| Dhairya | `dhairya-classical` | Laptop | `src/evaluate.py`, `models/svm.joblib`, `models/rf.joblib`, `results/*_metrics.json` |
| **Aryan** | **`aryan-cnn`** | **Colab (GPU) + Laptop** | **`models/cnn.keras`, `results/cnn_*`, `src/predict.py`** |
| Indraneel | `indraneel-app` | Colab (GPU) + Laptop | `models/lstm.keras`, `results/lstm_*`, `app/app.py` |

## Project Structure

```
Human_Activity_Recognition/
├── data/
│   ├── raw/          # Original UCI HAR dataset (not in git — use Drive)
│   └── processed/    # Akhil's .npy files (not in git — use Drive)
├── models/           # Saved model files (.keras, .joblib)
├── results/          # Metrics JSON files per model
├── src/
│   ├── evaluate.py   # Dhairya's — shared evaluation function
│   └── predict.py    # Aryan's — loads all 5 models, returns predictions
├── app/
│   └── app.py        # Indraneel's Streamlit app
├── notebooks/        # EDA and training notebooks
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Large Files

Models or data files over ~50 MB are stored on Google Drive.  
📁 **Drive folder**: _[add link here]_

## Running the App

```bash
streamlit run app/app.py
```
