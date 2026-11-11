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
│   ├── compare.py    # Dhairya's — combine metrics into CSV and chart
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

## Dhairya: classical models and shared evaluation

Place `X_train.npy`, `X_test.npy`, `y_train.npy`, and `y_test.npy` in
`data/processed/`. Run from the project root:

```bash
python src/evaluate.py --train-classical
```

This tunes and trains the SVM and Random Forest, saves `models/svm.joblib` and
`models/rf.joblib`, and writes the matching metrics and best-parameter JSON
files. Random Forest sensor-channel importances are saved in
`results/rf_feature_importance.csv`. Other model training scripts can import
`evaluate_predictions` or `evaluate_and_save` from `src.evaluate` to write the
same metrics JSON contract.

After the team metrics files are present, generate the comparison outputs:

```bash
python src/compare.py
```

This writes `results/comparison.csv` and `results/comparison_metrics.png`.
The Dhairya report draft is in `reports/dhairya_section.md`; fill in measured
test scores after training completes.

## Large Files

Models or data files over ~50 MB are stored on Google Drive.  
📁 **Drive folder**: _[add link here]_

## Running the App

```bash
streamlit run app/app.py
```
