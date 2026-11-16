# Classical Models and Evaluation — Dhairya Dabi

## SVM and Random Forest

The classical baselines use the project’s subject-separated UCI HAR training and test split. Each 128-sample, six-channel window is flattened into a feature vector for scikit-learn. Hyperparameters are selected with three-fold subject-grouped cross-validation on the training split only; the held-out test split is used once for final reporting. The SVM uses a pipeline so `StandardScaler` is fitted separately inside each training fold, preventing validation data from influencing scaling. Its search compares a linear `LinearSVC` baseline (`C=0.1`) against RBF `SVC` settings (`C=1` or `10`, with `gamma="scale"`; it also tests `C=10, gamma=0.001`). The Random Forest uses 300 trees and searches `max_features` (`sqrt` or 0.5) and `min_samples_leaf` (1 or 2). The selected settings and cross-validation accuracy are recorded in `results/{svm,rf}_best_params.json`.

## Shared evaluation and feature importance

Both models are saved under `models/` and scored through the same evaluator used by the team. Each metrics JSON includes accuracy, macro and weighted precision/recall/F1, class labels, sample count, confusion matrix, training time, and per-sample inference time. Inference latency is measured by predicting one window at a time over 100 warmed samples; batch throughput is recorded separately. This consistent structure lets `src/compare.py` combine model results without manual reformatting. For the Random Forest, built-in feature importances are summed over the 128 time positions for each of the six sensor channels. The resulting channel ranking is saved to `results/rf_feature_importance.csv`; it describes model reliance, not causal importance.

## Results and comparison

Run `python src/evaluate.py --train-classical --models svm` to retrain only the SVM after `src/data.py` has prepared the processed arrays. Run without `--models svm` to train both classical models. Then run `python src/compare.py` after the model metrics are available. The comparison script writes `results/comparison.csv` and `results/comparison_metrics.png` for the leaderboard.

**Final test results:** the updated SVM achieved **89.18% accuracy** and **0.890 macro-F1**. It selected an RBF kernel with `C=10` and `gamma=0.001` (grouped CV accuracy **89.94%**). The Random Forest achieved **83.81% accuracy** and **0.836 macro-F1**, selecting `max_features=sqrt` and `min_samples_leaf=1` (grouped CV accuracy 95.61%). In the current five-model comparison, CNN has the highest test accuracy at **91.45%**. The updated SVM averaged **2.19 ms** per single-window prediction; Random Forest averaged **49.90 ms** under the same measurement procedure. Timings depend on the machine and run.

The SVM is trained with `probability=False`. The Streamlit app turns its decision scores into softmax-normalized display scores; these are useful for showing relative class confidence but are **not calibrated probabilities**. The app labels them as estimates.

## Exploratory Data Analysis

The official subject-separated UCI HAR dataset contains six activities with 128-sample windows from six accelerometer and gyroscope channels. The class-count plot shows the train/test distribution; the signal plot illustrates the different axis patterns in static and dynamic movements. The magnitude boxplot summarizes per-window acceleration magnitude after train-fitted standardization. These visualizations are descriptive; they do not imply causal feature importance.

![UCI HAR activity counts in the train and test splits](../results/eda/class_distribution.png)

![Example accelerometer and gyroscope traces by activity](../results/eda/signals_by_activity.png)

![Per-window standardized acceleration magnitude by activity](../results/eda/magnitude_boxplot.png)
