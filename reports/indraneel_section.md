# LSTM and Streamlit Application — Indraneel Hajarnis

## 1. LSTM Architecture

The model consumes UCI HAR windows shaped `(128, 6)`: 128 time samples across three accelerometer and three gyroscope channels. The selected configuration is a unidirectional two-layer LSTM with 64 hidden units and 0.5 dropout.

| Layer | Type | Specifications | Output Shape | Parameters |
|---|---|---|---|---:|
| Input | InputLayer | Six sensor channels | `(128, 6)` | 0 |
| Recurrent 1 | LSTM | 64 units, `return_sequences=True` | `(128, 64)` | 18,176 |
| Regularization 1 | Dropout | Rate 0.5 | `(128, 64)` | 0 |
| Recurrent 2 | LSTM | 64 units | `(64)` | 33,024 |
| Dense head | Dense + ReLU | 64 units | `(64)` | 4,160 |
| Regularization 2 | Dropout | Rate 0.5 | `(64)` | 0 |
| Output | Dense + Softmax | Six activity classes | `(6)` | 390 |

**Total parameters:** 55,750.

## 2. Training Setup and Validation Strategy

- **Dataset:** UCI HAR official split, 7,352 training windows and 2,947 test windows. The channel-wise `StandardScaler` is fit on training data only.
- **Validation:** `GroupShuffleSplit` holds out 15% of training subjects (17 fit subjects and 4 validation subjects). Subjects do not overlap. The official test split is reserved for one final evaluation of the selected configuration.
- **Optimization:** Adam with gradient clipping (`clipnorm=1.0`) and sparse categorical cross-entropy. The sweep uses seed 42, batch size 128, up to 20 epochs per configuration, early stopping with restored best weights, and a 30-minute total fit budget.
- **Tuning:** Five settings vary hidden size, dropout, and direction. The selected model is chosen by validation macro-F1. The complete sweep recorded 58.9 seconds of training time.
- **Runtime:** Training and tuning ran on a Google Colab GPU. The selected model's training time was 11.61 seconds.

## 3. Results and Curve Interpretation

| Configuration | Validation Macro-F1 |
|---|---:|
| Baseline: 64 units, dropout 0.3 | 93.42% |
| Unidirectional: 32 units, dropout 0.3 | 94.63% |
| Unidirectional: 64 units, dropout 0.2 | 94.96% |
| **Unidirectional: 64 units, dropout 0.5 (selected)** | **95.23%** |
| Bidirectional: 32 units, dropout 0.3 | 91.17% |

The selected model improved validation macro-F1 by **1.82 percentage points** over the baseline. On the official test set it achieved **87.04% accuracy**, **86.85% macro-F1**, 86.99% macro precision, and 87.10% macro recall. Inference averaged **1.40 ms per sample** across 2,947 windows.

The selected run's training accuracy rises as training loss falls, while validation loss reaches its minimum before the final epoch and then fluctuates. Early stopping restores the best validation-loss weights. The largest test confusions were standing predicted as sitting (106 windows), sitting predicted as standing (81), and walking predicted as walking downstairs (59). The results reflect the difficulty of separating similar postures and movement patterns from short inertial windows.

## 4. Streamlit Application and Integration

The app provides four pages: **Overview** describes the task and six activities; **Live Demo** predicts from a random labeled test window or a six-channel CSV with at least 128 rows; **Stream Simulation** steps or plays through consecutive test windows and plots true versus predicted activity; and **Model Comparison** displays shared metrics, score charts, timing, and selectable confusion matrices. The live page also plots accelerometer and gyroscope signals, provides a sample CSV download, and can compare all five models on one window.

The app uses the team's model loader and adds local input adapters: raw CSV data is transformed with the saved channel scaler, while KNN uses Akhil's 85-feature extractor and train-fitted feature scaler. Model and scaler loading are cached. Missing models and per-model errors are labeled in the interface rather than hidden.

## 5. Limitations

UCI HAR contains six activities and excludes running and jogging. Stream Simulation replays the test set rather than a live sensor feed. If SVM was fitted with `probability=False`, its displayed decision-score estimates are not calibrated probabilities. The Colab model was saved with TensorFlow 2.21.0/Keras 3.13.2, while shared requirements pin TensorFlow 2.16.1; the team should align versions before the clean-environment demo. Plotly is optional in the current requirements, so Streamlit chart fallbacks are available when it is not installed.
