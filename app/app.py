"""Streamlit interface for the team's UCI HAR models.

Run from the repository root with ``streamlit run app/app.py``.
"""

from __future__ import annotations

# -- Runtime compatibility shim -- MUST be the very first import ----------
# Patches google.protobuf.runtime_version (missing in protobuf<5) and
# pre-initialises TensorFlow's native DLL before OpenMP/OpenBLAS
# (loaded by NumPy/SciPy/Scikit-learn) can lock the Windows C-runtime.
import sys
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import src.runtime_compat  # noqa: E402  -- shim fires before any heavy import

# -- Standard / third-party imports ----------------------------------------
import hashlib
import io
import json
import re
import time
import csv
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

ROOT = _APP_ROOT

from src.data import LABELS as DATA_LABELS  # noqa: E402
from src.data import prepare_data  # noqa: E402
from src.predict import load_models  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
WINDOW_SHAPE = (128, 6)
SAMPLE_RATE_HZ = 50
MODEL_NAMES = ("CNN", "LSTM", "SVM", "RF", "KNN")
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_COLORS = ("#4CC9F0", "#F72585", "#FFC857", "#80ED99", "#B388EB", "#FF9F1C")
CHANNEL_NAMES = (
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
)
CHANNEL_LABELS = (
    "Accelerometer X",
    "Accelerometer Y",
    "Accelerometer Z",
    "Gyroscope X",
    "Gyroscope Y",
    "Gyroscope Z",
)
ACTIVITY_EMOJI = {
    "WALKING": "🚶",
    "WALKING_UPSTAIRS": "🪜",
    "WALKING_DOWNSTAIRS": "🪜",
    "SITTING": "🪑",
    "STANDING": "🧍",
    "LAYING": "🛌",
}

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # Plotly is optional in the current shared requirements file.
    px = None
    go = None


st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Load (or prepare) the official UCI HAR test windows and labels."""
    arrays = prepare_data()
    X_test = np.asarray(arrays[1], dtype=np.float32)
    y_test = np.asarray(arrays[3], dtype=np.int64).reshape(-1)
    if X_test.ndim != 3 or X_test.shape[1:] != WINDOW_SHAPE:
        raise ValueError(f"Expected test windows shaped (N, 128, 6); got {X_test.shape}.")
    if X_test.shape[0] != y_test.size:
        raise ValueError("The UCI HAR test windows and labels have different lengths.")
    return X_test, y_test


@st.cache_resource(show_spinner=False)
def _load_models_safely() -> tuple[dict[str, Any], str | None]:
    """Use Aryan's loader, then isolate failures so one artifact cannot block all models."""
    try:
        return load_models(verbose=False), None
    except Exception as exc:  # Show runtime/version/model errors instead of crashing the app.
        errors = [f"Unified loader: {type(exc).__name__}: {exc}"]
        recovered: dict[str, Any] = {}
        try:
            import joblib

            for name in MODEL_NAMES:
                suffix = ".keras" if name in ("CNN", "LSTM") else ".joblib"
                model_path = MODEL_DIR / f"{name.lower()}{suffix}"
                if not model_path.is_file():
                    continue
                try:
                    if name in ("CNN", "LSTM"):
                        from tensorflow import keras

                        recovered[name] = keras.models.load_model(str(model_path), compile=False)
                    else:
                        recovered[name] = joblib.load(model_path)
                except Exception as model_error:
                    if name == "LSTM":
                        try:
                            recovered[name] = _load_lstm_from_weights(model_path)
                        except Exception as weights_error:
                            errors.append(
                                "LSTM whole-model load: "
                                f"{type(model_error).__name__}: {model_error}; "
                                "weights-only fallback: "
                                f"{type(weights_error).__name__}: {weights_error}"
                            )
                    else:
                        errors.append(f"{name}: {type(model_error).__name__}: {model_error}")
        except Exception as recovery_error:
            errors.append(f"Per-model recovery: {type(recovery_error).__name__}: {recovery_error}")
        return recovered, " | ".join(errors)


def _load_lstm_from_weights(model_path: Path) -> Any:
    """Rebuild Indraneel's selected architecture and load weights from .keras.

    This path avoids deserializing newer Keras layer config fields (for example
    ``quantization_config``) when the local Keras minor version is older than
    the Colab version that saved the model.
    """
    from src.train_lstm import build_lstm

    units, dropout, bidirectional = 64, 0.3, False
    tuning_path = RESULTS_DIR / "lstm_tuning.csv"
    if tuning_path.is_file():
        with tuning_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if str(row.get("selected", "")).strip().lower() == "true":
                    units = int(row["hidden_units"])
                    dropout = float(row["dropout_rate"])
                    bidirectional = str(row.get("bidirectional", "false")).strip().lower() == "true"
                    break

    model = build_lstm(
        input_shape=WINDOW_SHAPE,
        num_classes=len(DATA_LABELS),
        hidden_units=units,
        dropout_rate=dropout,
        bidirectional=bidirectional,
    )
    model.load_weights(str(model_path))
    return model


@st.cache_resource(show_spinner=False)
def _load_channel_scaler() -> Any:
    """Load the train-fitted scaler, preparing the dataset if it is absent."""
    scaler_path = DATA_DIR / "scaler.joblib"
    if not scaler_path.is_file():
        _load_test_data()
    if not scaler_path.is_file():
        raise FileNotFoundError(f"The shared channel scaler was not created at {scaler_path}.")
    import joblib

    return joblib.load(scaler_path)


@st.cache_resource(show_spinner=False)
def _load_knn_feature_scaler() -> Any:
    """Load Akhil's train-fitted feature scaler, generating it if absent locally."""
    scaler_path = DATA_DIR / "feature_scaler.joblib"
    if not scaler_path.is_file():
        _load_test_data()
        from src.features import main as prepare_feature_artifacts

        prepare_feature_artifacts()
    if not scaler_path.is_file():
        raise FileNotFoundError(f"The KNN feature scaler was not created at {scaler_path}.")
    import joblib

    return joblib.load(scaler_path)


@st.cache_data(show_spinner="Preparing a sample sensor CSV...")
def _sample_csv_bytes() -> bytes:
    """Return one real UCI HAR window in the raw CSV format accepted by the app."""
    X_test, _ = _load_test_data()
    scaler = _load_channel_scaler()
    raw_window = scaler.inverse_transform(X_test[0].reshape(-1, 6)).reshape(WINDOW_SHAPE)
    sample = pd.DataFrame(raw_window, columns=CHANNEL_NAMES)
    return sample.to_csv(index=False, float_format="%.8f").encode("utf-8")


def _activity_name(index: int) -> str:
    if 0 <= int(index) < len(DATA_LABELS):
        return str(DATA_LABELS[int(index)])
    return f"Class {index}"


def _class_index(value: Any) -> int | None:
    """Map a sklearn class value back to the shared zero-based label order."""
    try:
        index = int(value)
        return index if 0 <= index < len(DATA_LABELS) else None
    except (TypeError, ValueError):
        normalized = str(value).strip().upper().replace(" ", "_")
        try:
            return list(DATA_LABELS).index(normalized)
        except ValueError:
            return None


def _align_class_columns(scores: np.ndarray, classes: Any = None) -> np.ndarray:
    """Place a model's output columns into the shared activity order."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D prediction array, got shape {values.shape}.")

    class_values = list(classes) if classes is not None else list(range(values.shape[1]))
    aligned = np.zeros((values.shape[0], len(DATA_LABELS)), dtype=np.float64)
    for column, class_value in enumerate(class_values[: values.shape[1]]):
        index = _class_index(class_value)
        if index is not None:
            aligned[:, index] = values[:, column]
    return aligned


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    values = np.exp(np.clip(scores, -60.0, 60.0))
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.full_like(values, 1.0 / values.shape[1]), where=totals > 0)


def _model_input_matrix(model_name: str, batch: np.ndarray, model: Any | None = None) -> np.ndarray:
    """Adapt standardized windows to each sklearn model's trained feature shape."""
    if model_name == "KNN":
        from src.features import extract_batch

        raw_features = extract_batch(batch)
        inputs = _load_knn_feature_scaler().transform(raw_features).astype(np.float32)
    else:
        inputs = batch.reshape(batch.shape[0], -1)

    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and inputs.shape[1] != int(expected_features):
        raise ValueError(
            f"{model_name} model expects {int(expected_features)} features, "
            f"but the app prepared {inputs.shape[1]}."
        )
    return inputs


def _probability_matrix(
    model_name: str,
    model: Any,
    windows: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return per-class scores, adapting the current predict.py model contract.

    Aryan's current API accepts already standardized windows and returns class
    indices. This app uses its shared loader and the same input shapes, then
    reads probability outputs directly so the UI can display confidence bars.
    """
    batch = np.asarray(windows, dtype=np.float32)
    if batch.ndim == 2:
        batch = batch[np.newaxis, ...]
    if batch.ndim != 3 or batch.shape[1:] != WINDOW_SHAPE:
        raise ValueError(f"Expected windows shaped (N, 128, 6); got {batch.shape}.")

    if model_name in ("CNN", "LSTM"):
        raw = np.asarray(model.predict(batch, verbose=0), dtype=np.float64)
        probabilities = _align_class_columns(raw)
        method = "Neural-network softmax output"
        is_probability = True
    else:
        model_inputs = _model_input_matrix(model_name, batch, model)
        classes = getattr(model, "classes_", None)
        if hasattr(model, "predict_proba"):
            try:
                raw = np.asarray(model.predict_proba(model_inputs), dtype=np.float64)
                probabilities = _align_class_columns(raw, classes)
                method = "Classifier predict_proba output"
                is_probability = True
            except (AttributeError, NotImplementedError) as exc:
                # SVC exposes predict_proba as a method even when it was fitted
                # with probability=False; in that case sklearn raises here.
                if not hasattr(model, "decision_function"):
                    raise exc
                raw = np.asarray(model.decision_function(model_inputs), dtype=np.float64)
                if raw.ndim == 1 and classes is not None and len(classes) == 2:
                    raw = np.column_stack((-raw, raw))
                probabilities = _align_class_columns(raw, classes)
                method = "Estimated from decision scores; not calibrated probabilities (SVC probability=False)"
                is_probability = False
        elif hasattr(model, "decision_function"):
            raw = np.asarray(model.decision_function(model_inputs), dtype=np.float64)
            if raw.ndim == 1 and classes is not None and len(classes) == 2:
                raw = np.column_stack((-raw, raw))
            probabilities = _align_class_columns(raw, classes)
            method = "Estimated from decision scores; not calibrated probabilities"
            is_probability = False
        else:
            raw_labels = np.asarray(model.predict(model_inputs)).reshape(-1)
            probabilities = np.zeros((batch.shape[0], len(DATA_LABELS)), dtype=np.float64)
            for row, label in enumerate(raw_labels):
                index = _class_index(label)
                if index is not None:
                    probabilities[row, index] = 1.0
            method = "Predicted class only; this model exposes no probability API"
            is_probability = True

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{model_name} returned non-finite scores.")
    if is_probability:
        probabilities = np.clip(probabilities, 0.0, None)
        totals = probabilities.sum(axis=1, keepdims=True)
        probabilities = np.divide(
            probabilities,
            totals,
            out=np.full_like(probabilities, 1.0 / len(DATA_LABELS)),
            where=totals > 0,
        )
    else:
        probabilities = _softmax(probabilities)
    return probabilities, method


def _class_predictions(
    model_name: str,
    model: Any,
    windows: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Use the estimator's class prediction, even when its scores are approximate."""
    if model_name in ("CNN", "LSTM", "KNN"):
        return np.argmax(probabilities, axis=1).astype(np.int64)
    batch = np.asarray(windows)
    if batch.ndim == 2:
        batch = batch[np.newaxis, ...]
    model_inputs = _model_input_matrix(model_name, batch, model)
    raw_labels = np.asarray(model.predict(model_inputs)).reshape(-1)
    indices = [_class_index(value) for value in raw_labels]
    return np.asarray(
        [index if index is not None else int(np.argmax(probabilities[row])) for row, index in enumerate(indices)],
        dtype=np.int64,
    )


def _dummy_probabilities(window: np.ndarray) -> np.ndarray:
    """Make a repeatable demo-only distribution when a requested model is absent."""
    digest = hashlib.sha256(np.asarray(window, dtype=np.float32).tobytes()).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed).dirichlet(np.ones(len(DATA_LABELS)))


def _predict_one(
    model_name: str,
    standardized_window: np.ndarray,
) -> tuple[np.ndarray, int, float, str, str | None]:
    models, load_error = _load_models_safely()
    model = models.get(model_name)
    if model is None:
        probabilities = _dummy_probabilities(standardized_window)
        reason = f"{model_name} model is unavailable; showing demo-only random probabilities."
        if load_error:
            reason += f" Model loading also reported: {load_error}"
        return probabilities, int(np.argmax(probabilities)), 0.0, "Demo-only dummy output", reason

    started = time.perf_counter()
    probabilities, method = _probability_matrix(model_name, model, standardized_window)
    predicted_index = int(_class_predictions(model_name, model, standardized_window, probabilities)[0])
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return probabilities[0], predicted_index, elapsed_ms, method, None


def _canonical_column(name: Any) -> str | None:
    key = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    aliases = {
        "total_acc_x": "total_acc_x",
        "total_acc_y": "total_acc_y",
        "total_acc_z": "total_acc_z",
        "acc_x": "total_acc_x",
        "acc_y": "total_acc_y",
        "acc_z": "total_acc_z",
        "accel_x": "total_acc_x",
        "accel_y": "total_acc_y",
        "accel_z": "total_acc_z",
        "accelerometer_x": "total_acc_x",
        "accelerometer_y": "total_acc_y",
        "accelerometer_z": "total_acc_z",
        "body_gyro_x": "body_gyro_x",
        "body_gyro_y": "body_gyro_y",
        "body_gyro_z": "body_gyro_z",
        "gyro_x": "body_gyro_x",
        "gyro_y": "body_gyro_y",
        "gyro_z": "body_gyro_z",
        "gyroscope_x": "body_gyro_x",
        "gyroscope_y": "body_gyro_y",
        "gyroscope_z": "body_gyro_z",
    }
    return aliases.get(key)


def _read_uploaded_window(file_bytes: bytes) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Parse six raw sensor columns and return display and standardized windows."""
    try:
        header = pd.read_csv(io.BytesIO(file_bytes), nrows=0)
        header_is_numeric = all(
            _is_number(column) for column in header.columns
        )
        frame = pd.read_csv(io.BytesIO(file_bytes), header=None if header_is_numeric else 0)
    except Exception as exc:
        raise ValueError(f"Could not read this CSV: {exc}") from exc

    if frame.shape[1] != 6:
        raise ValueError(f"Expected exactly 6 sensor columns; found {frame.shape[1]}.")
    if len(frame) < WINDOW_SHAPE[0]:
        raise ValueError(f"Expected at least 128 rows; found {len(frame)}.")

    note = None
    if header_is_numeric:
        frame.columns = list(CHANNEL_NAMES)
    else:
        canonical = [_canonical_column(column) for column in frame.columns]
        if all(column is not None for column in canonical) and len(set(canonical)) == 6:
            frame = frame.iloc[:, [canonical.index(channel) for channel in CHANNEL_NAMES]]
            frame.columns = list(CHANNEL_NAMES)
        else:
            frame.columns = list(CHANNEL_NAMES)
            note = "Column names were not recognized; interpreted the six columns in the documented acc XYZ, gyro XYZ order."

    numeric = frame.iloc[: WINDOW_SHAPE[0]].apply(pd.to_numeric, errors="coerce")
    invalid = np.argwhere(numeric.isna().to_numpy())
    if invalid.size:
        row, column = invalid[0]
        raise ValueError(
            f"Sensor values must be numeric. Invalid value at data row {row + 1}, column '{numeric.columns[column]}'."
        )
    display_window = numeric.to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(display_window)):
        raise ValueError("Sensor values must be finite numbers (no NaN or infinity).")

    scaler = _load_channel_scaler()
    standardized = scaler.transform(display_window).astype(np.float32)
    if len(frame) > WINDOW_SHAPE[0]:
        extra_note = f"Using the first 128 rows of {len(frame)}; additional rows were ignored."
        note = f"{note} {extra_note}".strip() if note else extra_note
    return display_window, standardized, note


def _is_number(value: Any) -> bool:
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _render_signal_plots(window: np.ndarray) -> None:
    time_seconds = np.arange(WINDOW_SHAPE[0], dtype=np.float32) / SAMPLE_RATE_HZ
    frame = pd.DataFrame(window, columns=CHANNEL_LABELS)
    frame.insert(0, "Time (seconds)", time_seconds)
    left, right = st.columns(2)
    with left:
        st.subheader("Accelerometer")
        if px is not None:
            fig = px.line(
                frame,
                x="Time (seconds)",
                y=list(CHANNEL_LABELS[:3]),
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=PLOTLY_COLORS,
            )
            fig.update_layout(legend_title_text="Axis", height=340, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(frame.set_index("Time (seconds)").iloc[:, :3], height=340)
    with right:
        st.subheader("Gyroscope")
        if px is not None:
            fig = px.line(
                frame,
                x="Time (seconds)",
                y=list(CHANNEL_LABELS[3:]),
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=PLOTLY_COLORS,
            )
            fig.update_layout(legend_title_text="Axis", height=340, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(frame.set_index("Time (seconds)").iloc[:, 3:], height=340)


def _render_probability_chart(probabilities: np.ndarray) -> None:
    chart_data = pd.DataFrame(
        {"Activity": list(DATA_LABELS), "Probability": np.asarray(probabilities, dtype=float)}
    ).sort_values("Probability", ascending=True)
    if px is not None:
        fig = px.bar(
            chart_data,
            x="Probability",
            y="Activity",
            orientation="h",
            range_x=[0, 1],
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=PLOTLY_COLORS,
        )
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10), yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(chart_data.set_index("Activity"), height=330)


def _compare_models_on_window(window: np.ndarray) -> list[dict[str, Any]]:
    """Build a side-by-side result table for all five model choices."""
    display_names = {"RF": "Random Forest"}
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        try:
            probabilities, predicted_index, latency_ms, method, dummy_message = _predict_one(
                model_name, window
            )
            rows.append(
                {
                    "Model": display_names.get(model_name, model_name),
                    "Predicted activity": _activity_name(predicted_index).replace("_", " ").title(),
                    "Predicted-class score": f"{float(probabilities[predicted_index]):.1%}",
                    "Latency (ms)": round(latency_ms, 2),
                    "Score source": method,
                    "Status": "Demo fallback" if dummy_message else "Ready",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "Model": display_names.get(model_name, model_name),
                    "Predicted activity": "—",
                    "Predicted-class score": "—",
                    "Latency (ms)": None,
                    "Score source": "—",
                    "Status": f"Error: {exc}",
                }
            )
    return rows


def _page_overview() -> None:
    st.header("Overview")
    st.write(
        "This project compares five machine-learning models that classify short "
        "accelerometer and gyroscope windows from the UCI HAR smartphone dataset. "
        "Each window contains 128 samples across six sensor channels."
    )
    st.subheader("Activities")
    columns = st.columns(3)
    for index, label in enumerate(DATA_LABELS):
        with columns[index % 3]:
            emoji = ACTIVITY_EMOJI.get(label, "📍")
            st.markdown(f"### {emoji} {label.replace('_', ' ').title()}")
    st.subheader("Models")
    st.write("CNN · LSTM · SVM · Random Forest · KNN")
    st.info(
        "Live predictions use the saved train-fitted channel scaler. "
        "Uploaded sensor CSVs need six numeric columns in accelerometer X/Y/Z, "
        "gyroscope X/Y/Z order and at least 128 rows."
    )


def _page_live_demo() -> None:
    st.header("Live Demo")
    selected_model = st.selectbox("Choose a model", MODEL_NAMES, key="live_model")
    source = st.radio(
        "Choose an input",
        ("Random test window", "Upload sensor CSV"),
        horizontal=True,
        key="live_source",
    )

    true_label: str | None = None
    sample_note: str | None = None
    comparison_input_key: str
    if source == "Random test window":
        try:
            with st.spinner("Preparing the UCI HAR test split if needed..."):
                X_test, y_test = _load_test_data()
        except Exception as exc:
            st.error(f"Could not load the UCI HAR test windows: {exc}")
            return
        if "live_sample_index" not in st.session_state:
            st.session_state.live_sample_index = int(np.random.default_rng().integers(len(X_test)))
        if st.button("Pick another random window"):
            st.session_state.live_sample_index = int(np.random.default_rng().integers(len(X_test)))
        sample_index = int(st.session_state.live_sample_index) % len(X_test)
        model_window = X_test[sample_index]
        display_window = model_window
        true_label = _activity_name(int(y_test[sample_index]))
        sample_note = f"UCI HAR test window {sample_index + 1} of {len(X_test)}."
        comparison_input_key = f"test-window:{sample_index}"
    else:
        try:
            sample_csv = _sample_csv_bytes()
            st.download_button(
                "Download sample sensor CSV",
                data=sample_csv,
                file_name="sample_har_sensor_window.csv",
                mime="text/csv",
                help="A 128-row UCI HAR window in the accepted six-channel CSV format.",
                key="download_sample_sensor_csv",
            )
        except Exception as exc:
            st.warning(f"Could not prepare the sample CSV right now: {exc}")
        uploaded = st.file_uploader("Upload a CSV with six sensor columns", type=("csv",))
        st.caption(
            "Accepted headers include total_acc_x/y/z and body_gyro_x/y/z, "
            "or acc_x/y/z and gyro_x/y/z. Headerless files are read in acc XYZ, gyro XYZ order."
        )
        if uploaded is None:
            st.info("Upload at least 128 rows to run a prediction.")
            return
        try:
            file_bytes = uploaded.getvalue()
            display_window, model_window, sample_note = _read_uploaded_window(file_bytes)
            comparison_input_key = f"uploaded-csv:{hashlib.sha256(file_bytes).hexdigest()}"
        except Exception as exc:
            st.error(f"Invalid sensor CSV: {exc}")
            st.caption("Use exactly six numeric sensor columns and at least 128 data rows.")
            return

    if st.session_state.get("all_model_comparison_input") != comparison_input_key:
        st.session_state.all_model_comparison_input = comparison_input_key
        st.session_state.pop("all_model_comparison_rows", None)

    if sample_note:
        st.caption(sample_note)
    if sample_note and "not recognized" in sample_note:
        st.warning(sample_note)
    _render_signal_plots(display_window)

    try:
        with st.spinner(f"Running {selected_model} prediction..."):
            probabilities, predicted_index, latency_ms, probability_method, dummy_message = _predict_one(
                selected_model, model_window
            )
    except Exception as exc:
        st.error(f"Prediction failed for {selected_model}: {exc}")
        return

    if dummy_message:
        st.warning(dummy_message)
    if "not calibrated" in probability_method or "no probability API" in probability_method:
        st.info(probability_method)

    predicted_label = _activity_name(predicted_index)
    emoji = ACTIVITY_EMOJI.get(predicted_label, "📍")
    left, right = st.columns(2)
    with left:
        st.markdown(f"## {emoji} {predicted_label.replace('_', ' ').title()}")
        st.caption(f"Predicted activity · {selected_model}")
        st.metric("Prediction latency", f"{latency_ms:.2f} ms")
    with right:
        if true_label is not None:
            st.metric("True activity", f"{ACTIVITY_EMOJI.get(true_label, '📍')} {true_label.replace('_', ' ').title()}")
        confidence = float(probabilities[predicted_index])
        st.metric("Predicted-class score", f"{confidence:.1%}")
    st.subheader("Compare all five models on this window")
    if st.button("Compare all 5 models", key="compare_all_models"):
        with st.spinner("Running all five models on the same sensor window..."):
            st.session_state.all_model_comparison_rows = _compare_models_on_window(model_window)
            st.session_state.all_model_comparison_input = comparison_input_key
    if (
        st.session_state.get("all_model_comparison_input") == comparison_input_key
        and st.session_state.get("all_model_comparison_rows")
    ):
        comparison_rows = st.session_state.all_model_comparison_rows
        st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)
        if any(row["Status"] == "Demo fallback" for row in comparison_rows):
            st.warning("Rows marked Demo fallback are placeholder outputs, not model predictions.")
        if any(str(row["Status"]).startswith("Error:") for row in comparison_rows):
            st.warning("One or more models could not run; see that model's status in the table.")
    st.subheader("Activity scores")
    _render_probability_chart(probabilities)
    st.caption(f"Score source: {probability_method}.")


@st.cache_data(show_spinner=False)
def _predict_test_set(model_name: str) -> tuple[np.ndarray, np.ndarray, float, str, str | None]:
    """Cache all test-set predictions for the selected model during playback."""
    X_test, y_test = _load_test_data()
    models, load_error = _load_models_safely()
    model = models.get(model_name)
    if model is None:
        probability_rows = np.vstack([_dummy_probabilities(window) for window in X_test])
        note = "Demo-only dummy predictions; the requested model could not be loaded."
        if load_error:
            note += f" Loader error: {load_error}"
        return np.argmax(probability_rows, axis=1), y_test, 0.0, "Dummy output", note

    started = time.perf_counter()
    probability_rows, method = _probability_matrix(model_name, model, X_test)
    predictions = _class_predictions(model_name, model, X_test, probability_rows)
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(X_test))
    return predictions, y_test, elapsed_ms, method, None


def _set_stream_playing(value: bool) -> None:
    st.session_state.stream_playing = value


def _reset_stream() -> None:
    st.session_state.stream_index = int(st.session_state.get("stream_start_index", 0))
    st.session_state.stream_playing = False


def _randomize_stream_start() -> None:
    total = max(1, int(st.session_state.get("stream_total", 1)))
    start_index = int(np.random.default_rng().integers(total))
    st.session_state.stream_start_index = start_index
    st.session_state.stream_index = start_index
    st.session_state.stream_playing = False


def _step_stream(amount: int) -> None:
    st.session_state.stream_index = max(0, int(st.session_state.get("stream_index", 0)) + amount)


def _render_stream_timeline(
    predicted: np.ndarray,
    truth: np.ndarray,
    current_index: int,
    start_index: int,
    history_size: int = 100,
) -> None:
    first_index = max(start_index, current_index - history_size + 1)
    indices = np.arange(first_index, current_index + 1)
    frame = pd.DataFrame(
        {
            "Window": indices + 1,
            "True activity": truth[indices],
            "Predicted activity": predicted[indices],
        }
    )
    if go is not None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=frame["Window"], y=frame["True activity"], name="True",
                mode="lines+markers", line_shape="hv",
                line={"color": PLOTLY_COLORS[0], "width": 4},
                marker={"symbol": "circle", "size": 8},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["Window"], y=frame["Predicted activity"], name="Predicted",
                mode="lines+markers", line_shape="hv",
                line={"color": PLOTLY_COLORS[1], "width": 2, "dash": "dash"},
                marker={"symbol": "x", "size": 9},
            )
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(DATA_LABELS))),
            ticktext=list(DATA_LABELS),
            range=[-0.5, len(DATA_LABELS) - 0.5],
        )
        fig.update_layout(
            xaxis_title="Test window",
            yaxis_title="Activity",
            template=PLOTLY_TEMPLATE,
            colorway=list(PLOTLY_COLORS),
            height=400,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(frame.set_index("Window"), height=400)
        st.caption(f"Timeline shows up to the latest {history_size} windows; y-values use the shared activity order.")


def _page_stream_simulation() -> None:
    st.header("Stream Simulation")
    st.write("Play consecutive official test windows and compare predicted activities with their true labels.")
    selected_model = st.selectbox("Model for the stream", MODEL_NAMES, key="stream_model")
    speed = st.slider("Playback interval (seconds per window)", 0.2, 2.0, 0.8, 0.1)

    try:
        with st.spinner(f"Preparing test predictions for {selected_model}..."):
            predicted, truth, latency_ms, probability_method, dummy_message = _predict_test_set(selected_model)
    except Exception as exc:
        st.error(f"Could not prepare the activity stream: {exc}")
        return

    if dummy_message:
        st.warning(dummy_message)
    if len(predicted) == 0:
        st.error("The test split is empty.")
        return

    st.session_state.stream_total = len(predicted)
    if "stream_index" not in st.session_state:
        st.session_state.stream_index = int(np.random.default_rng().integers(len(predicted)))
    if "stream_start_index" not in st.session_state:
        st.session_state.stream_start_index = int(st.session_state.stream_index)
    if "stream_playing" not in st.session_state:
        st.session_state.stream_playing = False
    current_index = min(int(st.session_state.stream_index), len(predicted) - 1)
    st.session_state.stream_index = current_index
    start_index = min(int(st.session_state.stream_start_index), current_index)
    st.session_state.stream_start_index = start_index
    if current_index >= len(predicted) - 1:
        st.session_state.stream_playing = False

    controls = st.columns(5)
    with controls[0]:
        st.button(
            "⏸ Pause" if st.session_state.stream_playing else "▶ Play",
            on_click=_set_stream_playing,
            args=(not st.session_state.stream_playing,),
            use_container_width=True,
        )
    with controls[1]:
        st.button("⟲ Reset", on_click=_reset_stream, use_container_width=True)
    with controls[2]:
        st.button("Random start", on_click=_randomize_stream_start, use_container_width=True)
    with controls[3]:
        st.button(
            "◀ Previous",
            on_click=_step_stream,
            args=(-1,),
            disabled=st.session_state.stream_playing or current_index <= start_index,
            use_container_width=True,
        )
    with controls[4]:
        st.button(
            "Next ▶",
            on_click=_step_stream,
            args=(1,),
            disabled=st.session_state.stream_playing or current_index >= len(predicted) - 1,
            use_container_width=True,
        )

    true_name = _activity_name(int(truth[current_index]))
    predicted_name = _activity_name(int(predicted[current_index]))
    metrics = st.columns(3)
    metrics[0].metric("Window", f"{current_index + 1} / {len(predicted)}")
    metrics[1].metric("True activity", f"{ACTIVITY_EMOJI.get(true_name, '📍')} {true_name.replace('_', ' ').title()}")
    metrics[2].metric("Predicted activity", f"{ACTIVITY_EMOJI.get(predicted_name, '📍')} {predicted_name.replace('_', ' ').title()}")
    st.caption(f"Average inference time: {latency_ms:.2f} ms/window · score source: {probability_method}.")
    st.subheader("Prediction timeline")
    _render_stream_timeline(predicted, truth, current_index, start_index)

    if st.session_state.stream_playing and current_index < len(predicted) - 1:
        time.sleep(float(speed))
        if st.session_state.get("stream_playing", False):
            st.session_state.stream_index = current_index + 1
            st.rerun()


def _metric_value(metrics: dict[str, Any], key: str, fallback: str | None = None) -> float | None:
    value = metrics.get(key)
    if value is None and fallback:
        value = metrics.get(fallback)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _read_comparison_data() -> tuple[pd.DataFrame, dict[str, dict[str, Any]], set[str]]:
    """Combine the shared comparison CSV with current per-model JSON files."""
    rows: dict[str, dict[str, Any]] = {}
    csv_models: set[str] = set()
    comparison_csv = RESULTS_DIR / "comparison.csv"
    if comparison_csv.is_file():
        try:
            source = pd.read_csv(comparison_csv)
            if "model" in source.columns:
                for _, record in source.iterrows():
                    name = str(record["model"]).strip().upper()
                    if not name:
                        continue
                    rows[name] = record.to_dict()
                    csv_models.add(name)
        except Exception as exc:
            st.warning(f"Could not read {comparison_csv.name}: {exc}")

    model_metrics: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS_DIR.glob("*_metrics.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            st.warning(f"Could not read {path.name}: {exc}")
            continue
        name = str(data.get("model") or path.name.removesuffix("_metrics.json")).strip().upper()
        if not name:
            continue
        model_metrics[name] = data
        row = rows.setdefault(name, {"model": name})
        row["model"] = name
        row["accuracy"] = _metric_value(data, "accuracy")
        row["precision_macro"] = _metric_value(data, "precision_macro")
        row["recall_macro"] = _metric_value(data, "recall_macro")
        row["f1_macro"] = _metric_value(data, "f1_macro", "macro_f1")
        row["precision_weighted"] = _metric_value(data, "precision_weighted")
        row["recall_weighted"] = _metric_value(data, "recall_weighted")
        row["f1_weighted"] = _metric_value(data, "f1_weighted")
        row["train_time_s"] = _metric_value(data, "train_time_s")
        row["infer_ms_per_sample"] = _metric_value(data, "infer_ms_per_sample")

    ordered_names = [name for name in MODEL_NAMES if name in rows]
    ordered_names.extend(name for name in rows if name not in ordered_names)
    table = pd.DataFrame([rows[name] for name in ordered_names])
    return table, model_metrics, csv_models


def _page_model_comparison() -> None:
    st.header("Model Comparison")
    st.write("Metrics come from the shared comparison CSV and each model's metrics JSON.")
    table, model_metrics, csv_models = _read_comparison_data()
    if table.empty:
        st.warning("No comparison metrics found. Train the models and add results/*_metrics.json first.")
        return

    json_models = set(model_metrics)
    missing_from_csv = sorted(json_models - csv_models)
    if missing_from_csv:
        st.info(
            "The comparison CSV is missing current JSON rows for "
            f"{', '.join(missing_from_csv)}. They are included from their metrics files; "
            "regenerate the shared CSV with `python src/compare.py` when the team is ready."
        )

    display_columns = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "train_time_s",
        "infer_ms_per_sample",
    ]
    for column in display_columns:
        if column not in table.columns:
            table[column] = np.nan
    st.dataframe(
        table[display_columns].rename(
            columns={
                "model": "Model",
                "accuracy": "Accuracy",
                "precision_macro": "Macro precision",
                "recall_macro": "Macro recall",
                "f1_macro": "Macro F1",
                "train_time_s": "Train time (s)",
                "infer_ms_per_sample": "Inference (ms/sample)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    score_data = table[["model", "accuracy", "f1_macro"]].copy()
    score_data = score_data.melt(id_vars="model", var_name="Metric", value_name="Score").dropna()
    st.subheader("Accuracy and macro-F1")
    if px is not None and not score_data.empty:
        fig = px.bar(
            score_data,
            x="model",
            y="Score",
            color="Metric",
            barmode="group",
            range_y=[0, 1],
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=PLOTLY_COLORS,
        )
        fig.update_layout(height=370, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Model")
        st.plotly_chart(fig, use_container_width=True)
    elif not score_data.empty:
        wide = table.set_index("model")[["accuracy", "f1_macro"]].rename(
            columns={"accuracy": "Accuracy", "f1_macro": "Macro F1"}
        )
        st.bar_chart(wide, height=370)

    confusion_models = [
        name for name in MODEL_NAMES
        if name in model_metrics and isinstance(model_metrics[name].get("confusion_matrix"), list)
    ]
    if not confusion_models:
        st.info("No confusion matrices are available in the model metrics JSON files.")
        return

    st.subheader("Confusion matrix")
    selected_confusion = st.selectbox("Choose a model", confusion_models, key="confusion_model")
    metrics = model_metrics[selected_confusion]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    raw_labels = metrics.get("labels", list(DATA_LABELS))
    if not isinstance(raw_labels, list):
        raw_labels = list(DATA_LABELS)
    if len(raw_labels) == matrix.shape[0] and all(str(value).isdigit() for value in raw_labels):
        display_labels = [
            _activity_name(int(value)) if _class_index(value) is not None else str(value)
            for value in raw_labels
        ]
    else:
        display_labels = [str(value) for value in raw_labels]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(display_labels):
        st.warning("This model's confusion matrix dimensions do not match its label list.")
        st.dataframe(pd.DataFrame(matrix), use_container_width=True)
    elif go is not None:
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=display_labels,
                y=display_labels,
                colorscale="Blues",
                colorbar={"title": "Windows"},
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title="Predicted activity",
            yaxis_title="True activity",
            template=PLOTLY_TEMPLATE,
            height=560,
            margin=dict(l=10, r=10, t=20, b=100),
        )
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(
            pd.DataFrame(matrix, index=display_labels, columns=display_labels),
            use_container_width=True,
        )

    if px is None:
        st.caption("Install Plotly to enable interactive charts; Streamlit chart fallbacks are shown in this runtime.")


def main() -> None:
    st.sidebar.title("Human Activity Recognition")
    st.sidebar.caption("Team: Akhil · Dhairya · Aryan · Indraneel")
    page = st.sidebar.radio(
        "Navigate",
        ("Overview", "Live Demo", "Stream Simulation", "Model Comparison"),
    )
    st.title("Human Activity Recognition")
    st.caption("Team: Akhil · Dhairya · Aryan · Indraneel")
    st.divider()
    if page != "Stream Simulation":
        st.session_state.stream_playing = False

    if px is None and page in ("Live Demo", "Stream Simulation", "Model Comparison"):
        st.sidebar.warning("Plotly is not installed; using Streamlit chart fallbacks.")

    if page == "Overview":
        _page_overview()
    elif page == "Live Demo":
        _page_live_demo()
    elif page == "Stream Simulation":
        _page_stream_simulation()
    else:
        _page_model_comparison()


if __name__ == "__main__":
    main()
