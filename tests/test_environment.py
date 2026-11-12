"""
tests/test_environment.py
Verifies that every required package imports cleanly and
the runtime-compat shim resolves the Windows DLL conflict.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_runtime_compat_imports():
    import src.runtime_compat  # noqa: F401


def test_tensorflow_imports():
    import src.runtime_compat  # noqa: F401
    import tensorflow as tf
    assert tf.__version__.startswith("2."), f"Unexpected TF version: {tf.__version__}"


def test_tensorflow_cpu_device():
    import src.runtime_compat  # noqa: F401
    import tensorflow as tf
    devices = tf.config.list_physical_devices("CPU")
    assert len(devices) >= 1, "No CPU device found by TensorFlow"


def test_numpy_imports():
    import numpy as np
    arr = np.array([1, 2, 3], dtype=np.float32)
    assert arr.sum() == 6.0


def test_scipy_imports():
    import scipy
    assert hasattr(scipy, "__version__")


def test_sklearn_imports():
    import sklearn
    assert hasattr(sklearn, "__version__")


def test_pandas_imports():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2]})
    assert len(df) == 2


def test_plotly_imports():
    import plotly
    assert hasattr(plotly, "__version__")


def test_matplotlib_imports():
    import matplotlib
    assert hasattr(matplotlib, "__version__")


def test_streamlit_imports():
    import streamlit
    assert hasattr(streamlit, "__version__")


def test_joblib_imports():
    import joblib
    assert hasattr(joblib, "__version__")
