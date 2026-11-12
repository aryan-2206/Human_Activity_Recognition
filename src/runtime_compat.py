"""Runtime compatibility and stabilization layer for Human Activity Recognition.

Ensures reliable operation across Windows, Python 3.12, TensorFlow, MediaPipe,
and Scikit-learn by resolving C-runtime DLL initialization order and Protobuf
version shims.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

# 1. Configure safe, isolated environment variables
_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _ROOT / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = _CACHE_DIR / "matplotlib"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

if "KERAS_HOME" not in os.environ:
    _keras_dir = _ROOT / ".keras"
    _keras_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KERAS_HOME"] = str(_keras_dir)

if "TF_ENABLE_ONEDNN_OPTS" not in os.environ:
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# 2. Protobuf runtime_version compatibility shim
# MediaPipe 0.10.x specifies protobuf<5 (e.g. 4.25.x), while TensorFlow 2.20
# generated code expects google.protobuf.runtime_version (introduced in 5.26+).
# This shim allows both libraries to coexist without conflicts.
def _patch_protobuf_runtime_version() -> None:
    if "google.protobuf.runtime_version" not in sys.modules:
        try:
            import google.protobuf.runtime_version  # noqa: F401
        except (ImportError, AttributeError):
            rv = types.ModuleType("google.protobuf.runtime_version")
            rv.Domain = types.SimpleNamespace(PUBLIC=1, INTERNAL=2)
            rv.Version = lambda *args, **kwargs: None
            rv.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
            sys.modules["google.protobuf.runtime_version"] = rv
            try:
                import google.protobuf

                google.protobuf.runtime_version = rv
            except Exception:
                pass


_patch_protobuf_runtime_version()


# 3. Windows C-runtime DLL initialization order
# On Windows, Scipy/Scikit-learn initialises an OpenMP / OpenBLAS thread runtime.
# If TensorFlow's native _pywrap_tensorflow_internal is loaded afterwards,
# Windows throws 'A dynamic link library (DLL) initialization routine failed'.
# Pre-initializing the TensorFlow native wrapper before Scipy/Sklearn eliminates
# this conflict completely.
def _preinit_tensorflow_runtime() -> bool:
    if sys.platform == "win32":
        try:
            import importlib.machinery
            import importlib.util

            # Locate the native extension directly
            import tensorflow  # noqa: F401

            return True
        except Exception:
            try:
                from tensorflow.python import _pywrap_tensorflow_internal  # noqa: F401

                return True
            except Exception:
                return False
    return True


_preinit_tensorflow_runtime()


def init_runtime() -> None:
    """Explicit initializer callable from application entry points."""
    _patch_protobuf_runtime_version()
    _preinit_tensorflow_runtime()
