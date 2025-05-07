import importlib.util
import os

import numpy as np

AUTO_DENSIFY = bool(int(os.environ.get("SPARSE_AUTO_DENSIFY", "0")))
WARN_ON_TOO_DENSE = bool(int(os.environ.get("SPARSE_WARN_ON_TOO_DENSE", "0")))


def _is_nep18_enabled():
    class A:
        def __array_function__(self, *args, **kwargs):
            return True

    try:
        return np.concatenate([A()])
    except ValueError:
        return False


def _supported_array_type() -> type[np.ndarray]:
    try:
        import cupy as cp

        return np.ndarray | cp.ndarray
    except ImportError:
        return np.ndarray


def _cupy_available() -> bool:
    return importlib.util.find_spec("cupy") is not None


NEP18_ENABLED = _is_nep18_enabled()
NUMPY_DEVICE = np.asarray(5).device
SUPPORTED_ARRAY_TYPE = _supported_array_type()
CUPY_AVAILABLE = _cupy_available()
