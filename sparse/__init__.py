import os
import warnings
from enum import Enum

from ._version import __version__, __version_tuple__  # noqa: F401

__array_api_version__ = "2022.12"


class _BackendType(Enum):
    Numba = "Numba"
    Finch = "Finch"
    MLIR = "MLIR"


_ENV_VAR_NAME = "SPARSE_BACKEND"


class SparseFutureWarning(FutureWarning):
    pass


if os.environ.get(_ENV_VAR_NAME, "") != "":
    warnings.warn(
        "Changing back-ends is a development feature, please do not rely on it in production.",
        SparseFutureWarning,
        stacklevel=1,
    )
    _backend_name = os.environ[_ENV_VAR_NAME]
else:
    _backend_name = _BackendType.Numba.value

if _backend_name not in {v.value for v in _BackendType}:
    warnings.warn(f"Invalid backend identifier: {_backend_name}. Selecting Numba backend.", UserWarning, stacklevel=1)
    _BACKEND = _BackendType.Numba
else:
    _BACKEND = _BackendType[_backend_name]

del _backend_name

if _BackendType.Finch == _BACKEND:
    from sparse.finch_backend import *  # noqa: F403
    from sparse.finch_backend import __all__
elif _BackendType.MLIR == _BACKEND:
    from sparse.mlir_backend import *  # noqa: F403
    from sparse.mlir_backend import __all__
else:
    from sparse.numba_backend import *  # noqa: F403
    from sparse.numba_backend import (  # noqa: F401
        __all__,
        _common,
        _compressed,
        _coo,
        _dok,
        _io,
        _numba_extension,
        _settings,
        _slicing,
        _sparse_array,
        _umath,
        _utils,
    )
