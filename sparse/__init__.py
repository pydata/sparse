import os
import warnings
from enum import Enum

from ._version import __version__, __version_tuple__  # noqa: F401

__array_api_version__ = "2022.12"


class BackendType(Enum):
    Numba = "Numba"
    Finch = "Finch"


_ENV_VAR_NAME = "SPARSE_BACKEND"

if _ENV_VAR_NAME in os.environ:
    warnings.warn("Selectable backends feature in `sparse` might change in the future.", FutureWarning, stacklevel=1)
    backend_name = os.environ[_ENV_VAR_NAME]

    if BackendType[backend_name] == BackendType.Finch:
        from sparse.finch_backend import *  # noqa: F403
        from sparse.finch_backend import __all__
    elif BackendType[backend_name] == BackendType.Numba:
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

    else:
        warnings.warn(
            f"Invalid backend identifier: {backend_name}. Selecting Numba backend.", UserWarning, stacklevel=1
        )
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
