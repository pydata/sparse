import os
from contextvars import ContextVar
from enum import Enum

from ._version import __version__, __version_tuple__  # noqa: F401


class BackendType(Enum):
    Numba = "Numba"
    Finch = "Finch"


_ENV_VAR_NAME = "SPARSE_BACKEND"

backend_var = ContextVar("backend", default=BackendType.Numba)

if _ENV_VAR_NAME in os.environ:
    backend_var.set(BackendType[os.environ[_ENV_VAR_NAME]])


class Backend:
    def __init__(self, backend=BackendType.Numba):
        self.backend = backend
        self.token = None

    def __enter__(self):
        self.token = backend_var.set(self.backend)

    def __exit__(self, exc_type, exc_value, traceback):
        backend_var.reset(self.token)
        self.token = None

    @staticmethod
    def get_backend_module():
        backend = backend_var.get()
        if backend == BackendType.Numba:
            import sparse.numba_backend as backend_module
        elif backend == BackendType.Finch:
            import sparse.finch_backend as backend_module
        else:
            raise ValueError(f"Invalid backend identifier: {backend}")
        return backend_module


def __getattr__(attr):
    if attr == "numba_backend" or attr == "finch_backend":
        raise AttributeError

    return getattr(Backend.get_backend_module(), attr)
