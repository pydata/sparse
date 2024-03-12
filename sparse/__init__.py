import os
from contextvars import ContextVar
from enum import Enum


class BackendType(Enum):
    pydata = "pydata"
    finch = "finch"


_ENV_VAR_NAME = "SPARSE_BACKEND"

backend_var = ContextVar("backend", default=BackendType.pydata)

if _ENV_VAR_NAME in os.environ:
    backend_var.set(BackendType[os.environ[_ENV_VAR_NAME]])


class Backend:
    def __init__(self, backend=BackendType.pydata):
        self.backend = backend
        self.token = None

    def __enter__(self):
        token = backend_var.set(self.backend)
        self.token = token

    def __exit__(self, exc_type, exc_value, traceback):
        backend_var.reset(self.token)
        self.token = None

    @staticmethod
    def get_backend_module():
        backend = backend_var.get()
        if backend == BackendType.pydata:
            import sparse.pydata_backend as backend_module
        elif backend == BackendType.finch:
            import sparse.finch_backend as backend_module
        else:
            raise ValueError(f"Invalid backend identifier: {backend}")
        return backend_module


def __getattr__(attr):
    return getattr(Backend.get_backend_module(), attr)


def __dir__():
    return Backend.get_backend_module().__dir__()


from ._version import __version__, __version_tuple__  # noqa: F401

__array_api_version__ = "2022.12"
