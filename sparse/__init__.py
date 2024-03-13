import os
from contextvars import ContextVar
from enum import Enum

__array_api_version__ = "2022.12"


class BackendType(Enum):
    PyData = "PyData"
    Finch = "Finch"


_ENV_VAR_NAME = "SPARSE_BACKEND"

backend_var = ContextVar("backend", default=BackendType.PyData)

if _ENV_VAR_NAME in os.environ:
    backend_var.set(BackendType[os.environ[_ENV_VAR_NAME]])


class Backend:
    def __init__(self, backend=BackendType.PyData):
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
        if backend == BackendType.PyData or backend == BackendType.Finch:
            import sparse.finch_backend as backend_module
        else:
            raise ValueError(f"Invalid backend identifier: {backend}")
        return backend_module


def __getattr__(attr):
    if attr == "pydata_backend":
        import sparse.finch_backend as backend_module

        return backend_module
    if attr == "finch_backend":
        import sparse.finch_backend as backend_module

        return backend_module

    return getattr(Backend.get_backend_module(), attr)
