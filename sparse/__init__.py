import os

_pydata = "pydata"
_finch = "finch"

backend = os.getenv("SPARSE_BACKEND", default=_pydata)

if backend == _pydata:
    from .pydata_backend import *
elif backend == _finch:
    from .finch_backend import *
else:
    raise ValueError(f"Invalid backend type: {backend}")
