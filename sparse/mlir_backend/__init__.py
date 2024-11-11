try:
    import mlir_finch  # noqa: F401

    del mlir_finch
except ModuleNotFoundError as e:
    raise ImportError(
        "MLIR Python bindings not installed. Run "
        "`conda install conda-forge::mlir-python-bindings` "
        "to enable MLIR backend."
    ) from e

from . import levels
from ._conversions import asarray, from_constituent_arrays, to_numpy, to_scipy
from ._dtypes import (
    asdtype,
    complex64,
    complex128,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
from ._ops import add, reshape

__all__ = [
    "add",
    "asarray",
    "asdtype",
    "to_numpy",
    "to_scipy",
    "levels",
    "reshape",
    "from_constituent_arrays",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
