try:
    import mlir  # noqa: F401

    del mlir
except ModuleNotFoundError as e:
    raise ImportError(
        "MLIR Python bindings not installed. Run "
        "`conda install conda-forge::mlir-python-bindings` "
        "to enable MLIR backend."
    ) from e

from . import levels
from ._conversions import asarray, from_constituent_arrays, to_numpy, to_scipy
from ._dtypes import asdtype
from ._ops import add

__all__ = ["add", "asarray", "asdtype", "to_numpy", "to_scipy", "levels", "from_constituent_arrays"]
