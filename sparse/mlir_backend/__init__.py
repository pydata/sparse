try:
    import mlir  # noqa: F401
except ModuleNotFoundError as e:
    raise ImportError(
        "MLIR Python bindings not installed. Run "
        "`conda install conda-forge::mlir-python-bindings` "
        "to enable MLIR backend."
    ) from e

from ._constructors import (
    asarray,
)
from ._ops import (
    add,
)

__all__ = [
    "add",
    "asarray",
]
