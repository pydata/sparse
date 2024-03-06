try:
    import finch
except ModuleNotFoundError:
    raise ImportError("Finch not installed. Run `pip install sparse[finch]` to enable Finch backend")

from finch import (
    Tensor,
    astype,
    permute_dims
)

__all__ = ["Tensor", "astype", "permute_dims"]
