try:
    import finch
except ModuleNotFoundError:
    raise ImportError("Finch not installed. Run `pip install sparse[finch]` to enable Finch backend")

from finch import Tensor

__all__ = ["Tensor"]
