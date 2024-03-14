try:
    import finch  # noqa: F401
except ModuleNotFoundError as e:
    raise ImportError("Finch not installed. Run `pip install sparse[finch]` to enable Finch backend") from e

from finch import Tensor, astype, permute_dims

__all__ = ["Tensor", "astype", "permute_dims"]


class COO:
    def from_numpy(self):
        raise NotImplementedError
