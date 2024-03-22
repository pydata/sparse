try:
    import finch  # noqa: F401
except ModuleNotFoundError as e:
    raise ImportError("Finch not installed. Run `pip install sparse[finch]` to enable Finch backend") from e

from finch import Tensor, astype, permute_dims, random, multiply, subtract, sum, add, matmul, tensordot

__all__ = ["Tensor", "astype", "permute_dims", "random", "multiply", "subtract", "sum", "add", "matmul", "tensordot"]


class COO:
    def from_numpy(self):
        raise NotImplementedError
