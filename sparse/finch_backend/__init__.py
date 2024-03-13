import juliapkg

juliapkg.add("Finch", "9177782c-1635-4eb9-9bfb-d9dfa25e6bce", version="0.6.16")
juliapkg.resolve()

from juliacall import Main as jl  # noqa: E402

jl.seval("using Finch")

try:
    import finch  # noqa: F401
except ModuleNotFoundError:
    raise ImportError("Finch not installed. Run `pip install sparse[finch]` to enable Finch backend") from None

from finch import Tensor, astype, permute_dims  # noqa: E402

__all__ = ["Tensor", "permute_dims", "astype"]


class COO:
    def from_numpy(self):
        raise NotImplementedError
