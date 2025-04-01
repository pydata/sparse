try:
    import finch  # noqa: F401
except ModuleNotFoundError as e:
    raise ImportError("Finch not installed. Run `pip install sparse[finch]` to enable Finch backend") from e

from finch import *  # noqa: F403
from finch import __all__ as __all__
