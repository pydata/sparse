def _init_extension():
    """
    Load extensions when numba is loaded.
    This name must match the one in pyproject.toml
    """
    from ._coo import numba_extension  # noqa: F401
