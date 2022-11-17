def _init_extension():
    """
    Load extensions when numba is loaded.
    This name must match the one in setup.py
    """
    from ._coo import numba_extension
