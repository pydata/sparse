import numpy as np


def assert_eq(x, y, **kwargs):
    from .coo import COO
    assert x.shape == y.shape
    assert x.dtype == y.dtype

    if isinstance(x, COO):
        if x.sorted:
            assert is_lexsorted(x)
    if isinstance(y, COO):
        if y.sorted:
            assert is_lexsorted(y)

    if hasattr(x, 'todense'):
        xx = x.todense()
    else:
        xx = x
    if hasattr(y, 'todense'):
        yy = y.todense()
    else:
        yy = y
    assert np.allclose(xx, yy, **kwargs)


def is_lexsorted(x):
    return not x.shape or (np.diff(x.linear_loc()) > 0).all()


def _zero_of_dtype(dtype):
    """
    Creates a ()-shaped 0-dimensional zero array of a given dtype.

    Parameters
    ----------
    dtype : numpy.dtype
        The dtype for the array.

    Returns
    -------
    np.ndarray
        The zero array.
    """
    return np.zeros((), dtype=dtype)
