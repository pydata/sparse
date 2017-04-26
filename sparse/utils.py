import numpy as np


def assert_eq(x, y):
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    if hasattr(x, 'todense'):
        xx = x.todense()
    else:
        xx = x
    if hasattr(y, 'todense'):
        yy = y.todense()
    else:
        yy = y
    assert np.allclose(xx, yy)
