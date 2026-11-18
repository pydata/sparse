import sparse
from sparse.numba_backend._utils import assert_eq

import pytest

import numpy as np


@pytest.mark.parametrize("reduction", ["sum", "prod"])
@pytest.mark.parametrize("fill_value", [0.0, 2.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("shape, axis", [((0,), None), ((0, 3), 0), ((2, 0, 3), (1, 2)), ((0, 3), 1)])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("dtype", [None, np.float32])
def test_empty_reduction_fill_value(reduction, fill_value, shape, axis, keepdims, dtype):
    x = np.empty(shape)
    s = sparse.COO.from_numpy(x, fill_value=fill_value)
    kwargs = {"axis": axis, "keepdims": keepdims, "dtype": dtype}

    with np.errstate(invalid="raise"):
        expected = getattr(np, reduction)(x, **kwargs)
        assert_eq(expected, getattr(s, reduction)(**kwargs))
        assert_eq(expected, getattr(np, reduction)(s, **kwargs))
