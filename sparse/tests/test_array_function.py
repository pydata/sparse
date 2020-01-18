import sparse
from sparse._settings import NEP18_ENABLED
from sparse._utils import assert_eq
import numpy as np
import pytest


if not NEP18_ENABLED:
    pytest.skip("NEP18 is not enabled", allow_module_level=True)


@pytest.mark.parametrize(
    "func",
    [
        np.mean,
        np.std,
        np.var,
        np.sum,
        lambda x: np.sum(x, axis=0),
        lambda x: np.transpose(x),
    ],
)
def test_unary(func):
    y = sparse.random((50, 50), density=0.25)
    x = y.todense()
    xx = func(x)
    yy = func(y)
    assert_eq(xx, yy)


@pytest.mark.parametrize("arg_order", [(0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("func", [np.dot, np.result_type, np.tensordot, np.matmul])
def test_binary(func, arg_order):
    y = sparse.random((50, 50), density=0.25)
    x = y.todense()
    xx = func(x, x)
    args = [(x, y)[i] for i in arg_order]
    yy = func(*args)

    if isinstance(xx, np.ndarray):
        assert_eq(xx, yy)
    else:
        # result_type returns a dtype
        assert xx == yy


def test_stack():
    """stack(), by design, does not allow for mixed type inputs
    """
    y = sparse.random((50, 50), density=0.25)
    x = y.todense()
    xx = np.stack([x, x])
    yy = np.stack([y, y])
    assert_eq(xx, yy)


@pytest.mark.parametrize(
    "arg_order",
    [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
)
@pytest.mark.parametrize("func", [lambda a, b, c: np.where(a.astype(bool), b, c)])
def test_ternary(func, arg_order):
    y = sparse.random((50, 50), density=0.25)
    x = y.todense()
    xx = func(x, x, x)
    args = [(x, y)[i] for i in arg_order]
    yy = func(*args)
    assert_eq(xx, yy)


@pytest.mark.parametrize("func", [np.shape, np.size, np.ndim])
def test_property(func):
    y = sparse.random((50, 50), density=0.25)
    x = y.todense()
    xx = func(x)
    yy = func(y)
    assert xx == yy
