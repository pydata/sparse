import sparse
from sparse._settings import NEP18_ENABLED
from sparse._utils import assert_eq
import numpy as np
import pytest
from hypothesis import settings, given, strategies as st
from _utils import gen_sparse_random


if not NEP18_ENABLED:
    pytest.skip("NEP18 is not enabled", allow_module_level=True)


@settings(deadline=None)
@given(
    func=st.sampled_from(
        [
            np.mean,
            np.std,
            np.var,
            np.sum,
            lambda x: np.sum(x, axis=0),
            lambda x: np.transpose(x),
        ]
    ),
    y=gen_sparse_random((50, 50), density=0.25),
)
def test_unary(func, y):
    x = y.todense()
    xx = func(x)
    yy = func(y)
    assert_eq(xx, yy)


@settings(deadline=None)
@given(
    arg_order=st.sampled_from([(0, 1), (1, 0), (1, 1)]),
    func=st.sampled_from([np.dot, np.result_type, np.tensordot, np.matmul]),
    y=gen_sparse_random((50, 50), density=0.25),
)
def test_binary(func, arg_order, y):
    x = y.todense()
    xx = func(x, x)
    args = [(x, y)[i] for i in arg_order]
    yy = func(*args)

    if isinstance(xx, np.ndarray):
        assert_eq(xx, yy)
    else:
        # result_type returns a dtype
        assert xx == yy


@given(y=gen_sparse_random((50, 50), density=0.25))
def test_stack(y):
    """stack(), by design, does not allow for mixed type inputs"""
    x = y.todense()
    xx = np.stack([x, x])
    yy = np.stack([y, y])
    assert_eq(xx, yy)


@given(
    arg_order=st.sampled_from(
        [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    ),
    func=st.sampled_from([lambda a, b, c: np.where(a.astype(bool), b, c)]),
    y=gen_sparse_random((50, 50), density=0.25),
)
def test_ternary(func, arg_order, y):
    x = y.todense()
    xx = func(x, x, x)
    args = [(x, y)[i] for i in arg_order]
    yy = func(*args)
    assert_eq(xx, yy)


@given(
    func=st.sampled_from([np.shape, np.size, np.ndim]),
    y=gen_sparse_random((50, 50), density=0.25),
)
def test_property(func, y):
    x = y.todense()
    xx = func(x)
    yy = func(y)
    assert xx == yy


def test_broadcast_to_scalar():
    s = sparse.COO.from_numpy([0, 0, 1, 2])
    actual = np.broadcast_to(np.zeros_like(s, shape=()), (3,))
    expected = np.broadcast_to(np.zeros_like(s.todense(), shape=()), (3,))

    assert isinstance(actual, sparse.COO)
    assert_eq(actual, expected)


def test_zeros_like_order():
    s = sparse.COO.from_numpy([0, 0, 1, 2])
    actual = np.zeros_like(s, order="C")
    expected = np.zeros_like(s.todense(), order="C")

    assert isinstance(actual, sparse.COO)
    assert_eq(actual, expected)
