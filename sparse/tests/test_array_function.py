import sparse
from sparse._settings import NEP18_ENABLED
from sparse._utils import assert_eq
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

if not NEP18_ENABLED:
    pytest.skip("NEP18 is not enabled", allow_module_level=True)

def arg_order_generator(number_of_arguments):
    return st.tuples(*[st.integers(0, 1) for i in range(number_of_arguments)]).filter(
        lambda x: any(x)
    )


@settings(deadline=None)
@given(
    st.sampled_from(
        [
            np.mean,
            np.std,
            np.var,
            np.sum,
            lambda x: np.sum(x, axis=0),
            lambda x: np.transpose(x),
        ]
    ),
    st.tuples(st.integers(1, 10), st.integers(1, 10)),
)
def test_unary(func, shape):
    y = sparse.random(shape, density=0.25)
    x = y.todense()
    xx = func(x)
    yy = func(y)
    assert_eq(xx, yy)


@settings(deadline=None)
@given(
    st.sampled_from([np.dot, np.result_type, np.tensordot, np.matmul]),
    arg_order_generator(2),
    st.integers(2, 10),
)
def test_binary(func, arg_order, shape):
    y = sparse.random((shape, shape), density=0.25)
    x = y.todense()
    xx = func(x, x)
    args = [(x, y)[i] for i in arg_order]
    yy = func(*args)

    if isinstance(xx, np.ndarray):
        assert_eq(xx, yy)
    else:
        # result_type returns a dtype
        assert xx == yy


@given(st.tuples(st.integers(1, 10), st.integers(1, 10)))
def test_stack(shape):
    """stack(), by design, does not allow for mixed type inputs"""
    y = sparse.random(shape, density=0.25)
    x = y.todense()
    xx = np.stack([x, x])
    yy = np.stack([y, y])
    assert_eq(xx, yy)


@settings(deadline=None)
@given(
    st.sampled_from([lambda a, b, c: np.where(a.astype(bool), b, c)]),
    arg_order_generator(3),
    st.integers(2, 10),
)
def test_ternary(func, arg_order, shape):
    y = sparse.random((shape, shape), density=0.25)
    x = y.todense()
    xx = func(x, x, x)
    args = [(x, y)[i] for i in arg_order]
    yy = func(*args)
    assert_eq(xx, yy)


@given(
    st.sampled_from([np.shape, np.size, np.ndim]),
    st.tuples(st.integers(1, 10), st.integers(1, 10)),
)
def test_property(func, shape):
    y = sparse.random(shape, density=0.25)
    x = y.todense()
    xx = func(x)
    yy = func(y)
    assert xx == yy
