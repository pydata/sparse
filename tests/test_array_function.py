import os
import sparse
from sparse.utils import assert_eq
import pytest
np = pytest.importorskip('numpy', minversion='1.16')


env_name = "NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"

if np.__version__ < "1.17":
    array_function_env = os.getenv(env_name, "0")
else:
    array_function_env = os.getenv(env_name, "1")

if array_function_env != "1":
    pytest.skip(
        "NUMPY_EXPERIMENTAL_ARRAY_FUNCTION is not enabled", allow_module_level=True
    )


@pytest.mark.parametrize('func', [
    np.mean,
    np.std,
    np.var,
    np.sum,
    lambda x: np.sum(x, axis=0),
    np.transpose,
])
def test_unary(func):
    y = sparse.random((50, 50), density=.25)
    x = y.todense()
    xx = func(x)
    yy = func(y)
    assert_eq(xx, yy)


@pytest.mark.parametrize('arg_order', [
    (0, 1), (1, 0), (1, 1)
])
@pytest.mark.parametrize('func', [
    np.dot,
    lambda x, y: np.stack([x, y]),
    np.matmul,
    np.result_type,
    np.tensordot,
])
def test_binary(func, arg_order):
    y = sparse.random((50, 50), density=.25)
    x = y.todense()
    xx = func(x, x)
    args = [(x, y)[i] for i in arg_order]
    yy = func(*args)

    if isinstance(xx, np.ndarray):
        assert_eq(xx, yy)
    else:
        # result_type returns a dtype
        assert xx == yy


@pytest.mark.parametrize('arg_order', [
    (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0),
    (1, 0, 1), (1, 1, 0), (1, 1, 1)
])
@pytest.mark.parametrize('func', [
    lambda a, b, c: np.where(a.astype(bool), b, c),
])
def test_ternary(func, arg_order):
    y = sparse.random((50, 50), density=.25)
    x = y.todense()
    xx = func(x, x, x)
    args = [(x, y)[i] for i in arg_order]
    yy = func(*args)
    assert_eq(xx, yy)
