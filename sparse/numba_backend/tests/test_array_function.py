import sparse
from sparse.numba_backend._settings import NEP18_ENABLED
from sparse.numba_backend._utils import assert_eq

import pytest

import numpy as np
import scipy

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
    """stack(), by design, does not allow for mixed type inputs"""
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


@pytest.mark.parametrize("format", ["dok", "gcxs", "coo"])
def test_format(format):
    s = sparse.random((5, 5), density=0.2, format=format)
    assert s.format == format


class TestAsarray:
    np_eye = np.eye(5)

    @pytest.mark.parametrize(
        "input",
        [
            np_eye,
            scipy.sparse.csr_matrix(np_eye),
            scipy.sparse.csc_matrix(np_eye),
            4,
            np.array(5),
            np.arange(12).reshape((2, 3, 2)),
            sparse.COO.from_numpy(np_eye),
            sparse.GCXS.from_numpy(np_eye),
            sparse.DOK.from_numpy(np_eye),
        ],
    )
    @pytest.mark.parametrize("dtype", [np.int64, np.float64, np.complex128])
    @pytest.mark.parametrize("format", ["dok", "gcxs", "coo"])
    def test_asarray(self, input, dtype, format):
        if format == "dok" and (np.isscalar(input) or input.ndim == 0):
            # scalars and 0-D arrays aren't supported in DOK format
            return

        s = sparse.asarray(input, dtype=dtype, format=format)

        actual = s.todense() if hasattr(s, "todense") else s
        expected = input.todense() if hasattr(input, "todense") else np.asarray(input)

        np.testing.assert_equal(actual, expected)
