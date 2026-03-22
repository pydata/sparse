import sparse
from sparse.numba_backend import SparseArray
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
            scipy.sparse.csr_array(np_eye),
            scipy.sparse.csc_array(np_eye),
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

        if isinstance(input, SparseArray):
            assert sparse.asarray(input).__class__ is input.__class__


class TestArrayAPIReductions:
    """
    Array API standard compliance: reductions over the entire array must return
    a zero-dimensional array, not a NumPy scalar.

    See: https://github.com/pydata/sparse/issues/921
    """

    @pytest.mark.parametrize("format", ["coo", "gcxs"])
    @pytest.mark.parametrize(
        "fn, expected",
        [
            (sparse.sum, 2.0),
            (sparse.max, 1.0),
            (sparse.min, 0.0),
            (sparse.prod, 0.0),
            (sparse.mean, 0.5),
        ],
    )
    def test_full_reduction_returns_0d_array(self, fn, expected, format):
        x = sparse.asarray(np.eye(2), format=format)
        result = fn(x)
        assert result.ndim == 0, (
            f"{fn.__name__}() over entire array returned ndim={result.ndim}, expected 0-D array"
        )
        assert isinstance(result, SparseArray), (
            f"{fn.__name__}() returned {type(result).__name__}, expected a SparseArray"
        )
        assert abs(float(result) - expected) < 1e-9, (
            f"{fn.__name__}() returned {float(result)}, expected {expected}"
        )

    @pytest.mark.parametrize("fn", [sparse.any, sparse.all])
    def test_boolean_reduction_returns_0d_array(self, fn):
        x = sparse.asarray(np.eye(2), format="coo")
        result = fn(x)
        assert result.ndim == 0, (
            f"{fn.__name__}() returned ndim={result.ndim}, expected 0-D array"
        )
        assert isinstance(result, SparseArray), (
            f"{fn.__name__}() returned {type(result).__name__}, expected a SparseArray"
        )

    def test_partial_reduction_still_returns_nd_array(self):
        """Axis-specific reductions must still return N-D sparse arrays."""
        x = sparse.asarray(np.eye(2), format="coo")

        result_ax0 = sparse.sum(x, axis=0)
        assert result_ax0.shape == (2,), f"Expected shape (2,), got {result_ax0.shape}"
        assert isinstance(result_ax0, SparseArray)

        result_ax1 = sparse.sum(x, axis=1)
        assert result_ax1.shape == (2,), f"Expected shape (2,), got {result_ax1.shape}"
        assert isinstance(result_ax1, SparseArray)

    def test_keepdims_full_reduction(self):
        """keepdims=True must preserve all dimensions as size-1."""
        x = sparse.asarray(np.eye(2), format="coo")
        result = sparse.sum(x, keepdims=True)
        assert result.shape == (1, 1), f"Expected shape (1, 1), got {result.shape}"
        assert isinstance(result, SparseArray)

    @pytest.mark.parametrize("format", ["coo", "gcxs"])
    def test_1d_full_reduction_returns_0d_array(self, format):
        """1-D input fully reduced must also give a 0-D array."""
        x = sparse.asarray(np.array([1.0, 2.0, 3.0]), format=format)
        result = sparse.sum(x)
        assert result.ndim == 0, f"Expected 0-D array, got ndim={result.ndim}"
        assert isinstance(result, SparseArray)
        assert abs(float(result) - 6.0) < 1e-9
