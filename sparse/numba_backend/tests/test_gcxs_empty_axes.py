"""GCXS reductions over no axes must preserve values, shape and NumPy dtypes."""

import sparse

import pytest

import numpy as np


@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (0, 3)])
@pytest.mark.parametrize("reduction", ["sum", "prod", "min", "max", "any", "all", "mean", "var", "std"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_gcxs_reduce_no_axes(shape, reduction, keepdims):
    dense = (np.arange(np.prod(shape), dtype=np.float64) % 3).reshape(shape)
    array = sparse.GCXS.from_numpy(dense)
    expected = getattr(dense, reduction)(axis=(), keepdims=keepdims)

    result = getattr(array, reduction)(axis=(), keepdims=keepdims)

    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result.todense(), expected)


@pytest.mark.parametrize("value", [0, 7, -3.5, True, 2 + 3j])
@pytest.mark.parametrize("reduction", ["sum", "prod", "any", "all", "mean", "var", "std"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_gcxs_scalar_reduce_all_axes(value, reduction, keepdims):
    dense = np.asarray(value)
    array = sparse.GCXS.from_numpy(dense)
    expected = getattr(dense, reduction)(keepdims=keepdims)

    result = getattr(array, reduction)(keepdims=keepdims)

    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result.todense(), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
@pytest.mark.parametrize("ufunc", [np.add, np.multiply])
def test_gcxs_no_axes_accumulator_dtype(dtype, ufunc):
    dense = np.array([[0, 2], [3, 0]], dtype=np.int8)
    array = sparse.GCXS.from_numpy(dense)

    result = ufunc.reduce(array, axis=(), dtype=dtype)
    expected = ufunc.reduce(dense, axis=(), dtype=dtype)

    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result.todense(), expected)


def test_gcxs_no_axes_does_not_densify_large_array():
    array = sparse.COO(
        coords=np.array([[1, 999_999_999], [2, 3]]),
        data=np.array([4, 7], dtype=np.int8),
        shape=(1_000_000_000, 4),
    ).asformat("gcxs", compressed_axes=(1,))

    result = array.sum(axis=(), dtype=np.int64).asformat("coo")
    expected = array.tocoo()

    assert result.shape == array.shape
    assert result.dtype == np.dtype(np.int64)
    assert result.fill_value == 0
    assert result.nnz == 2
    np.testing.assert_array_equal(result.coords, expected.coords)
    np.testing.assert_array_equal(result.data, expected.data)
