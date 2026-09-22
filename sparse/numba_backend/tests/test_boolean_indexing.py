import sparse
from sparse.numba_backend._utils import assert_eq

import pytest

import numpy as np


@pytest.mark.parametrize("format", ["coo", "gcxs"])
@pytest.mark.parametrize("mask_format", ["coo", "gcxs"])
@pytest.mark.parametrize("mask_fill", [False, True])
@pytest.mark.parametrize("fill_value", [0, 2, np.nan])
@pytest.mark.parametrize("mask_ndim", [1, 2, 3])
def test_sparse_boolean_indexing(format, mask_format, mask_fill, fill_value, mask_ndim):
    dense = np.full((3, 4, 2), fill_value)
    dense[0, 1, 0] = 5
    dense[2, 3, 1] = -4
    dense_mask = np.arange(np.prod(dense.shape[:mask_ndim])).reshape(dense.shape[:mask_ndim]) % 3 != 1
    x = sparse.COO.from_numpy(dense, fill_value=fill_value).asformat(format)
    mask = sparse.COO.from_numpy(dense_mask, fill_value=mask_fill).asformat(mask_format)

    result = x[mask]
    assert isinstance(result, type(x))
    assert_eq(result, dense[dense_mask])
    assert_eq(x[(mask,)], result)
    assert_eq(x, dense)
    assert_eq(mask, dense_mask)


@pytest.mark.parametrize("format", ["coo", "gcxs"])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (4,), (2, 3)])
@pytest.mark.parametrize("mask_fill", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_sparse_boolean_indexing_constant_masks(format, shape, mask_fill, selected):
    dense = np.full(shape, 7)
    x = sparse.COO.from_numpy(dense, fill_value=7).asformat(format)
    dense_mask = np.full(shape, selected)
    mask = sparse.COO.from_numpy(dense_mask, fill_value=mask_fill)
    assert_eq(x[mask], dense[dense_mask])


@pytest.mark.parametrize("format", ["coo", "gcxs"])
@pytest.mark.parametrize("mask_fill", [False, True])
def test_sparse_boolean_indexing_explicit_fill(format, mask_fill):
    x = sparse.COO.from_numpy(np.array([0, 3, 0, 5, 0])).asformat(format)
    # Stored mask values may equal the fill value; those entries must not
    # change either the selected values or their positions in the result.
    mask = sparse.COO([[0, 1, 3]], [False, True, False], shape=(5,), fill_value=mask_fill)
    assert_eq(x[mask], x.todense()[mask.todense()])


@pytest.mark.parametrize("format", ["coo", "gcxs"])
@pytest.mark.parametrize("mask_format", ["coo", "gcxs"])
@pytest.mark.parametrize("mask_fill", [False, True])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("dense", [np.array(7), np.array([[0, 2], [3, 0]])])
def test_sparse_boolean_indexing_scalar_mask(format, mask_format, mask_fill, selected, dense):
    x = sparse.COO.from_numpy(dense).asformat(format)
    dense_mask = np.array(selected)
    mask = sparse.COO.from_numpy(dense_mask, fill_value=mask_fill).asformat(mask_format)
    assert_eq(x[mask], dense[dense_mask])


@pytest.mark.parametrize("format", ["coo", "gcxs"])
@pytest.mark.parametrize("mask_fill", [False, True])
def test_sparse_boolean_indexing_large_shape(format, mask_fill, monkeypatch):
    size = 10**12
    x = sparse.COO([[2, size - 1]], [3, 5], shape=(size,)).asformat(format)
    mask = sparse.COO([[1, 2]], [not mask_fill, not mask_fill], shape=x.shape, fill_value=mask_fill)

    def no_dense(*args, **kwargs):
        pytest.fail("Sparse boolean indexing must not densify its operands")

    with monkeypatch.context() as patch:
        patch.setattr(sparse.COO, "todense", no_dense)
        patch.setattr(sparse.GCXS, "todense", no_dense)
        result = x[mask].asformat("coo")

    if mask_fill:
        assert result.shape == (size - 2,)
        np.testing.assert_array_equal(result.coords, [[size - 3]])
        np.testing.assert_array_equal(result.data, [5])
    else:
        assert_eq(result, np.array([0, 3]))


@pytest.mark.parametrize("format", ["coo", "gcxs"])
@pytest.mark.parametrize("mask_shape", [(4,), (2, 2), (2, 3, 1)])
def test_sparse_boolean_indexing_shape_mismatch(format, mask_shape):
    x = sparse.zeros((2, 3), format=format)
    mask = sparse.zeros(mask_shape, dtype=bool)
    with pytest.raises(IndexError, match="boolean index"):
        x[mask]


def test_sparse_boolean_indexing_unsigned_coords():
    size = 2**54
    x = sparse.COO(np.array([[1], [size - 1]], dtype=np.uint64), [3], shape=(2, size))
    mask = sparse.COO.from_numpy(np.array([False, True]))
    result = x[mask]
    assert result.shape == (1, size)
    np.testing.assert_array_equal(result.coords, np.array([[0], [size - 1]], dtype=np.intp))
    assert np.issubdtype(result.coords.dtype, np.integer)
    np.testing.assert_array_equal(result.data, [3])
