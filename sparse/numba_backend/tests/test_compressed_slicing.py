import math

import sparse
from sparse.numba_backend._compressed import indexing
from sparse.numba_backend._compressed.compressed import CSC, CSR
from sparse.numba_backend._utils import assert_eq

import pytest

import numpy as np


@pytest.mark.parametrize(
    "index",
    [
        (1, Ellipsis),
        (1, slice(1, None, 2), slice(None, None, -1), slice(None), slice(1, None, 3)),
    ],
)
def test_large_basic_slice_avoids_dense_column_selector(monkeypatch, index):
    # Regression for gh-853. The full column selector alone would need 34 GB.
    a = sparse.COO(
        [[1, 100, 215, 66], [5, 101, 242, 11], [3, 5, 1, 11], [13, 1, 3, 1], [55, 1, 6, 8]],
        [5, 10, 2, 1],
        shape=(255,) * 5,
    )
    x = a.asformat("gcxs")
    expected = a[index]
    convert_to_flat = indexing.convert_to_flat

    def bounded_convert_to_flat(inds, shape, dtype):
        # Fail safely before the old implementation can exhaust the test host.
        requested = math.prod(len(ind) for ind in inds)
        assert requested < 4096, f"dense column selector requested {requested} entries"
        return convert_to_flat(inds, shape, dtype)

    monkeypatch.setattr(indexing, "convert_to_flat", bounded_convert_to_flat)
    result = x[index]
    assert_eq(result.tocoo(), expected)


@pytest.mark.parametrize("fill_value", [0.0, 7.0, np.nan])
@pytest.mark.parametrize(
    "compressed_axes, index",
    [
        ((0, 2), (slice(None, None, -1), slice(1, None, 2), slice(None), slice(None, None, -2))),
        ((1, 3), (slice(None, None, -1), slice(None), slice(None, None, -2), slice(1, None, 2))),
        ((0, 2), (1, slice(None, None, -1), 2, slice(None, None, -2))),
        ((1, 3), (1, slice(None, None, -1), 2, slice(None, None, -2))),
        ((0, 1, 2), (slice(None, None, -1), 1, slice(None, None, -1), 2)),
        ((0, 2), (None, 1, slice(None, None, -1), 2, slice(1, None, 2), None)),
        ((0, 2), (slice(None), slice(2, 2), slice(None), slice(None))),
        ((1, 3), (slice(None), slice(2, 2), slice(None), slice(None))),
    ],
)
def test_basic_slice_multiaxis_fill_value(compressed_axes, index, fill_value):
    dense = np.full((3, 4, 5, 6), fill_value)
    locations = np.arange(dense.size).reshape(dense.shape)
    stored = locations % 7 == 0
    dense[stored] = locations[stored] + 1
    x = sparse.GCXS.from_numpy(dense, compressed_axes=compressed_axes, fill_value=fill_value)
    original = (x.data.copy(), x.indices.copy(), x.indptr.copy())

    result = x[index]

    assert_eq(result, dense[index])
    for actual, expected in zip((x.data, x.indices, x.indptr), original, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("idx_dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("compressed_axes", [(0,), (0, 2)])
def test_basic_slice_index_dtypes(idx_dtype, compressed_axes):
    dense = np.arange(60).reshape(3, 4, 5)
    dense[dense % 3 != 0] = 0
    x = sparse.GCXS.from_numpy(dense, compressed_axes=compressed_axes, idx_dtype=idx_dtype)
    index = (slice(None, None, -1), slice(None, None, -2), slice(1, None, 2))

    result = x[index]

    assert_eq(result, dense[index])
    assert result.indices.dtype == x.indices.dtype


@pytest.mark.parametrize("step", [2**63, -(2**63), 2**64, -(2**64)])
@pytest.mark.parametrize("empty", [False, True])
def test_basic_slice_large_step(step, empty):
    dense = np.zeros((3, 4, 5), dtype=np.int64)
    if not empty:
        dense[0, 0, 1] = 4
        dense[1, 3, 2] = 7
        dense[1, 1, 4] = 9
    x = sparse.GCXS.from_numpy(dense, compressed_axes=(0,))
    index = (slice(None), slice(None, None, step), slice(None))

    assert_eq(x[index], dense[index])


@pytest.mark.parametrize("compressed_axes", [(0,), (1, 2)])
@pytest.mark.parametrize(
    "index",
    [
        (1, slice(None, None, -1), slice(None)),
        (slice(None, None, -1), 2, slice(None)),
        (slice(None), slice(0, 0), slice(None)),
        (None, slice(None), 2, slice(None, None, -1), None),
    ],
)
def test_basic_slice_no_stored_values(compressed_axes, index):
    dense = np.full((3, 4, 5), 9)
    x = sparse.GCXS.from_numpy(dense, compressed_axes=compressed_axes, fill_value=9)

    result = x[index]

    assert_eq(result, dense[index])
    assert result.nnz == 0


@pytest.mark.parametrize("cls", [CSR, CSC])
@pytest.mark.parametrize(
    "index",
    [
        (slice(None, None, -1), slice(1, None, 2)),
        (1, slice(None, None, -2)),
        (slice(0, 0), slice(None)),
    ],
)
def test_basic_slice_csr_csc(cls, index):
    dense = np.arange(30).reshape(5, 6)
    dense[dense % 4 != 0] = 0
    x = cls.from_numpy(dense)

    assert_eq(x[index], dense[index])


@pytest.mark.parametrize("compressed_axes", [(0,), (1,)])
@pytest.mark.parametrize(
    "index",
    [
        ([2, 0, 2], slice(None, None, -1)),
        (slice(None, None, -1), [3, 1, 3]),
    ],
)
def test_advanced_slice_repeated_indices(compressed_axes, index):
    dense = np.arange(20).reshape(4, 5)
    dense[dense % 3 != 0] = 0
    x = sparse.GCXS.from_numpy(dense, compressed_axes=compressed_axes)

    assert_eq(x[index], dense[index])
