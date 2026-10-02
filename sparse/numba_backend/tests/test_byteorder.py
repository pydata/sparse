import sparse
from sparse.numba_backend._compressed import CSC, CSR

import pytest

import numpy as np
import scipy.sparse as sps

FORMATS = [sparse.COO, sparse.GCXS, CSR, CSC]


def from_buffers(format, data, fill_value=None):
    if format is sparse.COO:
        coords = np.array([[0, 1], [1, 0]])
        result = format(coords, data, shape=(2, 2), sorted=True, has_duplicates=False, fill_value=fill_value)
        assert result.coords is coords
    else:
        indices = np.array([1, 0])
        indptr = np.array([0, 1, 2])
        kwargs = {"compressed_axes": (0,)} if format is sparse.GCXS else {}
        result = format((data, indices, indptr), shape=(2, 2), fill_value=fill_value, **kwargs)
        assert result.indices is indices
        assert result.indptr is indptr
    return result


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("dtype", ["f4", "f8", "c8", "c16", "i4", "u8", "i1", "bool", "S3", "U3", "O"])
@pytest.mark.parametrize("byteorder", ["=", "S"])
def test_constructor_byte_order(format, dtype, byteorder):
    dtype = np.dtype(dtype).newbyteorder(byteorder)
    data = np.array([0, 2, 0, 3], dtype=dtype)[::-2]
    if dtype.kind == "c":
        data += 1j
    original = data.tobytes()
    data.flags.writeable = False

    result = from_buffers(format, data)

    assert result.dtype == dtype.newbyteorder("=")
    np.testing.assert_array_equal(result.data, data)
    assert (result.data is data) == dtype.isnative
    assert data.tobytes() == original
    assert not data.flags.writeable


@pytest.mark.parametrize("format", FORMATS)
def test_constructor_non_native_memmap(format, tmp_path):
    dtype = np.dtype("f8").newbyteorder("S")
    path = tmp_path / "data.bin"
    np.array([2, 3], dtype=dtype).tofile(path)
    original = path.read_bytes()
    data = np.memmap(path, dtype=dtype, mode="r", shape=(2,))

    result = from_buffers(format, data)

    assert result.dtype.isnative
    np.testing.assert_array_equal(result.data, data)
    assert not np.shares_memory(result.data, data)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "format,source_format",
    [(format, source) for format in (sparse.COO, sparse.GCXS) for source in ("dense", "coo", "gcxs")]
    + [(sparse.COO, "dok"), (sparse.COO, "scipy_coo"), (sparse.GCXS, "scipy_csr"), (sparse.GCXS, "scipy_csc")],
)
def test_constructor_non_native_conversion(format, source_format):
    dense = np.array([[0, 2], [3, 0]], dtype=np.dtype("f8").newbyteorder("S"))
    if source_format == "dense":
        source = dense
    elif source_format.startswith("scipy_"):
        source = getattr(sps, source_format[6:] + "_array")(dense.astype("f8"))
        source.data = source.data.astype(dense.dtype)
    else:
        source = sparse.COO.from_numpy(dense).asformat(source_format)
        if source_format != "dok":
            source.data = source.data.astype(dense.dtype)
    original = source.data.tobytes() if hasattr(source, "data") and isinstance(source.data, np.ndarray) else None

    result = format(source)

    assert result.dtype.isnative
    np.testing.assert_array_equal(result.todense(), dense)
    if original is not None:
        assert source.data.tobytes() == original
        assert not source.dtype.isnative


@pytest.mark.parametrize("compressed_axes", [(0,), (1,)])
def test_gcxs_non_native_copy(compressed_axes):
    source = sparse.GCXS.from_numpy(np.array([[0, 2], [3, 0]]), compressed_axes=(0,))
    source.data = source.data.astype(source.dtype.newbyteorder("S"))
    original = source.data.tobytes()

    result = sparse.GCXS(source, compressed_axes=compressed_axes)

    assert result.dtype.isnative
    assert result.compressed_axes == compressed_axes
    np.testing.assert_array_equal(result.todense(), source.todense())
    assert source.data.tobytes() == original
    assert not source.dtype.isnative


@pytest.mark.parametrize("fill_value", [None, 5])
def test_coo_non_native_copy_cache(fill_value):
    source = sparse.COO.from_numpy(np.array([[0, 2], [3, 0]]))
    source.data = source.data.astype(source.dtype.newbyteorder("S"))
    source.enable_caching()
    transpose = source.T
    reshape = source.reshape((4,))
    csr = source.tocsr()
    csc = source.tocsc()
    cache = source._cache

    result = sparse.COO(source, fill_value=fill_value)

    assert result.dtype.isnative
    assert result.coords is source.coords
    assert result._cache is not None
    assert result._cache is not cache
    expected = source.todense()
    if fill_value is not None:
        expected[expected == source.fill_value] = fill_value
    np.testing.assert_array_equal(result.todense(), expected)
    np.testing.assert_array_equal(result.T.todense(), expected.T)
    for converted in [result.T, result.reshape((4,))]:
        assert converted.dtype.isnative
    if fill_value is None:
        assert result.tocsr().dtype.isnative
        assert result.tocsc().dtype.isnative
    assert source._cache is cache
    assert source.T is transpose
    assert source.reshape((4,)) is reshape
    assert source.tocsr() is csr
    assert source.tocsc() is csc
    assert not source.dtype.isnative


@pytest.mark.parametrize("format", [sparse.COO, sparse.GCXS])
@pytest.mark.parametrize("shape", [(), (0,), (3,), (0, 2), (2, 0)])
def test_constructor_non_native_shape(format, shape):
    dense = np.zeros(shape, dtype=np.dtype("c8").newbyteorder("S"))

    result = format.from_numpy(dense)

    assert result.dtype.isnative
    assert result.shape == shape
    np.testing.assert_array_equal(result.todense(), dense)


@pytest.mark.parametrize("format", [sparse.COO, sparse.GCXS])
@pytest.mark.parametrize("byteorder", ["=", "S"])
def test_constructor_structured_byte_order(format, byteorder):
    swapped = np.dtype("i4").newbyteorder(byteorder)
    dtype = np.dtype([("native", "f8"), ("nested", [("values", swapped, (2,))]), ("object", "O")])
    data = np.array([(2.5, ([3, 4],), "a"), (5.5, ([6, 7],), "b")], dtype=dtype)
    fill = np.array((1.5, ([8, 8],), "fill"), dtype=dtype)[()]
    original = data.copy()

    result = from_buffers(format, data, fill)

    assert result.dtype == dtype.newbyteorder("=")
    assert result.fill_value.dtype == result.dtype
    assert (result.data is data) == (dtype == dtype.newbyteorder("="))
    np.testing.assert_array_equal(result.data, original)
    expected = np.full((2, 2), fill, dtype=dtype.newbyteorder("="))
    expected[0, 1], expected[1, 0] = data
    np.testing.assert_array_equal(result.todense(), expected)
    np.testing.assert_array_equal(result.asformat("coo")["nested"]["values"].todense(), expected["nested"]["values"])
    np.testing.assert_array_equal(data, original)
    assert fill.dtype == dtype


def test_coo_non_native_duplicates_and_fill():
    dtype = np.dtype("f8").newbyteorder("S")
    data = np.array([1, 2, 3], dtype=dtype)

    result = sparse.COO([[1, 0, 1]], data, shape=(3,), prune=True, fill_value=2)

    assert result.dtype.isnative
    np.testing.assert_array_equal(result.todense(), [2, 4, 2])
    assert result.nnz == 1
    np.testing.assert_array_equal(data, [1, 2, 3])


def test_coo_native_copy_cache():
    source = sparse.COO.from_numpy(np.eye(2))
    source.enable_caching()
    source.transpose()
    source.tocsr()

    result = sparse.COO(source)

    assert result.data is source.data
    assert result.coords is source.coords
    assert result._cache is source._cache
    assert result.tocsr() is source.tocsr()


def test_coo_non_native_scalar_data():
    data = np.array(2, dtype=np.dtype("i4").newbyteorder("S"))

    result = sparse.COO([[0, 2]], data, shape=(3,))

    assert result.dtype.isnative
    np.testing.assert_array_equal(result.todense(), [2, 0, 2])
    assert not data.dtype.isnative


@pytest.mark.parametrize("format", ["coo", "csr", "csc"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("func", [sparse.dot, sparse.matmul])
def test_non_native_scipy_contraction(format, side, func):
    dense = np.array([[0, 2], [3, 0]], dtype="f8")
    scipy_array = getattr(sps, format + "_array")(dense)
    scipy_array.data = scipy_array.data.astype(np.dtype("f8").newbyteorder("S"))
    scipy_array.data.flags.writeable = False
    other = sparse.COO.from_numpy(dense)
    args = (scipy_array, other) if side == "left" else (other, scipy_array)

    result = func(*args)

    np.testing.assert_array_equal(result.todense(), dense @ dense)
    assert result.dtype.isnative
    assert not scipy_array.dtype.isnative
    assert not scipy_array.data.flags.writeable
