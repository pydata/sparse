import sparse
from sparse.numba_backend._compressed import CSC, CSR

import pytest

import numpy as np
import scipy.sparse as sps
from numpy.testing import assert_equal

FORMATS = [sparse.COO, sparse.GCXS, CSR, CSC]


def from_buffers(format, data, fill_value=None):
    if format is sparse.COO:
        coords = np.asarray([[0, 1], [1, 0]])
        result = format(coords, data, shape=(2, 2), sorted=True, has_duplicates=False, fill_value=fill_value)
        assert result.coords is coords
    else:
        indices = np.asarray([1, 0])
        indptr = np.asarray([0, 1, 2])
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
    data = np.asarray([0, 2, 0, 3], dtype=dtype)[::-2]
    if dtype.kind == "c":
        data += 1j
    original = data.tobytes()
    data.flags.writeable = False

    result = from_buffers(format, data)

    assert result.dtype == dtype.newbyteorder("=")
    assert_equal(result.data, data)
    assert (result.data is data) == dtype.isnative
    assert data.tobytes() == original
    assert not data.flags.writeable


@pytest.mark.parametrize("format", FORMATS)
def test_constructor_non_native_memmap(format, tmp_path):
    dtype = np.dtype("f8").newbyteorder("S")
    path = tmp_path / "data.bin"
    np.asarray([2, 3], dtype=dtype).tofile(path)
    original = path.read_bytes()
    data = np.memmap(path, dtype=dtype, mode="r", shape=(2,))

    result = from_buffers(format, data)

    assert result.dtype.isnative
    assert_equal(result.data, data)
    assert not np.shares_memory(result.data, data)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "format,source_format,shape",
    [(format, source, (2, 2)) for format in (sparse.COO, sparse.GCXS) for source in ("dense", "coo", "gcxs")]
    + [(sparse.COO, "dok", (2, 2)), (sparse.COO, "scipy_coo", (2, 2))]
    + [(sparse.GCXS, source, (2, 2)) for source in ("scipy_csr", "scipy_csc")]
    + [(format, "dense", shape) for format in (sparse.COO, sparse.GCXS) for shape in ((), (0,), (3,), (0, 2), (2, 0))],
)
def test_constructor_conversion_byte_order(format, source_format, shape):
    dense = np.arange(np.prod(shape)).reshape(shape).astype(np.dtype("f8").newbyteorder("S"))
    if source_format == "dense":
        source = dense
    elif source_format.startswith("scipy_"):
        source = getattr(sps, source_format[6:] + "_array")(dense.astype("f8"))
        source.data = source.data.astype(dense.dtype)
    else:
        source = sparse.COO.from_numpy(dense).asformat(source_format)

    result = format(source)

    assert_equal(result.todense(), dense)
    assert result.dtype == dense.dtype.newbyteorder("=")


@pytest.mark.parametrize("format", [sparse.COO, sparse.GCXS])
@pytest.mark.parametrize("byteorder", ["=", "S"])
def test_constructor_structured_byte_order(format, byteorder):
    swapped = np.dtype("i4").newbyteorder(byteorder)
    dtype = np.dtype([("native", "f8"), ("nested", [("values", swapped, (2,))]), ("object", "O")])
    data = np.asarray([(2.5, ([3, 4],), "a"), (5.5, ([6, 7],), "b")], dtype=dtype)
    fill = np.asarray((1.5, ([8, 8],), "fill"), dtype=dtype)[()]
    original = data.copy()

    result = from_buffers(format, data, fill)

    assert result.dtype == dtype.newbyteorder("=")
    assert (result.data is data) == (dtype == dtype.newbyteorder("="))
    assert_equal(result.data, original)
    expected = np.full((2, 2), fill, dtype=dtype.newbyteorder("="))
    expected[0, 1], expected[1, 0] = data
    assert_equal(result.todense(), expected)
    assert_equal(result.asformat("coo")["nested"]["values"].todense(), expected["nested"]["values"])
    assert_equal(data, original)
    assert fill.dtype == dtype


def test_coo_non_native_duplicates_and_fill():
    dtype = np.dtype("f8").newbyteorder("S")
    data = np.asarray([1, 2, 3], dtype=dtype)

    result = sparse.COO([[1, 0, 1]], data, shape=(3,), prune=True, fill_value=2)

    assert result.dtype.isnative
    assert_equal(result.todense(), [2, 4, 2])
    assert result.nnz == 1
    assert_equal(data, [1, 2, 3])


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
    data = np.asarray(2, dtype=np.dtype("i4").newbyteorder("S"))

    result = sparse.COO([[0, 2]], data, shape=(3,))

    assert result.dtype.isnative
    assert_equal(result.todense(), [2, 0, 2])
    assert not data.dtype.isnative


@pytest.mark.parametrize("format", ["coo", "csr", "csc"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("func", [sparse.dot, sparse.matmul])
def test_non_native_scipy_contraction(format, side, func):
    dense = np.asarray([[0, 2], [3, 0]], dtype="f8")
    scipy_array = getattr(sps, format + "_array")(dense)
    scipy_array.data = scipy_array.data.astype(np.dtype("f8").newbyteorder("S"))
    scipy_array.data.flags.writeable = False
    other = sparse.COO.from_numpy(dense)
    args = (scipy_array, other) if side == "left" else (other, scipy_array)

    result = func(*args)

    assert_equal(result.todense(), dense @ dense)
    assert result.dtype.isnative
    assert not scipy_array.dtype.isnative
    assert not scipy_array.data.flags.writeable


def test_gcxs_check_dimension_before_byte_order():
    data = np.ones((1, 1), dtype=np.dtype("f8").newbyteorder("S"))
    with pytest.raises(ValueError, match="data must be a scalar or 1-dimensional"):
        sparse.GCXS(
            (data, np.asarray([0]), np.asarray([0, 1])),
            shape=(1, 1),
            compressed_axes=(0,),
            fill_value="invalid",
        )
