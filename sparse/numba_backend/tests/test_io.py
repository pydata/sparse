import sparse
from sparse import COO, GCXS, load_npz, save_npz
from sparse.numba_backend._utils import assert_eq

import pytest

import numpy as np


@pytest.mark.parametrize("compression", [True, False])
@pytest.mark.parametrize("format", ["coo", "gcxs", "csr", "csc"])
def test_save_load_npz_formats(tmp_path, compression, format):
    x = sparse.random((4, 5), density=0.25, format=format)
    y = x.todense()

    filename = tmp_path / "mat.npz"
    save_npz(filename, x, compressed=compression)
    z = load_npz(filename)

    if format in ["coo", "gcxs"]:
        assert type(x) is type(z)
    else:
        assert isinstance(z, GCXS)
    assert_eq(x, z)
    assert_eq(y, z.todense())
    assert x.fill_value == z.fill_value


@pytest.mark.parametrize(
    "shape, format",
    [
        ((), "coo"),
        ((10,), "coo"),
        ((10,), "gcxs"),
        ((2, 3, 4), "coo"),
        ((2, 3, 4), "gcxs"),
    ],
)
def test_save_load_npz_dimensions(tmp_path, shape, format):
    sparse_arr = COO.from_numpy(np.array(5.0)) if shape == () else sparse.random(shape, density=0.3, format=format)

    filename = tmp_path / "mat_dim.npz"
    save_npz(filename, sparse_arr)
    z = load_npz(filename)

    assert type(sparse_arr) is type(z)
    assert_eq(sparse_arr, z)
    assert_eq(sparse_arr.todense(), z.todense())


@pytest.mark.parametrize("format", ["coo", "gcxs", "csr", "csc"])
@pytest.mark.parametrize("fill_value", [1.5, np.nan])
def test_save_load_npz_fill_value(tmp_path, format, fill_value):
    x = sparse.random((4, 5), density=0.3, format=format, fill_value=fill_value)
    filename = tmp_path / "mat_fv.npz"
    save_npz(filename, x)
    z = load_npz(filename)

    if format in ["coo", "gcxs"]:
        assert type(x) is type(z)
    else:
        assert isinstance(z, GCXS)
    assert_eq(x, z)
    if np.isnan(fill_value):
        assert np.isnan(z.fill_value)
    else:
        assert z.fill_value == fill_value


def test_load_legacy_coo_archive(tmp_path):
    # Archive without "format" key, only COO keys
    filename = tmp_path / "legacy_coo.npz"
    coords = np.array([[0, 1], [1, 2]], dtype=np.intp)
    data = np.array([3.0, 4.0], dtype=np.float64)
    shape = (3, 4)
    fill_value = 0.0

    np.savez(filename, coords=coords, data=data, shape=shape, fill_value=fill_value)
    z = load_npz(filename)

    assert isinstance(z, COO)
    assert z.shape == shape
    assert_eq(z, COO(coords, data, shape=shape, fill_value=fill_value))


def test_load_legacy_gcxs_archive(tmp_path):
    # Archive without "format" key, only GCXS keys
    filename = tmp_path / "legacy_gcxs.npz"
    g = GCXS.from_numpy(np.array([[1.0, 0.0], [0.0, 2.0]]))

    np.savez(
        filename,
        data=g.data,
        indices=g.indices,
        indptr=g.indptr,
        compressed_axes=g.compressed_axes,
        shape=g.shape,
        fill_value=g.fill_value,
    )
    z = load_npz(filename)

    assert isinstance(z, GCXS)
    assert_eq(z, g)


def test_load_unknown_format_exception(tmp_path):
    filename = tmp_path / "unknown_fmt.npz"
    np.savez(filename, format="unknown_format", data=np.array([1, 2]))
    with pytest.raises(RuntimeError, match="does not contain a valid sparse matrix"):
        load_npz(filename)


def test_load_corrupted_archive_exception(tmp_path):
    filename = tmp_path / "corrupted.npz"
    np.savez(filename, a=np.array([1, 2, 3]))
    with pytest.raises(RuntimeError, match="does not contain a valid sparse matrix"):
        load_npz(filename)


@pytest.mark.parametrize("format", ["coo", "csr", "csc", "gcxs"])
def test_load_corrupted_tagged_archive_exception(tmp_path, format):
    filename = tmp_path / f"corrupted_{format}.npz"
    np.savez(filename, format=format)
    with pytest.raises(RuntimeError, match="does not contain a valid sparse matrix"):
        load_npz(filename)


def test_save_invalid_type_exception(tmp_path):
    filename = tmp_path / "invalid_type.npz"
    with pytest.raises(ValueError, match="Cannot save array of type"):
        save_npz(filename, [1, 2, 3])

    with pytest.raises(ValueError, match="Cannot save array of type"):
        save_npz(filename, np.array([1, 2, 3]))

    # DOK is not serializable by design
    d = sparse.DOK((3, 3))
    with pytest.raises(ValueError, match="Cannot save array of type DOK"):
        save_npz(filename, d)
