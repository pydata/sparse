import sparse
from sparse import load_npz, save_npz
from sparse.numba_backend._utils import assert_eq

import pytest

import numpy as np


@pytest.mark.parametrize("compression", [True, False])
@pytest.mark.parametrize("format", ["coo", "gcxs"])
def test_save_load_npz_file(tmp_path, compression, format):
    x = sparse.random((2, 3, 4, 5), density=0.25, format=format)
    y = x.todense()

    filename = tmp_path / "mat.npz"
    save_npz(filename, x, compressed=compression)
    z = load_npz(filename)
    assert_eq(x, z)
    assert_eq(y, z.todense())


def test_load_wrong_format_exception(tmp_path):
    x = np.array([1, 2, 3])

    filename = tmp_path / "mat.npz"

    np.savez(filename, x)
    with pytest.raises(RuntimeError):
        load_npz(filename)
