import os
import tempfile
import shutil
import pytest
import numpy as np

import sparse

from sparse.io import save_npz, load_npz
from sparse.utils import assert_eq


def test_save_load_npz_file():
    x = sparse.random((2, 3, 4, 5), density=.25)
    y = x.todense()

    dir_name = tempfile.mkdtemp()
    filename = os.path.join(dir_name, 'mat.npz')

    # with compression
    save_npz(filename, x, compressed=True)
    z = load_npz(filename)
    assert_eq(x, z)
    assert_eq(y, z.todense())

    # without compression
    save_npz(filename, x, compressed=False)
    z = load_npz(filename)
    assert_eq(x, z)
    assert_eq(y, z.todense())

    # test exception on wrong format
    np.savez(filename, y)
    with pytest.raises(RuntimeError):
        load_npz(filename)

    shutil.rmtree(dir_name)
