import os
import tempfile
import shutil
import pytest
import numpy as np

import sparse

from sparse import save_npz, load_npz
from sparse.utils import assert_eq


@pytest.mark.parametrize('compression', [True, False])
def test_save_load_npz_file(compression):
    x = sparse.random((2, 3, 4, 5), density=.25)
    y = x.todense()

    dir_name = tempfile.mkdtemp()
    filename = os.path.join(dir_name, 'mat.npz')

    save_npz(filename, x, compressed=compression)
    z = load_npz(filename)
    assert_eq(x, z)
    assert_eq(y, z.todense())

    shutil.rmtree(dir_name)


def test_load_wrong_format_exception():
    x = np.array([1, 2, 3])

    dir_name = tempfile.mkdtemp()
    filename = os.path.join(dir_name, 'mat.npz')

    np.savez(filename, x)
    with pytest.raises(RuntimeError):
        load_npz(filename)

    shutil.rmtree(dir_name)
