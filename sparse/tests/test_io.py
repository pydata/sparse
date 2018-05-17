import os
import tempfile
import shutil

import sparse

from sparse.io import save_npz, load_npz
from sparse.utils import assert_eq


def test_save_load_npz_file():
    x = sparse.random((2, 3, 4, 5), density=.25)
    y = x.todense()

    dir_name = tempfile.mkdtemp()
    filename = os.path.join(dir_name, 'mat.npz')

    save_npz(filename, x)
    z = load_npz(filename)

    assert_eq(y, z.todense())

    shutil.rmtree(dir_name)
