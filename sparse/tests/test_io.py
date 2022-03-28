import pytest
import tempfile, pathlib
from hypothesis import settings, given, strategies as st
import numpy as np

import sparse

from sparse import save_npz, load_npz
from sparse._utils import assert_eq


@settings(deadline=None)
@given(
    compression=st.sampled_from([True, False]), format=st.sampled_from(["coo", "gcxs"])
)
def test_save_load_npz_file(compression, format):
    with tempfile.TemporaryDirectory() as tmp_path_str:
        tmp_path = pathlib.Path(tmp_path_str)
        x = sparse.random((2, 3, 4, 5), density=0.25, format=format)
        y = x.todense()

        filename = tmp_path / "mat.npz"
        save_npz(filename, x, compressed=compression)
        z = load_npz(filename)
        assert_eq(x, z)
        assert_eq(y, z.todense())


def test_load_wrong_format_exception():
    with tempfile.TemporaryDirectory() as tmp_path_str:
        tmp_path = pathlib.Path(tmp_path_str)
        x = np.array([1, 2, 3])

        filename = tmp_path / "mat.npz"

        np.savez(filename, x)
        with pytest.raises(RuntimeError):
            load_npz(filename)
