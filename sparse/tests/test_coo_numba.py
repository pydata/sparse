import sparse

import numba

import numpy as np


@numba.njit
def identity(x):
    """Pass an object through numba and back"""
    return x


def identity_constant(x):
    @numba.njit
    def get_it():
        """Pass an object through numba and back as a constant"""
        return x

    return get_it()


def assert_coo_equal(c1, c2):
    assert c1.shape == c2.shape
    assert sparse.all(c1 == c2)
    assert c1.data.dtype == c2.data.dtype
    assert c1.fill_value == c2.fill_value


def assert_coo_same_memory(c1, c2):
    assert_coo_equal(c1, c2)
    assert c1.coords.data == c2.coords.data
    assert c1.data.data == c2.data.data


class TestBasic:
    """Test very simple construction and field access"""

    def test_roundtrip(self):
        c1 = sparse.COO(np.eye(3), fill_value=1)
        c2 = identity(c1)
        assert type(c1) is type(c2)
        assert_coo_same_memory(c1, c2)

    def test_roundtrip_constant(self):
        c1 = sparse.COO(np.eye(3), fill_value=1)
        c2 = identity_constant(c1)
        # constants are always copies
        assert_coo_equal(c1, c2)

    def test_unpack_attrs(self):
        @numba.njit
        def unpack(c):
            return c.coords, c.data, c.shape, c.fill_value

        c1 = sparse.COO(np.eye(3), fill_value=1)
        coords, data, shape, fill_value = unpack(c1)
        c2 = sparse.COO(coords, data, shape, fill_value=fill_value)
        assert_coo_same_memory(c1, c2)

    def test_repack_attrs(self):
        @numba.njit
        def pack(coords, data, shape):
            return sparse.COO(coords, data, shape)

        # repacking fill_value isn't possible yet
        c1 = sparse.COO(np.eye(3))
        c2 = pack(c1.coords, c1.data, c1.shape)
        assert_coo_same_memory(c1, c2)
