import pytest

import numpy as np
import six

import sparse
from sparse import DOK
from sparse.utils import assert_eq


@pytest.mark.parametrize('shape', [
    (2,),
    (2, 3),
    (2, 3, 4),
])
@pytest.mark.parametrize('density', [
    0.1, 0.3, 0.5, 0.7
])
def test_random_shape_nnz(shape, density):
    s = sparse.random(shape, density, format='dok')

    assert isinstance(s, DOK)

    assert s.shape == shape
    expected_nnz = density * np.prod(shape)
    assert np.floor(expected_nnz) <= s.nnz <= np.ceil(expected_nnz)


def test_convert_to_coo():
    s1 = sparse.random((2, 3, 4), 0.5, format='dok')
    s2 = sparse.COO(s1)

    assert_eq(s1, s2)


def test_convert_from_coo():
    s1 = sparse.random((2, 3, 4), 0.5, format='coo')
    s2 = DOK(s1)

    assert_eq(s1, s2)


def test_convert_from_numpy():
    x = np.random.rand(2, 3, 4)
    s = DOK(x)

    assert_eq(x, s)


def test_convert_to_numpy():
    s = sparse.random((2, 3, 4), 0.5, format='dok')
    x = s.todense()

    assert_eq(x, s)


@pytest.mark.parametrize('shape, data', [
    (2, {
        0: 1
    }),
    ((2, 3), {
        (0, 1): 3,
        (1, 2): 4,
    }),
    ((2, 3, 4), {
        (0, 1): 3,
        (1, 2, 3): 4,
        (1, 1): [6, 5, 4, 1]
    }),
])
def test_construct(shape, data):
    s = DOK(shape, data)
    x = np.zeros(shape, dtype=s.dtype)

    for c, d in six.iteritems(data):
        x[c] = d

    assert_eq(x, s)


@pytest.mark.parametrize('shape', [
    (2,),
    (2, 3),
    (2, 3, 4),
])
@pytest.mark.parametrize('density', [
    0.1, 0.3, 0.5, 0.7
])
def test_getitem(shape, density):
    s = sparse.random(shape, density, format='dok')
    x = s.todense()

    for _ in range(s.nnz):
        idx = np.random.randint(np.prod(shape))
        idx = np.unravel_index(idx, shape)

        assert np.isclose(s[idx], x[idx])


@pytest.mark.parametrize('shape, index, value', [
    ((2,), slice(None), np.random.rand()),
    ((2,), slice(1, 2), np.random.rand()),
    ((2,), slice(0, 2), np.random.rand(2)),
    ((2,), 1, np.random.rand()),
    ((2, 3), (0, slice(None)), np.random.rand()),
    ((2, 3), (0, slice(1, 3)), np.random.rand()),
    ((2, 3), (1, slice(None)), np.random.rand(3)),
    ((2, 3), (0, slice(1, 3)), np.random.rand(2)),
    ((2, 3), (0, slice(2, 0, -1)), np.random.rand(2)),
    ((2, 3), (slice(None), 1), np.random.rand()),
    ((2, 3), (slice(None), 1), np.random.rand(2)),
    ((2, 3), (slice(1, 2), 1), np.random.rand()),
    ((2, 3), (slice(1, 2), 1), np.random.rand(1)),
    ((2, 3), (0, 2), np.random.rand()),
])
def test_setitem(shape, index, value):
    s = sparse.random(shape, 0.5, format='dok')
    x = s.todense()

    s[index] = value
    x[index] = value

    assert_eq(x, s)


def test_default_dtype():
    s = DOK((5,))

    assert s.dtype == np.float64


def test_int_dtype():
    data = {
        1: np.uint8(1),
        2: np.uint16(2),
    }

    s = DOK((5,), data)

    assert s.dtype == np.uint16


def test_float_dtype():
    data = {
        1: np.uint8(1),
        2: np.float32(2),
    }

    s = DOK((5,), data)

    assert s.dtype == np.float32


def test_set_zero():
    s = DOK((1,), dtype=np.uint8)
    s[0] = 1
    s[0] = 0

    assert s[0] == 0
    assert s.nnz == 0


@pytest.mark.parametrize('format', [
    'coo',
    'dok',
])
def test_asformat(format):
    s = sparse.random((2, 3, 4), density=0.5, format='dok')
    s2 = s.asformat(format)

    assert_eq(s, s2)
