import sparse
import pytest
import numpy as np
import scipy

from sparse._compressed import GXCS
from sparse._utils import assert_eq


@pytest.mark.parametrize('a,b', [
    [(3, 4), (5, 5)],
    [(12,), (3, 4)],
    [(12,), (3, 6)],
    [(5, 5, 5), (6, 6, 6)],
    [(3, 4), (9, 4)],
    [(5,), (4,)],
    [(2, 3, 4, 5), (2, 3, 4, 5, 6)],
    [(100,), (5, 5)],
    [(2, 3, 4, 5), (20, 6)],
    [(), ()],
])
def test_resize(a, b):
    s = sparse.random(a, density=0.5, format='gxcs')
    orig_size = s.size
    x = s.todense()
    x = np.resize(x, b)
    s.resize(b)
    temp = x.reshape(x.size)
    temp[orig_size:] = s.fill_value
    assert isinstance(s, sparse.SparseArray)
    assert_eq(x, s)


@pytest.mark.parametrize('a,b', [
    [(3, 4), (3, 4)],
    [(12,), (3, 4)],
    [(12,), (3, -1)],
    [(3, 4), (12,)],
    [(3, 4), (-1, 4)],
    [(3, 4), (3, -1)],
    [(2, 3, 4, 5), (8, 15)],
    [(2, 3, 4, 5), (24, 5)],
    [(2, 3, 4, 5), (20, 6)],
    [(), ()],
])
def test_reshape(a, b):
    s = sparse.random(a, density=0.5, format='gxcs')
    x = s.todense()

    assert_eq(x.reshape(b), s.reshape(b))


def test_reshape_same():
    s = sparse.random((3, 5), density=0.5, format='gxcs')

    assert s.reshape(s.shape) is s


def test_to_scipy_sparse():
    s = sparse.random((3, 5), density=0.5, format='gxcs', compressed_axes=(0,))
    a = s.to_scipy_sparse()
    b = scipy.sparse.csr_matrix(s.todense())

    assert_eq(a, b)


def test_tocoo():
    coo = sparse.random((5, 6), density=.5)
    b = GXCS.from_coo(coo)

    assert_eq(b.tocoo(), coo)


@pytest.mark.parametrize('index', [
    # Integer
    0,
    1,
    -1,
    (1, 1, 1),
    # Pure slices
    (slice(0, 2),),
    (slice(None, 2), slice(None, 2)),
    (slice(1, None), slice(1, None)),
    (slice(None, None),),
    (slice(None, None, -1),),
    (slice(None, 2, -1), slice(None, 2, -1)),
    (slice(1, None, 2), slice(1, None, 2)),
    (slice(None, None, 2),),
    (slice(None, 2, -1), slice(None, 2, -2)),
    (slice(1, None, 2), slice(1, None, 1)),
    (slice(None, None, -2),),
    # Combinations
    (0, slice(0, 2),),
    (slice(0, 1), 0),
    (None, slice(1, 3), 0),
    (slice(0, 3), None, 0),
    (slice(1, 2), slice(2, 4)),
    (slice(1, 2), slice(None, None)),
    (slice(1, 2), slice(None, None), 2),
    (slice(1, 2, 2), slice(None, None), 2),
    (slice(1, 2, None), slice(None, None, 2), 2),
    (slice(1, 2, -2), slice(None, None), -2),
    (slice(1, 2, None), slice(None, None, -2), 2),
    (slice(1, 2, -1), slice(None, None), -1),
    (slice(1, 2, None), slice(None, None, -1), 2),
    (slice(2, 0, -1), slice(None, None), -1),
    (slice(-2, None, None),),
    (slice(-1, None, None), slice(-2, None, None)),
    # With ellipsis
    (Ellipsis, slice(1, 3)),
    (1, Ellipsis, slice(1, 3)),
    (slice(0, 1), Ellipsis),
    (Ellipsis, None),
    (None, Ellipsis),
    (1, Ellipsis),
    (1, Ellipsis, None),
    (1, 1, 1, Ellipsis),
    (Ellipsis, 1, None),
    # Pathological - Slices larger than array
    (slice(None, 1000)),
    (slice(None), slice(None, 1000)),
    (slice(None), slice(1000, -1000, -1)),
    (slice(None), slice(1000, -1000, -50)),
    # Pathological - Wrong ordering of start/stop
    (slice(5, 0),),
    (slice(0, 5, -1),),
])
def test_slicing(index):
    s = sparse.random((2, 3, 4), density=0.5, format='gxcs')
    x = s.todense()

    assert_eq(x[index], s[index])


@pytest.mark.parametrize('index', [
    ([1, 0], 0),
    (1, [0, 2]),
    (0, [1, 0], 0),
    (1, [2, 0], 0),
    ([True, False], slice(1, None), slice(-2, None)),
    (slice(1, None), slice(-2, None), [True, False, True, False]),
    ([1, 0],),
    (Ellipsis, [2, 1, 3],),
    (slice(None), [2, 1, 2],),
    (1, [2, 0, 1],),
])
def test_advanced_indexing(index):
    s = sparse.random((2, 3, 4), density=0.5, format='gxcs')
    x = s.todense()

    assert_eq(x[index], s[index])


@pytest.mark.parametrize('index', [
    (Ellipsis, Ellipsis),
    (1, 1, 1, 1),
    (slice(None),) * 4,
    5,
    -5,
    'foo',
    [True, False, False],
    0.5,
    [0.5],
    {'potato': 'kartoffel'},
    ([[0, 1]],),
])
def test_slicing_errors(index):
    s = sparse.random((2, 3, 4), density=0.5, format='gxcs')

    with pytest.raises(IndexError):
        s[index]


def test_change_compressed_axes():
    coo = sparse.random((3, 4, 5), density=.5)
    s = GXCS.from_coo(coo, compressed_axes=(0, 1))
    b = GXCS.from_coo(coo, compressed_axes=(1, 2))

    s.change_compressed_axes((1, 2))

    assert_eq(s, b)
