import pytest

import random
import operator
import numpy as np
import scipy.sparse
from sparse import COO

import sparse
from sparse.utils import assert_eq


x = np.zeros(shape=(2, 3, 4), dtype=np.float32)
for i in range(10):
    x[random.randint(0, x.shape[0] - 1),
      random.randint(0, x.shape[1] - 1),
      random.randint(0, x.shape[2] - 1)] = random.randint(0, 100)
y = COO.from_numpy(x)


def random_x(shape, dtype=float):
    x = np.zeros(shape=shape, dtype=float)
    for i in range(max(5, np.prod(x.shape) // 10)):
        x[tuple(random.randint(0, d - 1) for d in x.shape)] = random.randint(0, 100)
    return x


@pytest.mark.parametrize('reduction,kwargs', [
    ('max', {}),
    ('sum', {}),
    ('sum', {'dtype': np.float16})
])
@pytest.mark.parametrize('axis', [None, 0, 1, 2, (0, 2)])
@pytest.mark.parametrize('keepdims', [True, False])
def test_reductions(reduction, axis, keepdims, kwargs):
    xx = getattr(x, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    yy = getattr(y, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    assert_eq(xx, yy)


@pytest.mark.parametrize('axis', [None, (1, 2, 0), (2, 1, 0), (0, 1, 2)])
def test_transpose(axis):
    xx = x.transpose(axis)
    yy = y.transpose(axis)
    assert_eq(xx, yy)


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
])
def test_reshape(a, b):
    x = random_x(a)
    s = COO.from_numpy(x)

    assert_eq(x.reshape(b), s.reshape(b))


def test_large_reshape():
    n = 100
    m = 10
    row = np.arange(n, dtype=np.uint16)# np.random.randint(0, n, size=n, dtype=np.uint16)
    col = row % m # np.random.randint(0, m, size=n, dtype=np.uint16)
    data = np.ones(n, dtype=np.uint8)

    x = COO((data, (row, col)))

    assert_eq(x, x.reshape(x.shape))


def test_reshape_same():
    x = random_x((3, 5))
    s = COO.from_numpy(x)

    assert s.reshape(s.shape) is s


def test_to_scipy_sparse():
    x = random_x((3, 5))
    s = COO.from_numpy(x)
    a = s.to_scipy_sparse()
    b = scipy.sparse.coo_matrix(x)

    assert_eq(a.data, b.data)
    assert_eq(a.todense(), b.todense())


@pytest.mark.parametrize('a_shape,b_shape,axes', [
    [(3, 4), (4, 3), (1, 0)],
    [(3, 4), (4, 3), (0, 1)],
    [(3, 4, 5), (4, 3), (1, 0)],
    [(3, 4), (5, 4, 3), (1, 1)],
    [(3, 4), (5, 4, 3), ((0, 1), (2, 1))],
    [(3, 4), (5, 4, 3), ((1, 0), (1, 2))],
    [(3, 4, 5), (4,), (1, 0)],
    [(4,), (3, 4, 5), (0, 1)],
    [(4,), (4,), (0, 0)],
])
def test_tensordot(a_shape, b_shape, axes):
    a = random_x(a_shape)
    b = random_x(b_shape)

    sa = COO.from_numpy(a)
    sb = COO.from_numpy(b)

    assert_eq(np.tensordot(a, b, axes),
              sparse.tensordot(sa, sb, axes))

    assert_eq(np.tensordot(a, b, axes),
              sparse.tensordot(sa, b, axes))

    # assert isinstance(sparse.tensordot(sa, b, axes), COO)

    assert_eq(np.tensordot(a, b, axes),
              sparse.tensordot(a, sb, axes))

    # assert isinstance(sparse.tensordot(a, sb, axes), COO)


def test_dot():
    a = random_x((3, 4, 5))
    b = random_x((5, 6))

    sa = COO.from_numpy(a)
    sb = COO.from_numpy(b)

    assert_eq(a.dot(b), sa.dot(sb))
    assert_eq(np.dot(a, b), sparse.dot(sa, sb))


@pytest.mark.parametrize('func', [np.expm1, np.log1p, np.sin, np.tan,
                                   np.sinh,  np.tanh, np.floor, np.ceil,
                                   np.sqrt, np.conj, np.round, np.rint,
                                   lambda x: x.astype('int32'), np.conjugate,
                                   lambda x: x.round(decimals=2), abs])
def test_elemwise(func):
    x = random_x((2, 3, 4))
    s = COO.from_numpy(x)

    assert isinstance(func(s), COO)

    assert_eq(func(x), func(s))


@pytest.mark.parametrize('func', [operator.mul])
@pytest.mark.parametrize('shape', [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_elemwise_binary(func, shape):
    x = random_x(shape)
    y = random_x(shape)

    xs = COO.from_numpy(x)
    ys = COO.from_numpy(y)

    assert_eq(func(xs, ys), func(x, y))


def test_elemwise_binary_empty():
    x = COO({}, shape=(10, 10))
    y = COO.from_numpy(random_x((10, 10)))

    for z in [x * y, y * x]:
        assert z.nnz == 0
        assert z.coords.shape == (2, 0)
        assert z.data.shape == (0,)


def test_gt():
    x = random_x((2, 3, 4))
    s = COO.from_numpy(x)

    m = x.mean()
    assert_eq(x > m, s > m)

    m = s.data[2]
    assert_eq(x > m, s > m)
    assert_eq(x >= m, s >= m)


@pytest.mark.parametrize('index', [
    0,
    1,
    -1,
    (slice(0, 2),),
    (slice(None, 2), slice(None, 2)),
    (slice(1, None), slice(1, None)),
    (slice(None, None),),
    (0, slice(0, 2),),
    (slice(0, 1), 0),
    ([1, 0], 0),
    (1, [0, 2]),
    (0, [1, 0], 0),
    (1, [2, 0], 0),
    (None, slice(1, 3), 0),
    (slice(0, 3), None, 0),
    (slice(1, 2), slice(2, 4)),
    (slice(1, 2), slice(None, None)),
    (slice(1, 2), slice(None, None), 2),
])
def test_slicing(index):
    x = random_x((2, 3, 4))
    s = COO.from_numpy(x)

    assert_eq(x[index], s[index])


def test_canonical():
    coords = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [1, 0, 3],
                       [0, 1, 0],
                       [1, 0, 3]]).T
    data = np.arange(5)

    old = COO(coords, data, shape=(2, 2, 5))
    x = COO(coords, data, shape=(2, 2, 5))
    y = x.sum_duplicates()

    assert_eq(old, y)
    # assert x.nnz == 5
    # assert x.has_duplicates
    assert y.nnz == 3
    assert not y.has_duplicates


def test_concatenate():
    x = random_x((2, 3, 4))
    xx = COO.from_numpy(x)
    y = random_x((5, 3, 4))
    yy = COO.from_numpy(y)
    z = random_x((4, 3, 4))
    zz = COO.from_numpy(z)

    assert_eq(np.concatenate([x, y, z], axis=0),
              sparse.concatenate([xx, yy, zz], axis=0))

    x = random_x((5, 3, 1))
    xx = COO.from_numpy(x)
    y = random_x((5, 3, 3))
    yy = COO.from_numpy(y)
    z = random_x((5, 3, 2))
    zz = COO.from_numpy(z)

    assert_eq(np.concatenate([x, y, z], axis=2),
              sparse.concatenate([xx, yy, zz], axis=2))

    assert_eq(np.concatenate([x, y, z], axis=-1),
              sparse.concatenate([xx, yy, zz], axis=-1))


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('func', ['stack', 'concatenate'])
def test_concatenate_mixed(func, axis):
    d = random_x((10, 10))
    d[d < 0.9] = 0
    s = COO.from_numpy(d)

    result = getattr(sparse, func)([d, s, s], axis=axis)
    expected = getattr(np, func)([d, d, d], axis=axis)

    assert isinstance(result, COO)

    assert_eq(result, expected)


@pytest.mark.parametrize('shape', [(5,), (2, 3, 4), (5, 2)])
@pytest.mark.parametrize('axis', [0, 1, -1])
def test_stack(shape, axis):
    x = random_x(shape)
    xx = COO.from_numpy(x)
    y = random_x(shape)
    yy = COO.from_numpy(y)
    z = random_x(shape)
    zz = COO.from_numpy(z)

    assert_eq(np.stack([x, y, z], axis=axis),
              sparse.stack([xx, yy, zz], axis=axis))


def test_coord_dtype():
    x = random_x((2, 3, 4))
    s = COO.from_numpy(x)
    assert s.coords.dtype == np.uint8

    x = np.zeros(1000)
    s = COO.from_numpy(x)
    assert s.coords.dtype == np.uint16


def test_addition():
    x = random_x((2, 3, 4))
    a = COO.from_numpy(x)

    y = random_x((2, 3, 4))
    b = COO.from_numpy(y)

    assert_eq(x + y, a + b)
    assert_eq(x - y, a - b)
    assert_eq(-x, -a)


def test_addition_ok_when_mostly_dense():
    x = np.arange(10)
    y = COO.from_numpy(x)

    assert_eq(x + 1, y + 1)
    assert_eq(x - 1, y - 1)
    assert_eq(1 - x, 1 - y)
    assert_eq(np.exp(x), np.exp(y))


def test_addition_not_ok_when_large_and_sparse():
    x = COO({(0, 0): 1}, shape=(1000000, 1000000))
    with pytest.raises(Exception):
        x + 1
    with pytest.raises(Exception):
        1 + x
    with pytest.raises(Exception):
        1 - x
    with pytest.raises(Exception):
        x - 1
    with pytest.raises(Exception):
        np.exp(x)


@pytest.mark.xfail
def test_addition_broadcasting():
    x = random_x((2, 3, 4))
    a = COO.from_numpy(x)

    z = random_x((3, 4))
    c = COO.from_numpy(z)

    assert_eq(x + z, a + c)


def test_scalar_multiplication():
    x = random_x((2, 3, 4))
    a = COO.from_numpy(x)

    assert_eq(x * 2, a * 2)
    assert_eq(2 * x, 2 * a)
    assert_eq(x / 2, a / 2)
    assert_eq(x / 2.5, a / 2.5)
    assert_eq(x // 2.5, a // 2.5)


def test_scalar_exponentiation():
    x = random_x((2, 3, 4))
    a = COO.from_numpy(x)

    assert_eq(x ** 2, a ** 2)
    assert_eq(x ** 0.5, a ** 0.5)

    with pytest.raises((ValueError, ZeroDivisionError)):
        assert_eq(x ** -1, a ** -1)


def test_create_with_lists_of_tuples():
    L = [((0, 0, 0), 1),
         ((1, 1, 1), 2),
         ((1, 2, 1), 1),
         ((1, 3, 2), 3)]

    s = COO(L)

    x = np.zeros((2, 4, 3), dtype=np.asarray([1, 2, 3]).dtype)
    for ind, value in L:
        x[ind] = value

    assert_eq(s, x)


def test_sizeof():
    import sys
    x = np.eye(100)
    y = COO.from_numpy(x)

    nb = sys.getsizeof(y)
    assert 400 < nb < x.nbytes / 10


def test_scipy_sparse_interface():
    n = 100
    m = 10
    row = np.random.randint(0, n, size=n, dtype=np.uint16)
    col = np.random.randint(0, m, size=n, dtype=np.uint16)
    data = np.ones(n, dtype=np.uint8)

    inp = (data, (row, col))

    x = scipy.sparse.coo_matrix(inp)
    xx = sparse.COO(inp)

    assert_eq(x, xx)
    assert_eq(x.T, xx.T)
    assert_eq(xx.to_scipy_sparse(), x)
    assert_eq(COO.from_scipy_sparse(xx.to_scipy_sparse()), xx)

    assert_eq(x, xx)
    assert_eq(x.T.dot(x), xx.T.dot(xx))


def test_cache_csr():
    x = random_x((10, 5))
    s = COO.from_numpy(x)

    assert isinstance(s.tocsr(), scipy.sparse.csr_matrix)
    assert isinstance(s.tocsc(), scipy.sparse.csc_matrix)
    assert s.tocsr() is s.tocsr()
    assert s.tocsc() is s.tocsc()

    st = s.T

    assert_eq(st._csr, st)
    assert_eq(st._csc, st)

    assert isinstance(st.tocsr(), scipy.sparse.csr_matrix)
    assert isinstance(st.tocsc(), scipy.sparse.csc_matrix)
