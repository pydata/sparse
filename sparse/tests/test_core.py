import pytest

import random
import numpy as np
import scipy.sparse
from sparse import COO

import sparse


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


def assert_eq(x, y):
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert np.allclose(x, y)


@pytest.mark.parametrize('reduction', ['max', 'sum'])
@pytest.mark.parametrize('axis', [None, 0, 1, 2, (0, 2)])
@pytest.mark.parametrize('keepdims', [True, False])
def test_reductions(reduction, axis, keepdims):
    xx = getattr(x, reduction)(axis=axis, keepdims=keepdims)
    yy = getattr(y, reduction)(axis=axis, keepdims=keepdims)
    assert_eq(xx, yy)


@pytest.mark.parametrize('axis', [None, (1, 2, 0), (2, 1, 0), (0, 1, 2)])
def test_transpose(axis):
    xx = x.transpose(axis)
    yy = y.transpose(axis)
    assert_eq(xx, yy)


@pytest.mark.parametrize('a,b', [[(3, 4), (3, 4)],
                                 [(12,), (3, 4)],
                                 [(12,), (3, -1)],
                                 [(3, 4), (12,)],
                                 [(2, 3, 4, 5), (8, 15)],
                                 [(2, 3, 4, 5), (24, 5)],
                                 [(2, 3, 4, 5), (20, 6)],
])
def test_reshape(a, b):
    x = random_x(a)
    s = COO.from_numpy(x)

    assert_eq(x.reshape(b), s.reshape(b))


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

    assert isinstance(sparse.tensordot(sa, b, axes), COO)

    assert_eq(np.tensordot(a, b, axes),
              sparse.tensordot(a, sb, axes))

    assert isinstance(sparse.tensordot(a, sb, axes), COO)


@pytest.mark.xfail
@pytest.mark.parametrize('ufunc', [np.expm1, np.log1p])
def test_ufunc(ufunc):
    x = random_x((2, 3, 4))
    s = COO.from_numpy(x)

    assert isinstance(ufunc(s), COO)

    assert_eq(ufunc(x), ufunc(s))


@pytest.mark.parametrize('index', [
    0,
    1,
    -1,
    (slice(0, 2),),
    (0, slice(0, 2),),
    (slice(0, 1), 0),
    ([1, 0], 0),
    (1, [0, 2]),
    (0, [1, 0], 0),
    (1, [2, 0], 0),
    (None, slice(1, 3), 0),
    (slice(0, 3), None, 0),
    (slice(1, 2), slice(2, 4)),
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

    x = COO(coords, data, shape=(2, 2, 5))
    y = x.sum_duplicates()

    assert_eq(x, y)
    assert x.nnz == 5
    assert x.has_duplicates
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


def test_scalar_exponentiation():
    x = random_x((2, 3, 4))
    a = COO.from_numpy(x)

    assert_eq(x ** 2, a ** 2)
    assert_eq(x ** 0.5, a ** 0.5)

    with pytest.raises((ValueError, ZeroDivisionError)):
        assert_eq(x ** -1, a ** -1)
