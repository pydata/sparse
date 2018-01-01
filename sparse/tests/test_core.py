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
    x = np.zeros(shape=shape, dtype=dtype)
    for i in range(max(5, np.prod(x.shape) // 10)):
        x[tuple(random.randint(0, d - 1) for d in x.shape)] = random.randint(0, 100)
    return x


def random_x_bool(shape):
    x = np.zeros(shape=shape, dtype=np.bool)
    for i in range(max(5, np.prod(x.shape) // 10)):
        x[tuple(random.randint(0, d - 1) for d in x.shape)] = True
    return x


@pytest.mark.parametrize('reduction,kwargs', [
    ('max', {}),
    ('sum', {}),
    ('sum', {'dtype': np.float16}),
    ('prod', {}),
    ('min', {}),
])
@pytest.mark.parametrize('axis', [None, 0, 1, 2, (0, 2)])
@pytest.mark.parametrize('keepdims', [True, False])
def test_reductions(reduction, axis, keepdims, kwargs):
    xx = getattr(x, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    yy = getattr(y, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    assert_eq(xx, yy)


@pytest.mark.parametrize('reduction,kwargs', [
    (np.max, {}),
    (np.sum, {}),
    (np.sum, {'dtype': np.float16}),
    (np.prod, {}),
    (np.min, {}),
])
@pytest.mark.parametrize('axis', [None, 0, 1, 2, (0, 2)])
@pytest.mark.parametrize('keepdims', [True, False])
def test_ufunc_reductions(reduction, axis, keepdims, kwargs):
    xx = reduction(x, axis=axis, keepdims=keepdims, **kwargs)
    yy = reduction(y, axis=axis, keepdims=keepdims, **kwargs)
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
    [(), ()],
])
def test_reshape(a, b):
    x = random_x(a)
    s = COO.from_numpy(x)

    assert_eq(x.reshape(b), s.reshape(b))


def test_large_reshape():
    n = 100
    m = 10
    row = np.arange(n, dtype=np.uint16)  # np.random.randint(0, n, size=n, dtype=np.uint16)
    col = row % m  # np.random.randint(0, m, size=n, dtype=np.uint16)
    data = np.ones(n, dtype=np.uint8)

    x = COO((data, (row, col)), sorted=True, has_duplicates=False)

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
    [(4,), (4,), 0],
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
    import operator
    a = random_x((3, 4, 5))
    b = random_x((5, 6))

    sa = COO.from_numpy(a)
    sb = COO.from_numpy(b)

    assert_eq(a.dot(b), sa.dot(sb))
    assert_eq(np.dot(a, b), sparse.dot(sa, sb))

    if hasattr(operator, 'matmul'):
        # Basic equivalences
        assert_eq(eval("a @ b"), eval("sa @ sb"))
        assert_eq(eval("sa @ sb"), sparse.dot(sa, sb))

        # Test that SOO's and np.array's combine correctly
        # Not possible due to https://github.com/numpy/numpy/issues/9028
        # assert_eq(eval("a @ sb"), eval("sa @ b"))


@pytest.mark.xfail
def test_dot_nocoercion():
    a = random_x((3, 4, 5))
    b = random_x((5, 6))

    la = a.tolist()
    lb = b.tolist()
    la, lb  # silencing flake8

    sa = COO.from_numpy(a)
    sb = COO.from_numpy(b)
    sa, sb  # silencing flake8

    if hasattr(operator, 'matmul'):
        # Operations with naive collection (list)
        assert_eq(eval("la @ b"), eval("la @ sb"))
        assert_eq(eval("a @ lb"), eval("sa @ lb"))


@pytest.mark.parametrize('func', [np.expm1, np.log1p, np.sin, np.tan,
                                  np.sinh, np.tanh, np.floor, np.ceil,
                                  np.sqrt, np.conj, np.round, np.rint,
                                  lambda x: x.astype('int32'), np.conjugate,
                                  np.conj, lambda x: x.round(decimals=2), abs])
def test_elemwise(func):
    x = random_x((2, 3, 4))
    s = COO.from_numpy(x)

    fs = func(s)

    assert isinstance(fs, COO)

    assert_eq(func(x), fs)


@pytest.mark.parametrize('func', [
    operator.mul, operator.add, operator.sub, operator.gt,
    operator.lt, operator.ne
])
@pytest.mark.parametrize('shape', [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_elemwise_binary(func, shape):
    x = random_x(shape)
    y = random_x(shape)

    xs = COO.from_numpy(x)
    ys = COO.from_numpy(y)

    assert_eq(func(xs, ys), func(x, y))


@pytest.mark.parametrize('func', [
    operator.pow, operator.truediv, operator.floordiv,
    operator.ge, operator.le, operator.eq
])
@pytest.mark.filterwarnings('ignore:divide by zero')
@pytest.mark.filterwarnings('ignore:invalid value')
def test_auto_densification_fails(func):
    xs = COO.from_numpy(random_x((2, 3, 4)))
    ys = COO.from_numpy(random_x((2, 3, 4)))

    with pytest.raises(ValueError):
        func(xs, ys)


def test_op_scipy_sparse():
    x = random_x((3, 4))
    y = random_x((3, 4))

    xs = COO.from_numpy(x)
    ys = scipy.sparse.csr_matrix(y)

    assert_eq(x + y, xs + ys)


@pytest.mark.parametrize('func, scalar', [
    (operator.mul, 5),
    (operator.add, 0),
    (operator.sub, 0),
    (operator.pow, 5),
    (operator.truediv, 3),
    (operator.floordiv, 4),
    (operator.gt, 5),
    (operator.lt, -5),
    (operator.ne, 0),
    (operator.ge, 5),
    (operator.le, -3),
    (operator.eq, 1)
])
def test_elemwise_scalar(func, scalar):
    x = random_x((2, 3, 4))
    y = scalar

    xs = COO.from_numpy(x)
    fs = func(xs, y)

    assert isinstance(fs, COO)
    assert xs.nnz >= fs.nnz

    assert_eq(fs, func(x, y))


@pytest.mark.parametrize('func, scalar', [
    (operator.add, 5),
    (operator.sub, -5),
    (operator.pow, -3),
    (operator.truediv, 0),
    (operator.floordiv, 0),
    (operator.gt, -5),
    (operator.lt, 5),
    (operator.ne, 1),
    (operator.ge, -3),
    (operator.le, 3),
    (operator.eq, 0)
])
@pytest.mark.filterwarnings('ignore:divide by zero')
@pytest.mark.filterwarnings('ignore:invalid value')
def test_scalar_densification_fails(func, scalar):
    xs = COO.from_numpy(random_x((2, 3, 4)))
    y = scalar

    with pytest.raises(ValueError):
        func(xs, y)


@pytest.mark.parametrize('func', [operator.and_, operator.or_, operator.xor])
@pytest.mark.parametrize('shape', [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_bitwise_binary(func, shape):
    x = random_x(shape, dtype=np.int_)
    y = random_x(shape, dtype=np.int_)

    xs = COO.from_numpy(x)
    ys = COO.from_numpy(y)

    assert_eq(func(xs, ys), func(x, y))


@pytest.mark.parametrize('func', [operator.and_, operator.or_, operator.xor])
@pytest.mark.parametrize('shape', [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_bitwise_binary_bool(func, shape):
    x = random_x_bool(shape)
    y = random_x_bool(shape)

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
    (slice(None, 2, -1), slice(None, 2, -1)),
    (slice(1, None, 2), slice(1, None, 2)),
    (slice(None, None, 2),),
    (slice(None, 2, -1), slice(None, 2, -2)),
    (slice(1, None, 2), slice(1, None, 1)),
    (slice(None, None, -2),),
    (0, slice(0, 2),),
    (slice(0, 1), 0),
    ([1, 0], 0),
    (1, [0, 2]),
    (0, [1, 0], 0),
    (1, [2, 0], 0),
    (None, slice(1, 3), 0),
    (Ellipsis, slice(1, 3)),
    (1, Ellipsis, slice(1, 3)),
    (slice(0, 1), Ellipsis),
    (Ellipsis, None),
    (None, Ellipsis),
    (1, Ellipsis),
    (1, Ellipsis, None),
    (1, 1, 1),
    (1, 1, 1, Ellipsis),
    (Ellipsis, 1, None),
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
    ([True, False], slice(1, None), slice(-2, None)),
    (slice(1, None), slice(-2, None), [True, False, True, False]),
])
def test_slicing(index):
    x = random_x((2, 3, 4))
    s = COO.from_numpy(x)

    assert_eq(x[index], s[index])


def test_custom_dtype_slicing():
    dt = np.dtype([('part1', np.float_),
                   ('part2', np.int_, (2,)),
                   ('part3', np.int_, (2, 2))])

    x = np.zeros((2, 3, 4), dtype=dt)
    x[1, 1, 1] = (0.64, [4, 2], [[1, 2], [3, 0]])

    s = COO.from_numpy(x)

    assert x[1, 1, 1] == s[1, 1, 1]
    assert x[0, 1, 2] == s[0, 1, 2]

    assert_eq(x['part1'], s['part1'])
    assert_eq(x['part2'], s['part2'])
    assert_eq(x['part3'], s['part3'])


@pytest.mark.parametrize('index', [
    (Ellipsis, Ellipsis),
    (1, 1, 1, 1),
    (slice(None),) * 4,
    5,
    -5,
    'foo',
    ([True, False, False]),
])
def test_slicing_errors(index):
    x = random_x((2, 3, 4))
    s = COO.from_numpy(x)

    try:
        x[index]
    except Exception as e:
        e1 = e
    else:
        raise Exception("exception not raised")

    try:
        s[index]
    except Exception as e:
        assert type(e) == type(e1)
    else:
        raise Exception("exception not raised")


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


def test_large_concat_stack():
    data = np.array([1], dtype=np.uint8)
    coords = np.array([[255]], dtype=np.uint8)

    xs = COO(coords, data, shape=(256,), has_duplicates=False, sorted=True)
    x = xs.todense()

    assert_eq(np.stack([x, x]),
              sparse.stack([xs, xs]))

    assert_eq(np.concatenate((x, x)),
              sparse.concatenate((xs, xs)))


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


def test_addition_not_ok_when_large_and_sparse():
    x = COO({(0, 0): 1}, shape=(1000000, 1000000))
    with pytest.raises(ValueError):
        x + 1
    with pytest.raises(ValueError):
        1 + x
    with pytest.raises(ValueError):
        1 - x
    with pytest.raises(ValueError):
        x - 1
    with pytest.raises(ValueError):
        np.exp(x)


@pytest.mark.parametrize('func', [operator.add, operator.mul])
@pytest.mark.parametrize('shape1,shape2', [((2, 3, 4), (3, 4)),
                                           ((3, 4), (2, 3, 4)),
                                           ((3, 1, 4), (3, 2, 4)),
                                           ((1, 3, 4), (3, 4)),
                                           ((3, 4, 1), (3, 4, 2)),
                                           ((1, 5), (5, 1))])
def test_broadcasting(func, shape1, shape2):
    x = random_x(shape1)
    a = COO.from_numpy(x)

    z = random_x(shape2)
    c = COO.from_numpy(z)

    expected = func(x, z)
    actual = func(a, c)

    assert_eq(expected, actual)

    assert np.count_nonzero(expected) == actual.nnz


@pytest.mark.parametrize('shape1,shape2', [((3, 4), (2, 3, 4)),
                                           ((3, 1, 4), (3, 2, 4)),
                                           ((3, 4, 1), (3, 4, 2))])
def test_broadcast_to(shape1, shape2):
    x = random_x(shape1)
    a = COO.from_numpy(x)

    assert_eq(np.broadcast_to(x, shape2), a.broadcast_to(shape2))


def test_scalar_multiplication():
    x = random_x((2, 3, 4))
    a = COO.from_numpy(x)

    assert_eq(x * 2, a * 2)
    assert_eq(2 * x, 2 * a)
    assert_eq(x / 2, a / 2)
    assert_eq(x / 2.5, a / 2.5)
    assert_eq(x // 2.5, a // 2.5)


@pytest.mark.filterwarnings('ignore:divide by zero')
def test_scalar_exponentiation():
    x = random_x((2, 3, 4))
    a = COO.from_numpy(x)

    assert_eq(x ** 2, a ** 2)
    assert_eq(x ** 0.5, a ** 0.5)

    with pytest.raises((ValueError, ZeroDivisionError)):
        assert_eq(x ** -1, a ** -1)


def test_create_with_lists_of_tuples():
    L = [((0, 0, 0), 1),
         ((1, 2, 1), 1),
         ((1, 1, 1), 2),
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
    s = COO(x, cache=True)

    assert isinstance(s.tocsr(), scipy.sparse.csr_matrix)
    assert isinstance(s.tocsc(), scipy.sparse.csc_matrix)
    assert s.tocsr() is s.tocsr()
    assert s.tocsc() is s.tocsc()


def test_empty_shape():
    x = COO(np.empty((0, 1), dtype=np.int8), [1.0])
    assert x.shape == ()
    assert ((2 * x).todense() == np.array(2.0)).all()


def test_single_dimension():
    x = COO([1, 3], [1.0, 3.0])
    assert x.shape == (4,)
    assert_eq(x, np.array([0, 1.0, 0, 3.0]))


def test_raise_dense():
    x = COO({(10000, 10000): 1.0})
    with pytest.raises((ValueError, NotImplementedError)) as exc_info:
        np.exp(x)

    assert 'dense' in str(exc_info.value).lower()

    with pytest.raises((ValueError, NotImplementedError)):
        x + 1


def test_large_sum():
    n = 500000
    x = np.random.randint(0, 10000, size=(n,))
    y = np.random.randint(0, 1000, size=(n,))
    z = np.random.randint(0, 3, size=(n,))

    data = np.random.random(n)

    a = COO((x, y, z), data)
    assert a.shape == (10000, 1000, 3)

    b = a.sum(axis=2)
    assert b.nnz > 100000


def test_add_many_sparse_arrays():
    x = COO({(1, 1): 1})
    y = sum([x] * 100)
    assert y.nnz < np.prod(y.shape)


def test_caching():
    x = COO({(10, 10, 10): 1})
    assert x[:].reshape((100, 10)).transpose().tocsr() is not x[:].reshape((100, 10)).transpose().tocsr()

    x = COO({(10, 10, 10): 1}, cache=True)
    assert x[:].reshape((100, 10)).transpose().tocsr() is x[:].reshape((100, 10)).transpose().tocsr()

    x = COO({(1, 1, 1, 1, 1, 1, 1, 2): 1}, cache=True)

    for i in range(x.ndim):
        x.reshape((1,) * i + (2,) + (1,) * (x.ndim - i - 1))

    assert len(x._cache['reshape']) < 5


def test_scalar_slicing():
    x = np.array([0, 1])
    s = COO(x)
    assert np.isscalar(s[0])
    assert_eq(x[0], s[0])

    assert isinstance(s[0, ...], COO)
    assert s[0, ...].shape == ()
    assert_eq(x[0, ...], s[0, ...])

    assert np.isscalar(s[1])
    assert_eq(x[1], s[1])

    assert isinstance(s[1, ...], COO)
    assert s[1, ...].shape == ()
    assert_eq(x[1, ...], s[1, ...])


@pytest.mark.parametrize('shape, k', [
    ((3, 4), 0),
    ((3, 4, 5), 1),
    ((4, 2), -1),
    ((2, 4), -2),
    ((4, 4), 1000),
])
def test_triul(shape, k):
    x = random_x(shape)
    s = COO.from_numpy(x)

    assert_eq(np.triu(x, k), sparse.triu(s, k))
    assert_eq(np.tril(x, k), sparse.tril(s, k))


def test_empty_reduction():
    x = np.zeros((2, 3, 4), dtype=np.float_)
    xs = COO.from_numpy(x)

    assert_eq(x.sum(axis=(0, 2)),
              xs.sum(axis=(0, 2)))
