import pytest

from packaging import version
import operator
import numpy as np
import scipy.sparse
import scipy.stats
from sparse import COO

import sparse
from sparse.utils import assert_eq, is_lexsorted, random_value_array


@pytest.mark.parametrize('reduction,kwargs,eqkwargs', [
    ('max', {}, {}),
    ('sum', {}, {}),
    ('sum', {'dtype': np.float16}, {'atol': 1e-2}),
    ('prod', {}, {}),
    ('min', {}, {}),
])
@pytest.mark.parametrize('axis', [None, 0, 1, 2, (0, 2)])
@pytest.mark.parametrize('keepdims', [True, False])
def test_reductions(reduction, axis, keepdims, kwargs, eqkwargs):
    x = sparse.random((2, 3, 4), density=.25)
    y = x.todense()
    xx = getattr(x, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    yy = getattr(y, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    assert_eq(xx, yy, check_nnz=False, **eqkwargs)


@pytest.mark.parametrize('reduction,kwargs,eqkwargs', [
    (np.max, {}, {}),
    (np.sum, {}, {}),
    (np.sum, {'dtype': np.float16}, {'atol': 1e-2}),
    (np.prod, {}, {}),
    (np.min, {}, {}),
])
@pytest.mark.parametrize('axis', [None, 0, 1, 2, (0, 2)])
@pytest.mark.parametrize('keepdims', [True, False])
def test_ufunc_reductions(reduction, axis, keepdims, kwargs, eqkwargs):
    x = sparse.random((2, 3, 4), density=.5)
    y = x.todense()
    xx = reduction(x, axis=axis, keepdims=keepdims, **kwargs)
    yy = reduction(y, axis=axis, keepdims=keepdims, **kwargs)
    assert_eq(xx, yy, check_nnz=False, **eqkwargs)


@pytest.mark.parametrize('reduction', [
    'nansum',
    'nanprod',
    'nanmax',
    'nanmin',
])
@pytest.mark.parametrize('axis', [None, 0, 1])
@pytest.mark.parametrize('keepdims', [False])
@pytest.mark.parametrize('fraction', [0.25, 0.5, 0.75, 1.0])
@pytest.mark.filterwarnings('ignore:All-NaN')
def test_nan_reductions(reduction, axis, keepdims, fraction):
    s = sparse.random((2, 3, 4), data_rvs=random_value_array(np.nan, fraction),
                      density=.25)
    x = s.todense()
    expected = getattr(np, reduction)(x, axis=axis, keepdims=keepdims)
    actual = getattr(sparse, reduction)(s, axis=axis, keepdims=keepdims)
    assert_eq(expected, actual, equal_nan=True, check_nnz=False)


@pytest.mark.parametrize('reduction', [
    'nanmax',
    'nanmin',
])
@pytest.mark.parametrize('axis', [None, 0, 1])
def test_all_nan_reduction_warning(reduction, axis):
    x = random_value_array(np.nan, 1.0)(2 * 3 * 4).reshape(2, 3, 4)
    s = COO.from_numpy(x)

    with pytest.warns(RuntimeWarning):
        getattr(sparse, reduction)(s, axis=axis)


@pytest.mark.parametrize('axis', [
    None,
    (1, 2, 0),
    (2, 1, 0),
    (0, 1, 2),
    (0, 1, -1),
    (0, -2, -1),
    (-3, -2, -1),
])
def test_transpose(axis):
    x = sparse.random((2, 3, 4), density=.25)
    y = x.todense()
    xx = x.transpose(axis)
    yy = y.transpose(axis)
    assert_eq(xx, yy)


@pytest.mark.parametrize('axis', [
    (0, 1),  # too few
    (0, 1, 2, 3),  # too many
    (3, 1, 0),  # axis 3 illegal
    (0, -1, -4),  # axis -4 illegal
    (0, 0, 1),  # duplicate axis 0
    (0, -1, 2),  # duplicate axis -1 == 2
])
def test_transpose_error(axis):
    x = sparse.random((2, 3, 4), density=.25)
    y = x.todense()

    with pytest.raises(ValueError):
        x.transpose(axis)
    with pytest.raises(ValueError):
        y.transpose(axis)


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
    s = sparse.random(a, density=0.5)
    x = s.todense()

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
    s = sparse.random((3, 5), density=0.5)

    assert s.reshape(s.shape) is s


def test_to_scipy_sparse():
    s = sparse.random((3, 5), density=0.5)
    a = s.to_scipy_sparse()
    b = scipy.sparse.coo_matrix(s.todense())

    assert_eq(a, b)


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
    sa = sparse.random(a_shape, density=0.5)
    sb = sparse.random(b_shape, density=0.5)

    a = sa.todense()
    b = sb.todense()

    assert_eq(np.tensordot(a, b, axes),
              sparse.tensordot(sa, sb, axes))

    assert_eq(np.tensordot(a, b, axes),
              sparse.tensordot(sa, b, axes))

    # assert isinstance(sparse.tensordot(sa, b, axes), COO)

    assert_eq(np.tensordot(a, b, axes),
              sparse.tensordot(a, sb, axes))

    # assert isinstance(sparse.tensordot(a, sb, axes), COO)


@pytest.mark.parametrize('a_shape, b_shape', [
    ((3, 4, 5), (5, 6)),
    ((4, 5), (5, 6)),
    ((5,), (5, 6)),
    ((4, 5), (5,)),
    ((5,), (5,)),
])
def test_dot(a_shape, b_shape):
    sa = sparse.random(a_shape, density=0.5)
    sb = sparse.random(b_shape, density=0.5)

    a = sa.todense()
    b = sb.todense()

    assert_eq(a.dot(b), sa.dot(sb))
    assert_eq(np.dot(a, b), sparse.dot(sa, sb))

    if hasattr(operator, 'matmul'):
        # Basic equivalences
        assert_eq(operator.matmul(a, b), operator.matmul(sa, sb))

        # Test that SOO's and np.array's combine correctly
        # Not possible due to https://github.com/numpy/numpy/issues/9028
        # assert_eq(eval("a @ sb"), eval("sa @ b"))


@pytest.mark.xfail
def test_dot_nocoercion():
    sa = sparse.random((3, 4, 5), density=0.5)
    sb = sparse.random((5, 6), density=0.5)

    a = sa.todense()
    b = sb.todense()

    la = a.tolist()
    lb = b.tolist()
    la, lb  # silencing flake8

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
    s = sparse.random((2, 3, 4), density=0.5)
    x = s.todense()

    fs = func(s)
    assert isinstance(fs, COO)
    assert fs.nnz <= s.nnz

    assert_eq(func(x), fs)


@pytest.mark.parametrize('func', [
    operator.mul, operator.add, operator.sub, operator.gt,
    operator.lt, operator.ne
])
@pytest.mark.parametrize('shape', [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_elemwise_binary(func, shape):
    xs = sparse.random(shape, density=0.5)
    ys = sparse.random(shape, density=0.5)

    x = xs.todense()
    y = ys.todense()

    assert_eq(func(xs, ys), func(x, y))


@pytest.mark.parametrize('func', [
    lambda x, y, z: x + y + z,
    lambda x, y, z: x * y * z,
    lambda x, y, z: x + y * z,
    lambda x, y, z: (x + y) * z
])
@pytest.mark.parametrize('shape', [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_elemwise_trinary(func, shape):
    xs = sparse.random(shape, density=0.5)
    ys = sparse.random(shape, density=0.5)
    zs = sparse.random(shape, density=0.5)

    x = xs.todense()
    y = ys.todense()
    z = zs.todense()

    fs = sparse.elemwise(func, xs, ys, zs)
    assert isinstance(fs, COO)

    assert_eq(fs, func(x, y, z))


@pytest.mark.parametrize('func', [operator.add, operator.mul])
@pytest.mark.parametrize('shape1,shape2', [((2, 3, 4), (3, 4)),
                                           ((3, 4), (2, 3, 4)),
                                           ((3, 1, 4), (3, 2, 4)),
                                           ((1, 3, 4), (3, 4)),
                                           ((3, 4, 1), (3, 4, 2)),
                                           ((1, 5), (5, 1)),
                                           ((3, 1), (3, 4)),
                                           ((3, 1), (1, 4)),
                                           ((1, 4), (3, 4))])
def test_binary_broadcasting(func, shape1, shape2):
    xs = sparse.random(shape1, density=0.5)
    x = xs.todense()

    ys = sparse.random(shape2, density=0.5)
    y = ys.todense()

    expected = func(x, y)
    actual = func(xs, ys)

    assert isinstance(actual, COO)
    assert_eq(expected, actual)

    assert np.count_nonzero(expected) == actual.nnz


@pytest.mark.parametrize('shape1,shape2', [((3, 4), (2, 3, 4)),
                                           ((3, 1, 4), (3, 2, 4)),
                                           ((3, 4, 1), (3, 4, 2))])
def test_broadcast_to(shape1, shape2):
    a = sparse.random(shape1, density=0.5)
    x = a.todense()

    assert_eq(np.broadcast_to(x, shape2), a.broadcast_to(shape2))


@pytest.mark.parametrize('shapes', [
    [
        (2,),
        (3, 2),
        (4, 3, 2),
    ],
    [
        (3,),
        (2, 3),
        (2, 2, 3),
    ],
    [
        (2,),
        (2, 2),
        (2, 2, 2),
    ],
    [
        (4,),
        (4, 4),
        (4, 4, 4),
    ],
    [
        (4,),
        (4, 4),
        (4, 4, 4),
    ],
    [
        (4,),
        (4, 4),
        (4, 4, 4),
    ],
    [
        (1, 1, 2),
        (1, 3, 1),
        (4, 1, 1)
    ],
    [
        (2,),
        (2, 1),
        (2, 1, 1)
    ],
])
@pytest.mark.parametrize('func', [
    lambda x, y, z: (x + y) * z,
    lambda x, y, z: x * (y + z),
    lambda x, y, z: x * y * z,
    lambda x, y, z: x + y + z,
    lambda x, y, z: x + y - z,
    lambda x, y, z: x - y + z,
])
def test_trinary_broadcasting(shapes, func):
    args = [sparse.random(s, density=0.5) for s in shapes]
    dense_args = [arg.todense() for arg in args]

    fs = sparse.elemwise(func, *args)
    assert isinstance(fs, COO)

    assert_eq(fs, func(*dense_args))


@pytest.mark.parametrize('shapes, func', [
    ([
         (2,),
         (3, 2),
         (4, 3, 2),
     ], lambda x, y, z: (x + y) * z),
    ([
         (3,),
         (2, 3),
         (2, 2, 3),
     ], lambda x, y, z: x * (y + z)),
    ([
         (2,),
         (2, 2),
         (2, 2, 2),
     ], lambda x, y, z: x * y * z),
    ([
         (4,),
         (4, 4),
         (4, 4, 4),
     ], lambda x, y, z: x + y + z),
])
@pytest.mark.parametrize('value', [
    np.nan,
    np.inf,
    -np.inf
])
@pytest.mark.parametrize('fraction', [0.25, 0.5, 0.75, 1.0])
@pytest.mark.filterwarnings('ignore:invalid value')
def test_trinary_broadcasting_pathological(shapes, func, value, fraction):
    args = [sparse.random(s, density=0.5, data_rvs=random_value_array(value, fraction))
            for s in shapes]
    dense_args = [arg.todense() for arg in args]

    fs = sparse.elemwise(func, *args)
    assert isinstance(fs, COO)

    assert_eq(fs, func(*dense_args), equal_nan=True)


def test_sparse_broadcasting(monkeypatch):
    orig_unmatch_coo = sparse.coo._unmatch_coo

    state = {'num_matches': 0}

    xs = sparse.random((3, 4), density=0.5)
    ys = sparse.random((3, 4), density=0.5)

    def mock_unmatch_coo(*args, **kwargs):
        result = orig_unmatch_coo(*args, **kwargs)
        state['num_matches'] += len(result[0])
        return result

    monkeypatch.setattr(sparse.coo, '_unmatch_coo', mock_unmatch_coo)

    xs * ys

    # Less than in case there's absolutely no overlap in some cases.
    assert state['num_matches'] <= 1


def test_dense_broadcasting(monkeypatch):
    orig_unmatch_coo = sparse.coo._unmatch_coo

    state = {'num_matches': 0}

    xs = sparse.random((3, 4), density=0.5)
    ys = sparse.random((3, 4), density=0.5)

    def mock_unmatch_coo(*args, **kwargs):
        result = orig_unmatch_coo(*args, **kwargs)
        state['num_matches'] += len(result[0])
        return result

    monkeypatch.setattr(sparse.coo, '_unmatch_coo', mock_unmatch_coo)

    xs + ys

    # Less than in case there's absolutely no overlap in some cases.
    assert state['num_matches'] <= 3


@pytest.mark.parametrize('format', ['coo', 'dok'])
def test_sparsearray_elemwise(format):
    xs = sparse.random((3, 4), density=0.5, format=format)
    ys = sparse.random((3, 4), density=0.5, format=format)

    x = xs.todense()
    y = ys.todense()

    fs = sparse.elemwise(operator.add, xs, ys)
    assert isinstance(fs, COO)

    assert_eq(fs, x + y)


def test_ndarray_densification_fails():
    xs = sparse.random((3, 4), density=0.5)
    y = np.random.rand(3, 4)

    with pytest.raises(ValueError):
        xs + y


def test_elemwise_noargs():
    def func():
        return np.float_(5.0)

    assert sparse.elemwise(func) == func()


@pytest.mark.parametrize('func', [
    operator.pow, operator.truediv, operator.floordiv,
    operator.ge, operator.le, operator.eq, operator.mod
])
@pytest.mark.filterwarnings('ignore:divide by zero')
@pytest.mark.filterwarnings('ignore:invalid value')
def test_auto_densification_fails(func):
    xs = sparse.random((2, 3, 4), density=0.5)
    ys = sparse.random((2, 3, 4), density=0.5)

    with pytest.raises(ValueError):
        func(xs, ys)


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
    (operator.eq, 1),
    (operator.mod, 5)
])
@pytest.mark.parametrize('convert_to_np_number', [True, False])
def test_elemwise_scalar(func, scalar, convert_to_np_number):
    xs = sparse.random((2, 3, 4), density=0.5)
    if convert_to_np_number:
        scalar = np.float32(scalar)
    y = scalar

    x = xs.todense()
    fs = func(xs, y)

    assert isinstance(fs, COO)
    assert xs.nnz >= fs.nnz

    assert_eq(fs, func(x, y))


@pytest.mark.parametrize('func, scalar', [
    (operator.mul, 5),
    (operator.add, 0),
    (operator.sub, 0),
    (operator.gt, -5),
    (operator.lt, 5),
    (operator.ne, 0),
    (operator.ge, -5),
    (operator.le, 3),
    (operator.eq, 1),
])
@pytest.mark.parametrize('convert_to_np_number', [True, False])
def test_leftside_elemwise_scalar(func, scalar, convert_to_np_number):
    xs = sparse.random((2, 3, 4), density=0.5)
    if convert_to_np_number:
        scalar = np.float32(scalar)
    y = scalar

    x = xs.todense()
    fs = func(y, xs)

    assert isinstance(fs, COO)
    assert xs.nnz >= fs.nnz

    assert_eq(fs, func(y, x))


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
    xs = sparse.random((2, 3, 4), density=0.5)
    y = scalar

    with pytest.raises(ValueError):
        func(xs, y)


@pytest.mark.parametrize('func', [
    operator.and_, operator.or_, operator.xor
])
@pytest.mark.parametrize('shape', [
    (2,),
    (2, 3),
    (2, 3, 4),
    (2, 3, 4, 5)
])
def test_bitwise_binary(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int_)
    ys = (sparse.random(shape, density=0.5) * 100).astype(np.int_)

    x = xs.todense()
    y = ys.todense()

    assert_eq(func(xs, ys), func(x, y))


@pytest.mark.parametrize('func', [
    operator.lshift, operator.rshift
])
@pytest.mark.parametrize('shape', [
    (2,),
    (2, 3),
    (2, 3, 4),
    (2, 3, 4, 5)
])
def test_bitshift_binary(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int_)

    # Can't merge into test_bitwise_binary because left/right shifting
    # with something >= 64 isn't defined.
    ys = (sparse.random(shape, density=0.5) * 64).astype(np.int_)

    x = xs.todense()
    y = ys.todense()

    assert_eq(func(xs, ys), func(x, y))


@pytest.mark.parametrize('func', [
    operator.and_
])
@pytest.mark.parametrize('shape', [
    (2,),
    (2, 3),
    (2, 3, 4),
    (2, 3, 4, 5)
])
def test_bitwise_scalar(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int_)

    # Can't merge into test_bitwise_binary because left/right shifting
    # with something >= 64 isn't defined.
    y = np.random.randint(100)

    x = xs.todense()

    assert_eq(func(xs, y), func(x, y))
    assert_eq(func(y, xs), func(y, x))


@pytest.mark.parametrize('func', [
    operator.lshift, operator.rshift
])
@pytest.mark.parametrize('shape', [
    (2,),
    (2, 3),
    (2, 3, 4),
    (2, 3, 4, 5)
])
def test_bitshift_scalar(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int_)

    # Can't merge into test_bitwise_binary because left/right shifting
    # with something >= 64 isn't defined.
    y = np.random.randint(64)

    x = xs.todense()

    assert_eq(func(xs, y), func(x, y))


@pytest.mark.parametrize('func', [operator.invert])
@pytest.mark.parametrize('shape', [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_unary_bitwise_densification_fails(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int_)

    with pytest.raises(ValueError):
        func(xs)


@pytest.mark.parametrize('func', [operator.or_, operator.xor])
@pytest.mark.parametrize('shape', [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_binary_bitwise_densification_fails(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int_)
    y = np.random.randint(1, 100)

    with pytest.raises(ValueError):
        func(xs, y)

    with pytest.raises(ValueError):
        func(y, xs)


@pytest.mark.parametrize('func', [operator.lshift, operator.rshift])
@pytest.mark.parametrize('shape', [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_binary_bitshift_densification_fails(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    x = np.random.randint(1, 100)
    ys = (sparse.random(shape, density=0.5) * 64).astype(np.int_)

    with pytest.raises(ValueError):
        func(x, ys)


@pytest.mark.parametrize('func', [operator.and_, operator.or_, operator.xor])
@pytest.mark.parametrize('shape', [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_bitwise_binary_bool(func, shape):
    # Small arrays need high density to have nnz entries
    xs = sparse.random(shape, density=0.5).astype(bool)
    ys = sparse.random(shape, density=0.5).astype(bool)

    x = xs.todense()
    y = ys.todense()

    assert_eq(func(xs, ys), func(x, y))


def test_elemwise_binary_empty():
    x = COO({}, shape=(10, 10))
    y = sparse.random((10, 10), density=0.5)

    for z in [x * y, y * x]:
        assert z.nnz == 0
        assert z.coords.shape == (2, 0)
        assert z.data.shape == (0,)


def test_gt():
    s = sparse.random((2, 3, 4), density=0.5)
    x = s.todense()

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
    s = sparse.random((2, 3, 4), density=0.5)
    x = s.todense()

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
    pytest.param(
        [True, False, False],
        marks=pytest.mark.skipif(
            version.parse(np.version.version) < version.parse("1.13.0"),
            reason="NumPy < 1.13.0 does not raise these Exceptions"
        )
    ),
])
def test_slicing_errors(index):
    s = sparse.random((2, 3, 4), density=0.5)
    x = s.todense()

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
    data = np.arange(5) + 1

    old = COO(coords, data, shape=(2, 2, 5))
    x = COO(coords, data, shape=(2, 2, 5))
    x.sum_duplicates()

    assert_eq(old, x)
    # assert x.nnz == 5
    # assert x.has_duplicates
    assert x.nnz == 3
    assert not x.has_duplicates


def test_concatenate():
    xx = sparse.random((2, 3, 4), density=0.5)
    x = xx.todense()
    yy = sparse.random((5, 3, 4), density=0.5)
    y = yy.todense()
    zz = sparse.random((4, 3, 4), density=0.5)
    z = zz.todense()

    assert_eq(np.concatenate([x, y, z], axis=0),
              sparse.concatenate([xx, yy, zz], axis=0))

    xx = sparse.random((5, 3, 1), density=0.5)
    x = xx.todense()
    yy = sparse.random((5, 3, 3), density=0.5)
    y = yy.todense()
    zz = sparse.random((5, 3, 2), density=0.5)
    z = zz.todense()

    assert_eq(np.concatenate([x, y, z], axis=2),
              sparse.concatenate([xx, yy, zz], axis=2))

    assert_eq(np.concatenate([x, y, z], axis=-1),
              sparse.concatenate([xx, yy, zz], axis=-1))


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('func', ['stack', 'concatenate'])
def test_concatenate_mixed(func, axis):
    s = sparse.random((10, 10), density=0.5)
    d = s.todense()

    result = getattr(sparse, func)([d, s, s], axis=axis)
    expected = getattr(np, func)([d, d, d], axis=axis)

    assert isinstance(result, COO)

    assert_eq(result, expected)


@pytest.mark.parametrize('shape', [(5,), (2, 3, 4), (5, 2)])
@pytest.mark.parametrize('axis', [0, 1, -1])
def test_stack(shape, axis):
    xx = sparse.random(shape, density=0.5)
    x = xx.todense()
    yy = sparse.random(shape, density=0.5)
    y = yy.todense()
    zz = sparse.random(shape, density=0.5)
    z = zz.todense()

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
    s = sparse.random((2, 3, 4), density=0.5)
    assert s.coords.dtype == np.uint8

    s = COO.from_numpy(np.zeros(1000))
    assert s.coords.dtype == np.uint16


def test_addition():
    a = sparse.random((2, 3, 4), density=0.5)
    x = a.todense()

    b = sparse.random((2, 3, 4), density=0.5)
    y = b.todense()

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


@pytest.mark.parametrize('scalar', [2, 2.5, np.float32(2.0), np.int8(3)])
def test_scalar_multiplication(scalar):
    a = sparse.random((2, 3, 4), density=0.5)
    x = a.todense()

    assert_eq(x * scalar, a * scalar)
    assert (a * scalar).nnz == a.nnz
    assert_eq(scalar * x, scalar * a)
    assert (scalar * a).nnz == a.nnz
    assert_eq(x / scalar, a / scalar)
    assert (a / scalar).nnz == a.nnz
    assert_eq(x // scalar, a // scalar)
    # division may reduce nnz.


@pytest.mark.filterwarnings('ignore:divide by zero')
def test_scalar_exponentiation():
    a = sparse.random((2, 3, 4), density=0.5)
    x = a.todense()

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

    assert_eq(x, xx, check_nnz=False)
    assert_eq(x.T, xx.T, check_nnz=False)
    assert_eq(xx.to_scipy_sparse(), x, check_nnz=False)
    assert_eq(COO.from_scipy_sparse(xx.to_scipy_sparse()), xx, check_nnz=False)

    assert_eq(x, xx, check_nnz=False)
    assert_eq(x.T.dot(x), xx.T.dot(xx), check_nnz=False)
    assert isinstance(x + xx, COO)
    assert isinstance(xx + x, COO)


@pytest.mark.parametrize('scipy_format', ['coo', 'csr', 'dok', 'csc'])
def test_scipy_sparse_interaction(scipy_format):
    x = sparse.random((10, 20), density=0.2).todense()
    sp = getattr(scipy.sparse, scipy_format + '_matrix')(x)
    coo = COO(x)
    assert isinstance(sp + coo, COO)
    assert isinstance(coo + sp, COO)
    assert_eq(sp, coo)


@pytest.mark.parametrize('func', [
    operator.mul, operator.add, operator.sub, operator.gt,
    operator.lt, operator.ne
])
def test_op_scipy_sparse(func):
    xs = sparse.random((3, 4), density=0.5)
    y = sparse.random((3, 4), density=0.5).todense()

    ys = scipy.sparse.csr_matrix(y)
    x = xs.todense()

    assert_eq(func(x, y), func(xs, ys))


@pytest.mark.parametrize('func', [
    operator.add,
    operator.sub,
    pytest.mark.xfail(operator.mul, reason='Scipy sparse auto-densifies in this case.'),
    pytest.mark.xfail(operator.gt, reason='Scipy sparse doesn\'t support this yet.'),
    pytest.mark.xfail(operator.lt, reason='Scipy sparse doesn\'t support this yet.'),
    pytest.mark.xfail(operator.ne, reason='Scipy sparse doesn\'t support this yet.'),
])
def test_op_scipy_sparse_left(func):
    ys = sparse.random((3, 4), density=0.5)
    x = sparse.random((3, 4), density=0.5).todense()

    xs = scipy.sparse.csr_matrix(x)
    y = ys.todense()

    assert_eq(func(x, y), func(xs, ys))


def test_cache_csr():
    x = sparse.random((10, 5), density=0.5).todense()
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
    s = sparse.random(shape, density=0.5)
    x = s.todense()

    assert_eq(np.triu(x, k), sparse.triu(s, k))
    assert_eq(np.tril(x, k), sparse.tril(s, k))


def test_empty_reduction():
    x = np.zeros((2, 3, 4), dtype=np.float_)
    xs = COO.from_numpy(x)

    assert_eq(x.sum(axis=(0, 2)),
              xs.sum(axis=(0, 2)))


@pytest.mark.parametrize('shape', [
    (2,),
    (2, 3),
    (2, 3, 4),
])
@pytest.mark.parametrize('density', [
    0.1, 0.3, 0.5, 0.7
])
def test_random_shape(shape, density):
    s = sparse.random(shape, density)

    assert isinstance(s, COO)

    assert s.shape == shape
    expected_nnz = density * np.prod(shape)
    assert np.floor(expected_nnz) <= s.nnz <= np.ceil(expected_nnz)


def test_two_random_unequal():
    s1 = sparse.random((2, 3, 4), 0.3)
    s2 = sparse.random((2, 3, 4), 0.3)

    assert not np.allclose(s1.todense(), s2.todense())


def test_two_random_same_seed():
    state = np.random.randint(100)
    s1 = sparse.random((2, 3, 4), 0.3, random_state=state)
    s2 = sparse.random((2, 3, 4), 0.3, random_state=state)

    assert_eq(s1, s2)


def test_random_sorted():
    s = sparse.random((2, 3, 4), canonical_order=True)

    assert is_lexsorted(s)


@pytest.mark.parametrize('rvs, dtype', [
    (None, np.float64),
    (scipy.stats.poisson(25, loc=10).rvs, np.int),
    (lambda x: np.random.choice([True, False], size=x), np.bool),
])
@pytest.mark.parametrize('shape', [
    (2, 4, 5),
    (20, 40, 50),
])
@pytest.mark.parametrize('density', [
    0.0, 0.01, 0.1, 0.2,
])
def test_random_rvs(rvs, dtype, shape, density):
    x = sparse.random(shape, density, data_rvs=rvs)
    assert x.shape == shape
    assert x.dtype == dtype


def test_scalar_shape_construction():
    x = np.random.rand(5)
    coords = np.arange(5)[None]

    s = COO(coords, x, shape=5)

    assert_eq(x, s)


def test_len():
    s = sparse.random((20, 30, 40))
    assert len(s) == 20


def test_density():
    s = sparse.random((20, 30, 40), density=0.1)
    assert np.isclose(s.density, 0.1)


def test_size():
    s = sparse.random((20, 30, 40))
    assert s.size == 20 * 30 * 40


def test_np_array():
    s = sparse.random((20, 30, 40))
    x = np.array(s)
    assert isinstance(x, np.ndarray)
    assert_eq(x, s)


@pytest.mark.parametrize('shapes', [
    [
        (2,),
        (3, 2),
        (4, 3, 2),
    ],
    [
        (3,),
        (2, 3),
        (2, 2, 3),
    ],
    [
        (2,),
        (2, 2),
        (2, 2, 2),
    ],
    [
        (4,),
        (4, 4),
        (4, 4, 4),
    ],
    [
        (4,),
        (4, 4),
        (4, 4, 4),
    ],
    [
        (4,),
        (4, 4),
        (4, 4, 4),
    ],
    [
        (1, 1, 2),
        (1, 3, 1),
        (4, 1, 1)
    ],
    [
        (2,),
        (2, 1),
        (2, 1, 1)
    ],
])
def test_three_arg_where(shapes):
    cs = sparse.random(shapes[0], density=0.5).astype(np.bool)
    xs = sparse.random(shapes[1], density=0.5)
    ys = sparse.random(shapes[2], density=0.5)

    c = cs.todense()
    x = xs.todense()
    y = ys.todense()

    expected = np.where(c, x, y)
    actual = sparse.where(cs, xs, ys)

    assert isinstance(actual, COO)
    assert_eq(expected, actual)


def test_one_arg_where():
    s = sparse.random((2, 3, 4), density=0.5)
    x = s.todense()

    expected = np.where(x)
    actual = sparse.where(s)

    assert len(expected) == len(actual)

    for e, a in zip(expected, actual):
        assert_eq(e, a, compare_dtype=False)


def test_one_arg_where_dense():
    x = np.random.rand(2, 3, 4)

    with pytest.raises(ValueError):
        sparse.where(x)


def test_two_arg_where():
    cs = sparse.random((2, 3, 4), density=0.5).astype(np.bool)
    xs = sparse.random((2, 3, 4), density=0.5)

    with pytest.raises(ValueError):
        sparse.where(cs, xs)
