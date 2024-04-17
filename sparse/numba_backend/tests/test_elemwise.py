import operator

import sparse
from sparse import COO, DOK
from sparse.numba_backend._compressed import GCXS
from sparse.numba_backend._utils import assert_eq, random_value_array

import pytest

import numpy as np


@pytest.mark.parametrize(
    "func",
    [
        np.expm1,
        np.log1p,
        np.sin,
        np.tan,
        np.sinh,
        np.tanh,
        np.floor,
        np.ceil,
        np.sqrt,
        np.conj,
        np.round,
        np.rint,
        lambda x: x.astype("int32"),
        np.conjugate,
        np.conj,
        lambda x: x.round(decimals=2),
        abs,
    ],
)
@pytest.mark.parametrize("format", [COO, GCXS, DOK])
def test_elemwise(func, format):
    s = sparse.random((2, 3, 4), density=0.5, format=format)
    x = s.todense()

    fs = func(s)
    assert isinstance(fs, format)
    assert fs.nnz <= s.nnz

    assert_eq(func(x), fs)


@pytest.mark.parametrize(
    "func",
    [
        np.expm1,
        np.log1p,
        np.sin,
        np.tan,
        np.sinh,
        np.tanh,
        np.floor,
        np.ceil,
        np.sqrt,
        np.conj,
        np.round,
        np.rint,
        np.conjugate,
        np.conj,
        lambda x, out: x.round(decimals=2, out=out),
    ],
)
@pytest.mark.parametrize("format", [COO, GCXS, DOK])
def test_elemwise_inplace(func, format):
    s = sparse.random((2, 3, 4), density=0.5, format=format)
    x = s.todense()

    func(s, out=s)
    func(x, out=x)
    assert isinstance(s, format)

    assert_eq(x, s)


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((2, 3, 4), (3, 4)),
        ((3, 4), (2, 3, 4)),
        ((3, 1, 4), (3, 2, 4)),
        ((1, 3, 4), (3, 4)),
        ((3, 4, 1), (3, 4, 2)),
        ((1, 5), (5, 1)),
        ((3, 1), (3, 4)),
        ((3, 1), (1, 4)),
        ((1, 4), (3, 4)),
        ((2, 2, 2), (1, 1, 1)),
    ],
)
@pytest.mark.parametrize("format", [COO, GCXS, DOK])
def test_elemwise_mixed(shape1, shape2, format, rng):
    s1 = sparse.random(shape1, density=0.5, format=format)
    x2 = rng.random(shape2)

    x1 = s1.todense()

    assert_eq(s1 * x2, x1 * x2)


@pytest.mark.parametrize("format", [COO, GCXS, DOK])
def test_elemwise_mixed_empty(format, rng):
    s1 = sparse.random((2, 0, 4), density=0.5, format=format)
    x2 = rng.random((2, 0, 4))

    x1 = s1.todense()

    assert_eq(s1 * x2, x1 * x2)


@pytest.mark.parametrize("format", [COO, GCXS, DOK])
def test_elemwise_unsupported(format):
    class A:
        pass

    s1 = sparse.random((2, 3, 4), density=0.5, format=format)
    x2 = A()

    with pytest.raises(TypeError):
        s1 + x2

    assert sparse.elemwise(operator.add, s1, x2) is NotImplemented


@pytest.mark.parametrize("format", [COO, GCXS, DOK])
def test_elemwise_mixed_broadcast(format, rng):
    s1 = sparse.random((2, 3, 4), density=0.5, format=format)
    s2 = sparse.random(4, density=0.5)
    x3 = rng.random((3, 4))

    x1 = s1.todense()
    x2 = s2.todense()

    def func(x1, x2, x3):
        return x1 * x2 * x3

    assert_eq(sparse.elemwise(func, s1, s2, x3), func(x1, x2, x3))


@pytest.mark.parametrize(
    "func",
    [operator.mul, operator.add, operator.sub, operator.gt, operator.lt, operator.ne],
)
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
@pytest.mark.parametrize("format", [COO, GCXS, DOK])
def test_elemwise_binary(func, shape, format):
    xs = sparse.random(shape, density=0.5, format=format)
    ys = sparse.random(shape, density=0.5, format=format)

    x = xs.todense()
    y = ys.todense()

    assert_eq(func(xs, ys), func(x, y))


@pytest.mark.parametrize("func", [operator.imul, operator.iadd, operator.isub])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
@pytest.mark.parametrize("format", [COO, GCXS, DOK])
def test_elemwise_binary_inplace(func, shape, format):
    xs = sparse.random(shape, density=0.5, format=format)
    ys = sparse.random(shape, density=0.5, format=format)

    x = xs.todense()
    y = ys.todense()

    xs = func(xs, ys)
    x = func(x, y)

    assert_eq(xs, x)


@pytest.mark.parametrize(
    "func",
    [
        lambda x, y, z: x + y + z,
        lambda x, y, z: x * y * z,
        lambda x, y, z: x + y * z,
        lambda x, y, z: (x + y) * z,
    ],
)
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
@pytest.mark.parametrize(
    "formats",
    [
        [COO, COO, COO],
        [GCXS, GCXS, GCXS],
        [COO, GCXS, GCXS],
    ],
)
def test_elemwise_trinary(func, shape, formats):
    xs = sparse.random(shape, density=0.5, format=formats[0])
    ys = sparse.random(shape, density=0.5, format=formats[1])
    zs = sparse.random(shape, density=0.5, format=formats[2])

    x = xs.todense()
    y = ys.todense()
    z = zs.todense()

    fs = sparse.elemwise(func, xs, ys, zs)
    assert_eq(fs, func(x, y, z))


@pytest.mark.parametrize("func", [operator.add, operator.mul])
@pytest.mark.parametrize(
    "shape1,shape2",
    [
        ((2, 3, 4), (3, 4)),
        ((3, 4), (2, 3, 4)),
        ((3, 1, 4), (3, 2, 4)),
        ((1, 3, 4), (3, 4)),
        ((3, 4, 1), (3, 4, 2)),
        ((1, 5), (5, 1)),
        ((3, 1), (3, 4)),
        ((3, 1), (1, 4)),
        ((1, 4), (3, 4)),
        ((2, 2, 2), (1, 1, 1)),
    ],
)
def test_binary_broadcasting(func, shape1, shape2):
    density1 = 1 if np.prod(shape1) == 1 else 0.5
    density2 = 1 if np.prod(shape2) == 1 else 0.5

    xs = sparse.random(shape1, density=density1)
    x = xs.todense()

    ys = sparse.random(shape2, density=density2)
    y = ys.todense()

    expected = func(x, y)
    actual = func(xs, ys)

    assert isinstance(actual, COO)
    assert_eq(expected, actual)

    assert np.count_nonzero(expected) == actual.nnz


@pytest.mark.parametrize(
    "shape1,shape2",
    [((3, 4), (2, 3, 4)), ((3, 1, 4), (3, 2, 4)), ((3, 4, 1), (3, 4, 2))],
)
def test_broadcast_to(shape1, shape2):
    a = sparse.random(shape1, density=0.5)
    x = a.todense()

    assert_eq(np.broadcast_to(x, shape2), a.broadcast_to(shape2))


@pytest.mark.parametrize(
    "shapes",
    [
        [(2,), (3, 2), (4, 3, 2)],
        [(3,), (2, 3), (2, 2, 3)],
        [(2,), (2, 2), (2, 2, 2)],
        [(4,), (4, 4), (4, 4, 4)],
        [(4,), (4, 4), (4, 4, 4)],
        [(4,), (4, 4), (4, 4, 4)],
        [(1, 1, 2), (1, 3, 1), (4, 1, 1)],
        [(2,), (2, 1), (2, 1, 1)],
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        lambda x, y, z: (x + y) * z,
        lambda x, y, z: x * (y + z),
        lambda x, y, z: x * y * z,
        lambda x, y, z: x + y + z,
        lambda x, y, z: x + y - z,
        lambda x, y, z: x - y + z,
    ],
)
def test_trinary_broadcasting(shapes, func):
    args = [sparse.random(s, density=0.5) for s in shapes]
    dense_args = [arg.todense() for arg in args]

    fs = sparse.elemwise(func, *args)
    assert isinstance(fs, COO)

    assert_eq(fs, func(*dense_args))


@pytest.mark.parametrize(
    "shapes, func",
    [
        ([(2,), (3, 2), (4, 3, 2)], lambda x, y, z: (x + y) * z),
        ([(3,), (2, 3), (2, 2, 3)], lambda x, y, z: x * (y + z)),
        ([(2,), (2, 2), (2, 2, 2)], lambda x, y, z: x * y * z),
        ([(4,), (4, 4), (4, 4, 4)], lambda x, y, z: x + y + z),
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.filterwarnings("ignore:invalid value")
def test_trinary_broadcasting_pathological(shapes, func, value, fraction):
    args = [sparse.random(s, density=0.5, data_rvs=random_value_array(value, fraction)) for s in shapes]
    dense_args = [arg.todense() for arg in args]

    fs = sparse.elemwise(func, *args)
    assert isinstance(fs, COO)

    assert_eq(fs, func(*dense_args))


def test_sparse_broadcasting(monkeypatch):
    orig_unmatch_coo = sparse.numba_backend._umath._Elemwise._get_func_coords_data

    state = {"num_matches": 0}

    xs = sparse.random((3, 4), density=0.5)
    ys = sparse.random((3, 4), density=0.5)

    def mock_unmatch_coo(*args, **kwargs):
        result = orig_unmatch_coo(*args, **kwargs)
        if result is not None:
            state["num_matches"] += 1
        return result

    monkeypatch.setattr(sparse.numba_backend._umath._Elemwise, "_get_func_coords_data", mock_unmatch_coo)

    xs * ys

    # Less than in case there's absolutely no overlap in some cases.
    assert state["num_matches"] <= 1


def test_dense_broadcasting(monkeypatch):
    orig_unmatch_coo = sparse.numba_backend._umath._Elemwise._get_func_coords_data

    state = {"num_matches": 0}

    xs = sparse.random((3, 4), density=0.5)
    ys = sparse.random((3, 4), density=0.5)

    def mock_unmatch_coo(*args, **kwargs):
        result = orig_unmatch_coo(*args, **kwargs)
        if result is not None:
            state["num_matches"] += 1
        return result

    monkeypatch.setattr(sparse.numba_backend._umath._Elemwise, "_get_func_coords_data", mock_unmatch_coo)

    xs + ys

    # Less than in case there's absolutely no overlap in some cases.
    assert state["num_matches"] <= 3


@pytest.mark.parametrize("format", ["coo", "dok", "gcxs"])
def test_sparsearray_elemwise(format):
    xs = sparse.random((3, 4), density=0.5, format=format)
    ys = sparse.random((3, 4), density=0.5, format=format)

    x = xs.todense()
    y = ys.todense()

    fs = sparse.elemwise(operator.add, xs, ys)
    if format == "gcxs":
        assert isinstance(fs, GCXS)
    elif format == "dok":
        assert isinstance(fs, DOK)
    else:
        assert isinstance(fs, COO)

    assert_eq(fs, x + y)


def test_ndarray_densification_fails(rng):
    xs = sparse.random((2, 3, 4), density=0.5)
    y = rng.random((3, 4))

    with pytest.raises(ValueError):
        xs + y


def test_elemwise_noargs():
    def func():
        return np.float64(5.0)

    with pytest.raises(ValueError, match=r"None of the args is sparse:"):
        sparse.elemwise(func)


@pytest.mark.parametrize(
    "func",
    [
        operator.pow,
        operator.truediv,
        operator.floordiv,
        operator.ge,
        operator.le,
        operator.eq,
        operator.mod,
    ],
)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value")
@pytest.mark.parametrize("format", [COO, GCXS, DOK])
def test_nonzero_outout_fv_ufunc(func, format):
    xs = sparse.random((2, 3, 4), density=0.5, format=format)
    ys = sparse.random((2, 3, 4), density=0.5, format=format)

    x = xs.todense()
    y = ys.todense()

    f = func(x, y)
    fs = func(xs, ys)
    assert isinstance(fs, format)

    assert_eq(f, fs)


@pytest.mark.parametrize(
    "func, scalar",
    [
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
        (operator.mod, 5),
    ],
)
@pytest.mark.parametrize("convert_to_np_number", [True, False])
@pytest.mark.parametrize("format", [COO, GCXS, DOK])
def test_elemwise_scalar(func, scalar, convert_to_np_number, format):
    xs = sparse.random((2, 3, 4), density=0.5, format=format)
    if convert_to_np_number:
        scalar = np.float32(scalar)
    y = scalar

    x = xs.todense()
    fs = func(xs, y)

    assert isinstance(fs, format)
    assert xs.nnz >= fs.nnz

    assert_eq(fs, func(x, y))


@pytest.mark.parametrize(
    "func, scalar",
    [
        (operator.mul, 5),
        (operator.add, 0),
        (operator.sub, 0),
        (operator.gt, -5),
        (operator.lt, 5),
        (operator.ne, 0),
        (operator.ge, -5),
        (operator.le, 3),
        (operator.eq, 1),
    ],
)
@pytest.mark.parametrize("convert_to_np_number", [True, False])
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


@pytest.mark.parametrize(
    "func, scalar",
    [
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
        (operator.eq, 0),
    ],
)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value")
def test_scalar_output_nonzero_fv(func, scalar):
    xs = sparse.random((2, 3, 4), density=0.5)
    y = scalar

    x = xs.todense()

    f = func(x, y)
    fs = func(xs, y)

    assert isinstance(fs, COO)
    assert fs.nnz <= xs.nnz

    assert_eq(f, fs)


@pytest.mark.parametrize("func", [operator.and_, operator.or_, operator.xor])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
@pytest.mark.parametrize("format", [COO, GCXS, DOK])
def test_bitwise_binary(func, shape, format):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5, format=format) * 100).astype(np.int64)
    ys = (sparse.random(shape, density=0.5, format=format) * 100).astype(np.int64)

    x = xs.todense()
    y = ys.todense()

    assert_eq(func(xs, ys), func(x, y))


@pytest.mark.parametrize("func", [operator.iand, operator.ior, operator.ixor])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
@pytest.mark.parametrize("format", [COO, GCXS, DOK])
def test_bitwise_binary_inplace(func, shape, format):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5, format=format) * 100).astype(np.int64)
    ys = (sparse.random(shape, density=0.5, format=format) * 100).astype(np.int64)

    x = xs.todense()
    y = ys.todense()

    xs = func(xs, ys)
    x = func(x, y)

    assert_eq(xs, x)


@pytest.mark.parametrize("func", [operator.lshift, operator.rshift])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_bitshift_binary(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int64)

    # Can't merge into test_bitwise_binary because left/right shifting
    # with something >= 64 isn't defined.
    ys = (sparse.random(shape, density=0.5) * 64).astype(np.int64)

    x = xs.todense()
    y = ys.todense()

    assert_eq(func(xs, ys), func(x, y))


@pytest.mark.parametrize("func", [operator.ilshift, operator.irshift])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_bitshift_binary_inplace(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int64)

    # Can't merge into test_bitwise_binary because left/right shifting
    # with something >= 64 isn't defined.
    ys = (sparse.random(shape, density=0.5) * 64).astype(np.int64)

    x = xs.todense()
    y = ys.todense()

    xs = func(xs, ys)
    x = func(x, y)

    assert_eq(xs, x)


@pytest.mark.parametrize("func", [operator.and_])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_bitwise_scalar(func, shape, rng):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int64)
    y = rng.integers(100)

    x = xs.todense()

    assert_eq(func(xs, y), func(x, y))
    assert_eq(func(y, xs), func(y, x))


@pytest.mark.parametrize("func", [operator.lshift, operator.rshift])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_bitshift_scalar(func, shape, rng):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int64)

    # Can't merge into test_bitwise_binary because left/right shifting
    # with something >= 64 isn't defined.
    y = rng.integers(64)

    x = xs.todense()

    assert_eq(func(xs, y), func(x, y))


@pytest.mark.parametrize("func", [operator.invert])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_unary_bitwise_nonzero_output_fv(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int64)
    x = xs.todense()

    f = func(x)
    fs = func(xs)

    assert isinstance(fs, COO)
    assert fs.nnz <= xs.nnz

    assert_eq(f, fs)


@pytest.mark.parametrize("func", [operator.or_, operator.xor])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_binary_bitwise_nonzero_output_fv(func, shape, rng):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int64)
    y = rng.integers(1, 100)

    x = xs.todense()

    f = func(x, y)
    fs = func(xs, y)

    assert isinstance(fs, COO)
    assert fs.nnz <= xs.nnz

    assert_eq(f, fs)


@pytest.mark.parametrize(
    "func",
    [operator.mul, operator.add, operator.sub, operator.gt, operator.lt, operator.ne],
)
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_elemwise_nonzero_input_fv(func, shape, rng):
    xs = sparse.random(shape, density=0.5, fill_value=rng.random())
    ys = sparse.random(shape, density=0.5, fill_value=rng.random())

    x = xs.todense()
    y = ys.todense()

    assert_eq(func(xs, ys), func(x, y))


@pytest.mark.parametrize("func", [operator.lshift, operator.rshift])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_binary_bitshift_densification_fails(func, shape, rng):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    x = rng.integers(1, 100)
    ys = (sparse.random(shape, density=0.5) * 64).astype(np.int64)

    y = ys.todense()

    f = func(x, y)
    fs = func(x, ys)

    assert isinstance(fs, COO)
    assert fs.nnz <= ys.nnz

    assert_eq(f, fs)


@pytest.mark.parametrize("func", [operator.and_, operator.or_, operator.xor])
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
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


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_nanmean_regression(dtype):
    array = np.array([0.0 + 0.0j, 0.0 + np.nan * 1j], dtype=dtype)
    sparray = sparse.COO.from_numpy(array)
    assert_eq(array, sparray)


# Regression test for gh-580
@pytest.mark.filterwarnings("error")
def test_no_deprecation_warning():
    a = np.array([1, 2])
    s = sparse.COO(a, a, shape=(3,))
    assert_eq(s == s, np.broadcast_to(True, s.shape))


# Regression test for gh-587
def test_no_out_upcast():
    a = sparse.COO([[0, 1], [0, 1]], [1, 1], shape=(2, 2))
    with pytest.raises(TypeError):
        a *= 0.5
