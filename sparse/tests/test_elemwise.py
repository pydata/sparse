import numpy as np
import sparse
import pytest
from hypothesis import settings, given, strategies as st
from hypothesis.strategies import composite
from _utils import gen_bin_brdcst
import operator
import random
from sparse import COO, DOK
from sparse._compressed import GCXS
from sparse._utils import assert_eq, random_value_array


@settings(deadline=None)
@given(
    format=st.sampled_from([COO, GCXS, DOK]),
    func=st.sampled_from(
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
        ]
    ),
)
def test_elemwise(func, format):
    s = sparse.random((2, 3, 4), density=0.5, format=format)
    x = s.todense()

    fs = func(s)
    assert isinstance(fs, format)
    assert fs.nnz <= s.nnz

    assert_eq(func(x), fs)


@given(
    func=st.sampled_from(
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
        ]
    ),
    format=st.sampled_from([COO, GCXS, DOK]),
)
def test_elemwise_inplace(func, format):
    s = sparse.random((2, 3, 4), density=0.5, format=format)
    x = s.todense()

    func(s, out=s)
    func(x, out=x)
    assert isinstance(s, format)

    assert_eq(x, s)


# @pytest.mark.parametrize(
#     "shape1, shape2",
#     [
#         ((2, 3, 4), (3, 4)),
#         ((3, 4), (2, 3, 4)),
#         ((3, 1, 4), (3, 2, 4)),
#         ((1, 3, 4), (3, 4)),
#         ((3, 4, 1), (3, 4, 2)),
#         ((1, 5), (5, 1)),
#         ((3, 1), (3, 4)),
#         ((3, 1), (1, 4)),
#         ((1, 4), (3, 4)),
#         ((2, 2, 2), (1, 1, 1)),
#     ],
# )
@composite
def generate_shapes(draw, elements=st.integers(min_value=1, max_value=5)):
    shape = st.tuples(elements)
    return shape


@given(
    shape1=generate_shapes(),
    shape2=generate_shapes(),
    format=st.sampled_from([COO, GCXS, DOK]),
)
def test_elemwise_mixed(shape1, shape2, format):
    s1 = sparse.random(shape1, density=0.5, format=format)
    x2 = np.random.rand(*shape2)

    x1 = s1.todense()

    assert_eq(s1 * x2, x1 * x2)


@given(format=st.sampled_from([COO, GCXS, DOK]))
def test_elemwise_mixed_empty(format):
    s1 = sparse.random((2, 0, 4), density=0.5, format=format)
    x2 = np.random.rand(2, 0, 4)

    x1 = s1.todense()

    assert_eq(s1 * x2, x1 * x2)


@given(format=st.sampled_from([COO, GCXS, DOK]))
def test_elemwise_unsupported(format):
    class A:
        pass

    s1 = sparse.random((2, 3, 4), density=0.5, format=format)
    x2 = A()

    with pytest.raises(TypeError):
        s1 + x2

    assert sparse.elemwise(operator.add, s1, x2) is NotImplemented


@given(format=st.sampled_from([COO, GCXS, DOK]))
def test_elemwise_mixed_broadcast(format):
    s1 = sparse.random((2, 3, 4), density=0.5, format=format)
    s2 = sparse.random(4, density=0.5)
    x3 = np.random.rand(3, 4)

    x1 = s1.todense()
    x2 = s2.todense()

    def func(x1, x2, x3):
        return x1 * x2 * x3

    assert_eq(sparse.elemwise(func, s1, s2, x3), func(x1, x2, x3))


@given(
    func=st.sampled_from(
        [
            operator.mul,
            operator.add,
            operator.sub,
            operator.gt,
            operator.lt,
            operator.ne,
        ]
    ),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
    format=st.sampled_from([COO, GCXS, DOK]),
)
def test_elemwise_binary(func, shape, format):
    xs = sparse.random(shape, density=0.5, format=format)
    ys = sparse.random(shape, density=0.5, format=format)

    x = xs.todense()
    y = ys.todense()

    assert_eq(func(xs, ys), func(x, y))


@given(
    func=st.sampled_from([operator.imul, operator.iadd, operator.isub]),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
    format=st.sampled_from([COO, GCXS, DOK]),
)
def test_elemwise_binary_inplace(func, shape, format):
    xs = sparse.random(shape, density=0.5, format=format)
    ys = sparse.random(shape, density=0.5, format=format)

    x = xs.todense()
    y = ys.todense()

    xs = func(xs, ys)
    x = func(x, y)

    assert_eq(xs, x)


@given(
    func=st.sampled_from(
        [
            lambda x, y, z: x + y + z,
            lambda x, y, z: x * y * z,
            lambda x, y, z: x + y * z,
            lambda x, y, z: (x + y) * z,
        ],
    ),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
    formats=st.sampled_from(
        [
            [COO, COO, COO],
            [GCXS, GCXS, GCXS],
            [COO, GCXS, GCXS],
        ]
    ),
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


@given(func=st.sampled_from([operator.add, operator.mul]), sd=gen_bin_brdcst())
def test_binary_broadcasting(func, sd):
    shape1, shape2 = sd
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


@given(sd=gen_bin_brdcst())
def test_broadcast_to(sd):
    shape1, shape2 = sd
    a = sparse.random(shape1, density=0.5)
    x = a.todense()

    assert_eq(np.broadcast_to(x, shape2), a.broadcast_to(shape2))


@given(
    func=st.sampled_from(
        [
            lambda x, y, z: (x + y) * z,
            lambda x, y, z: x * (y + z),
            lambda x, y, z: x * y * z,
            lambda x, y, z: x + y + z,
            lambda x, y, z: x + y - z,
            lambda x, y, z: x - y + z,
        ]
    ),
    shapes=st.sampled_from(
        [
            [(2,), (3, 2), (4, 3, 2)],
            [(3,), (2, 3), (2, 2, 3)],
            [(2,), (2, 2), (2, 2, 2)],
            [(4,), (4, 4), (4, 4, 4)],
            [(4,), (4, 4), (4, 4, 4)],
            [(4,), (4, 4), (4, 4, 4)],
            [(1, 1, 2), (1, 3, 1), (4, 1, 1)],
            [(2,), (2, 1), (2, 1, 1)],
        ]
    ),
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
@given(
    value=st.sampled_from([np.nan, np.inf, -np.inf]),
    fraction=st.sampled_from([0.25, 0.5, 0.75, 1.0]),
)
@pytest.mark.filterwarnings("ignore:invalid value")
def test_trinary_broadcasting_pathological(shapes, func, value, fraction):
    args = [
        sparse.random(s, density=0.5, data_rvs=random_value_array(value, fraction))
        for s in shapes
    ]
    dense_args = [arg.todense() for arg in args]

    fs = sparse.elemwise(func, *args)
    assert isinstance(fs, COO)

    assert_eq(fs, func(*dense_args))


def test_sparse_broadcasting(monkeypatch):
    orig_unmatch_coo = sparse._umath._Elemwise._get_func_coords_data

    state = {"num_matches": 0}

    xs = sparse.random((3, 4), density=0.5)
    ys = sparse.random((3, 4), density=0.5)

    def mock_unmatch_coo(*args, **kwargs):
        result = orig_unmatch_coo(*args, **kwargs)
        if result is not None:
            state["num_matches"] += 1
        return result

    monkeypatch.setattr(
        sparse._umath._Elemwise, "_get_func_coords_data", mock_unmatch_coo
    )

    xs * ys

    # Less than in case there's absolutely no overlap in some cases.
    assert state["num_matches"] <= 1


def test_dense_broadcasting(monkeypatch):
    orig_unmatch_coo = sparse._umath._Elemwise._get_func_coords_data

    state = {"num_matches": 0}

    xs = sparse.random((3, 4), density=0.5)
    ys = sparse.random((3, 4), density=0.5)

    def mock_unmatch_coo(*args, **kwargs):
        result = orig_unmatch_coo(*args, **kwargs)
        if result is not None:
            state["num_matches"] += 1
        return result

    monkeypatch.setattr(
        sparse._umath._Elemwise, "_get_func_coords_data", mock_unmatch_coo
    )

    xs + ys

    # Less than in case there's absolutely no overlap in some cases.
    assert state["num_matches"] <= 3


@given(format=st.sampled_from(["coo", "dok", "gcxs"]))
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


def test_ndarray_densification_fails():
    xs = sparse.random((2, 3, 4), density=0.5)
    y = np.random.rand(3, 4)

    with pytest.raises(ValueError):
        xs + y


def test_elemwise_noargs():
    def func():
        return np.float_(5.0)

    assert_eq(sparse.elemwise(func), func())


@given(
    func=st.sampled_from(
        [
            operator.pow,
            operator.truediv,
            operator.floordiv,
            operator.ge,
            operator.le,
            operator.eq,
            operator.mod,
        ]
    ),
    format=st.sampled_from([COO, GCXS, DOK]),
)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value")
def test_nonzero_outout_fv_ufunc(func, format):
    xs = sparse.random((2, 3, 4), density=0.5, format=format)
    ys = sparse.random((2, 3, 4), density=0.5, format=format)

    x = xs.todense()
    y = ys.todense()

    f = func(x, y)
    fs = func(xs, ys)
    assert isinstance(fs, format)

    assert_eq(f, fs)


@given(
    convert_to_np_number=st.sampled_from([True, False]),
    format=st.sampled_from([COO, GCXS, DOK]),
    func=st.sampled_from(
        [
            (operator.mul),
            (operator.add),
            (operator.sub),
            (operator.pow),
            (operator.truediv),
            (operator.floordiv),
            (operator.gt),
            (operator.lt),
            (operator.ne),
            (operator.ge),
            (operator.le),
            (operator.eq),
            (operator.mod),
        ]
    ),
    scalar=st.integers(min_value=-10, max_value=10),
)
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


@given(
    func=st.sampled_from(
        [
            (operator.mul),
            (operator.add),
            (operator.sub),
            (operator.gt),
            (operator.lt),
            (operator.ne),
            (operator.ge),
            (operator.le),
            (operator.eq),
        ]
    ),
    scalar=st.integers(min_value=-10, max_value=10),
    convert_to_np_number=st.sampled_from([True, False]),
)
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


@given(
    func=st.sampled_from(
        [
            (operator.add),
            (operator.sub),
            (operator.pow),
            (operator.truediv),
            (operator.floordiv),
            (operator.gt),
            (operator.lt),
            (operator.ne),
            (operator.ge),
            (operator.le),
            (operator.eq),
        ]
    ),
    scalar=st.integers(min_value=-10, max_value=10),
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


@given(
    func=st.sampled_from([operator.and_, operator.or_, operator.xor]),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
    format=st.sampled_from([COO, GCXS, DOK]),
)
def test_bitwise_binary(func, shape, format):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5, format=format) * 100).astype(np.int_)
    ys = (sparse.random(shape, density=0.5, format=format) * 100).astype(np.int_)

    x = xs.todense()
    y = ys.todense()

    assert_eq(func(xs, ys), func(x, y))


@given(
    func=st.sampled_from([operator.iand, operator.ior, operator.ixor]),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
    format=st.sampled_from([COO, GCXS, DOK]),
)
def test_bitwise_binary_inplace(func, shape, format):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5, format=format) * 100).astype(np.int_)
    ys = (sparse.random(shape, density=0.5, format=format) * 100).astype(np.int_)

    x = xs.todense()
    y = ys.todense()

    xs = func(xs, ys)
    x = func(x, y)

    assert_eq(xs, x)


@given(
    func=st.sampled_from([operator.lshift, operator.rshift]),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
)
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


@given(
    func=st.sampled_from([operator.ilshift, operator.irshift]),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
)
def test_bitshift_binary_inplace(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int_)

    # Can't merge into test_bitwise_binary because left/right shifting
    # with something >= 64 isn't defined.
    ys = (sparse.random(shape, density=0.5) * 64).astype(np.int_)

    x = xs.todense()
    y = ys.todense()

    xs = func(xs, ys)
    x = func(x, y)

    assert_eq(xs, x)


@given(
    func=st.sampled_from([operator.and_]),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
)
def test_bitwise_scalar(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int_)
    y = np.random.randint(100)

    x = xs.todense()

    assert_eq(func(xs, y), func(x, y))
    assert_eq(func(y, xs), func(y, x))


@given(
    func=st.sampled_from([operator.lshift, operator.rshift]),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
)
def test_bitshift_scalar(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int_)

    # Can't merge into test_bitwise_binary because left/right shifting
    # with something >= 64 isn't defined.
    y = np.random.randint(64)

    x = xs.todense()

    assert_eq(func(xs, y), func(x, y))


@given(
    func=st.sampled_from([operator.invert]),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
)
def test_unary_bitwise_nonzero_output_fv(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int_)
    x = xs.todense()

    f = func(x)
    fs = func(xs)

    assert isinstance(fs, COO)
    assert fs.nnz <= xs.nnz

    assert_eq(f, fs)


@given(
    func=st.sampled_from([operator.or_, operator.xor]),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
)
def test_binary_bitwise_nonzero_output_fv(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    xs = (sparse.random(shape, density=0.5) * 100).astype(np.int_)
    y = np.random.randint(1, 100)

    x = xs.todense()

    f = func(x, y)
    fs = func(xs, y)

    assert isinstance(fs, COO)
    assert fs.nnz <= xs.nnz

    assert_eq(f, fs)


@given(
    func=st.sampled_from(
        [
            operator.mul,
            operator.add,
            operator.sub,
            operator.gt,
            operator.lt,
            operator.ne,
        ]
    ),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
)
def test_elemwise_nonzero_input_fv(func, shape):
    xs = sparse.random(shape, density=0.5, fill_value=np.random.rand())
    ys = sparse.random(shape, density=0.5, fill_value=np.random.rand())

    x = xs.todense()
    y = ys.todense()

    assert_eq(func(xs, ys), func(x, y))


@given(
    func=st.sampled_from([operator.lshift, operator.rshift]),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
)
def test_binary_bitshift_densification_fails(func, shape):
    # Small arrays need high density to have nnz entries
    # Casting floats to int will result in all zeros, hence the * 100
    x = np.random.randint(1, 100)
    ys = (sparse.random(shape, density=0.5) * 64).astype(np.int_)

    y = ys.todense()

    f = func(x, y)
    fs = func(x, ys)

    assert isinstance(fs, COO)
    assert fs.nnz <= ys.nnz

    assert_eq(f, fs)


@given(
    func=st.sampled_from([operator.and_, operator.or_, operator.xor]),
    shape=st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]),
)
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
