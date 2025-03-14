import contextlib
import operator
import pickle
import sys

import sparse
from sparse import COO, DOK
from sparse.numba_backend._settings import NEP18_ENABLED
from sparse.numba_backend._utils import assert_eq, html_table, random_value_array

import pytest

import numpy as np
import scipy.sparse
import scipy.stats


@pytest.fixture(scope="module", params=["f8", "f4", "i8", "i4"])
def random_sparse(request, rng):
    dtype = request.param
    if np.issubdtype(dtype, np.integer):

        def data_rvs(n):
            return rng.integers(-1000, 1000, n)

    else:
        data_rvs = None
    return sparse.random((20, 30, 40), density=0.25, data_rvs=data_rvs).astype(dtype)


@pytest.fixture(scope="module", params=["f8", "f4", "i8", "i4"])
def random_sparse_small(request, rng):
    dtype = request.param
    if np.issubdtype(dtype, np.integer):

        def data_rvs(n):
            return rng.integers(-10, 10, n)

    else:
        data_rvs = None
    return sparse.random((20, 30, 40), density=0.25, data_rvs=data_rvs).astype(dtype)


@pytest.mark.parametrize("reduction, kwargs", [("sum", {}), ("sum", {"dtype": np.float32}), ("prod", {})])
@pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 2), -3, (1, -1)])
@pytest.mark.parametrize("keepdims", [True, False])
def test_reductions_fv(reduction, random_sparse_small, axis, keepdims, kwargs, rng):
    x = random_sparse_small + rng.integers(-1, 1, dtype="i4")
    y = x.todense()
    xx = getattr(x, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    yy = getattr(y, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    assert_eq(xx, yy)


@pytest.mark.parametrize(
    "reduction, kwargs",
    [
        ("sum", {}),
        ("sum", {"dtype": np.float32}),
        ("mean", {}),
        ("mean", {"dtype": np.float32}),
        ("prod", {}),
        ("max", {}),
        ("min", {}),
        ("std", {}),
        ("var", {}),
    ],
)
@pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 2), -3, (1, -1)])
@pytest.mark.parametrize("keepdims", [True, False])
def test_reductions(reduction, random_sparse, axis, keepdims, kwargs):
    x = random_sparse
    y = x.todense()
    xx = getattr(x, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    yy = getattr(y, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    assert_eq(xx, yy)


@pytest.mark.xfail(reason=("Setting output dtype=float16 produces results inconsistent with numpy"))
@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.parametrize(
    "reduction, kwargs",
    [("sum", {"dtype": np.float16}), ("mean", {"dtype": np.float16})],
)
@pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 2)])
def test_reductions_float16(random_sparse, reduction, kwargs, axis):
    x = random_sparse
    y = x.todense()
    xx = getattr(x, reduction)(axis=axis, **kwargs)
    yy = getattr(y, reduction)(axis=axis, **kwargs)
    assert_eq(xx, yy, atol=1e-2)


@pytest.mark.parametrize("reduction,kwargs", [("any", {}), ("all", {})])
@pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 2), -3, (1, -1)])
@pytest.mark.parametrize("keepdims", [True, False])
def test_reductions_bool(random_sparse, reduction, kwargs, axis, keepdims):
    y = np.zeros((2, 3, 4), dtype=bool)
    y[0] = True
    y[1, 1, 1] = True
    x = sparse.COO.from_numpy(y)
    xx = getattr(x, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    yy = getattr(y, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    assert_eq(xx, yy)


@pytest.mark.parametrize(
    "reduction,kwargs",
    [
        (np.max, {}),
        (np.sum, {}),
        (np.sum, {"dtype": np.float32}),
        (np.mean, {}),
        (np.mean, {"dtype": np.float32}),
        (np.prod, {}),
        (np.min, {}),
    ],
)
@pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 2), -1, (0, -1)])
@pytest.mark.parametrize("keepdims", [True, False])
def test_ufunc_reductions(random_sparse, reduction, kwargs, axis, keepdims):
    x = random_sparse
    y = x.todense()
    xx = reduction(x, axis=axis, keepdims=keepdims, **kwargs)
    yy = reduction(y, axis=axis, keepdims=keepdims, **kwargs)
    assert_eq(xx, yy)
    # If not a scalar/1 element array, must be a sparse array
    if xx.size > 1:
        assert isinstance(xx, COO)


@pytest.mark.parametrize(
    "reduction,kwargs",
    [
        (np.max, {}),
        (np.sum, {"axis": 0}),
        (np.prod, {"keepdims": True}),
        (np.add.reduce, {}),
        (np.add.reduce, {"keepdims": True}),
        (np.minimum.reduce, {"axis": 0}),
    ],
)
def test_ufunc_reductions_kwargs(reduction, kwargs):
    x = sparse.random((2, 3, 4), density=0.5)
    y = x.todense()
    xx = reduction(x, **kwargs)
    yy = reduction(y, **kwargs)
    assert_eq(xx, yy)
    # If not a scalar/1 element array, must be a sparse array
    if xx.size > 1:
        assert isinstance(xx, COO)


@pytest.mark.parametrize("reduction", ["nansum", "nanmean", "nanprod", "nanmax", "nanmin"])
@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [False])
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.filterwarnings("ignore:All-NaN")
@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_nan_reductions(reduction, axis, keepdims, fraction):
    s = sparse.random((2, 3, 4), data_rvs=random_value_array(np.nan, fraction), density=0.25)
    x = s.todense()
    expected = getattr(np, reduction)(x, axis=axis, keepdims=keepdims)
    actual = getattr(sparse, reduction)(s, axis=axis, keepdims=keepdims)
    assert_eq(expected, actual)


@pytest.mark.parametrize("reduction", ["nanmax", "nanmin", "nanmean"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_all_nan_reduction_warning(reduction, axis):
    x = random_value_array(np.nan, 1.0)(2 * 3 * 4).reshape(2, 3, 4)
    s = COO.from_numpy(x)

    with pytest.warns(RuntimeWarning):
        getattr(sparse, reduction)(s, axis=axis)


@pytest.mark.parametrize(
    "axis",
    [None, (1, 2, 0), (2, 1, 0), (0, 1, 2), (0, 1, -1), (0, -2, -1), (-3, -2, -1)],
)
def test_transpose(axis):
    x = sparse.random((2, 3, 4), density=0.25)
    y = x.todense()
    xx = x.transpose(axis)
    yy = y.transpose(axis)
    assert_eq(xx, yy)


@pytest.mark.parametrize(
    "axis",
    [
        (0, 1),  # too few
        (0, 1, 2, 3),  # too many
        (3, 1, 0),  # axis 3 illegal
        (0, -1, -4),  # axis -4 illegal
        (0, 0, 1),  # duplicate axis 0
        (0, -1, 2),  # duplicate axis -1 == 2
        0.3,  # Invalid type in axis
        ((0, 1, 2),),  # Iterable inside iterable
    ],
)
def test_transpose_error(axis):
    x = sparse.random((2, 3, 4), density=0.25)

    with pytest.raises(ValueError):
        x.transpose(axis)


@pytest.mark.parametrize("axis1", [-3, -2, -1, 0, 1, 2])
@pytest.mark.parametrize("axis2", [-3, -2, -1, 0, 1, 2])
def test_swapaxes(axis1, axis2):
    x = sparse.random((2, 3, 4), density=0.25)
    y = x.todense()
    xx = x.swapaxes(axis1, axis2)
    yy = y.swapaxes(axis1, axis2)
    assert_eq(xx, yy)


@pytest.mark.parametrize("axis1", [-4, 3])
@pytest.mark.parametrize("axis2", [-4, 3, 0])
def test_swapaxes_error(axis1, axis2):
    x = sparse.random((2, 3, 4), density=0.25)

    with pytest.raises(ValueError):
        x.swapaxes(axis1, axis2)


@pytest.mark.parametrize(
    "source, destination",
    [
        [0, 1],
        [2, 1],
        [-2, 1],
        [-2, -3],
        [(0, 1), (2, 3)],
        [(-1, 0), (0, 1)],
        [(0, 1, 2), (2, 1, 0)],
        [(0, 1, 2), (-2, -3, -1)],
    ],
)
def test_moveaxis(source, destination):
    x = sparse.random((2, 3, 4, 5), density=0.25)
    y = x.todense()
    xx = sparse.moveaxis(x, source, destination)
    yy = np.moveaxis(y, source, destination)
    assert_eq(xx, yy)


@pytest.mark.parametrize("source, destination", [[0, -4], [(0, 5), (1, 2)], [(0, 1, 2), (2, 1)]])
def test_moveaxis_error(source, destination):
    x = sparse.random((2, 3, 4), density=0.25)

    with pytest.raises(ValueError):
        sparse.moveaxis(x, source, destination)


@pytest.mark.parametrize(
    "a,b",
    [
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
    ],
)
@pytest.mark.parametrize("format", ["coo", "dok"])
def test_reshape(a, b, format):
    s = sparse.random(a, density=0.5, format=format)
    x = s.todense()

    assert_eq(x.reshape(b), s.reshape(b))


def test_large_reshape():
    n = 100
    m = 10
    row = np.arange(n, dtype=np.uint16)
    col = row % m
    data = np.ones(n, dtype=np.uint8)

    x = COO((data, (row, col)), shape=(100, 10), sorted=True, has_duplicates=False)
    assert_eq(x, x.reshape(x.shape))


def test_reshape_same():
    s = sparse.random((3, 5), density=0.5)

    assert s.reshape(s.shape) is s


@pytest.mark.parametrize("format", [COO, DOK])
def test_reshape_function(format):
    s = sparse.random((5, 3), density=0.5, format=format)
    x = s.todense()
    shape = (3, 5)

    s2 = np.reshape(s, shape)
    assert isinstance(s2, format)
    assert_eq(s2, x.reshape(shape))


def test_reshape_upcast():
    a = sparse.random((10, 10, 10), density=0.5, format="coo", idx_dtype=np.uint8)
    assert a.reshape(1000).coords.dtype == np.uint16


@pytest.mark.parametrize("format", [COO, DOK])
def test_reshape_errors(format):
    s = sparse.random((5, 3), density=0.5, format=format)
    with pytest.raises(NotImplementedError):
        s.reshape((3, 5, 1), order="F")


@pytest.mark.parametrize("a_ndim", [1, 2, 3])
@pytest.mark.parametrize("b_ndim", [1, 2, 3])
def test_kron(a_ndim, b_ndim):
    a_shape = (2, 3, 4)[:a_ndim]
    b_shape = (5, 6, 7)[:b_ndim]

    sa = sparse.random(a_shape, density=0.5)
    a = sa.todense()
    sb = sparse.random(b_shape, density=0.5)
    b = sb.todense()

    sol = np.kron(a, b)
    assert_eq(sparse.kron(sa, sb), sol)
    assert_eq(sparse.kron(sa, b), sol)
    assert_eq(sparse.kron(a, sb), sol)

    with pytest.raises(ValueError):
        assert_eq(sparse.kron(a, b), sol)


@pytest.mark.parametrize("a_spmatrix, b_spmatrix", [(True, True), (True, False), (False, True)])
def test_kron_spmatrix(a_spmatrix, b_spmatrix):
    sa = sparse.random((3, 4), density=0.5)
    a = sa.todense()
    sb = sparse.random((5, 6), density=0.5)
    b = sb.todense()

    if a_spmatrix:
        sa = sa.tocsr()

    if b_spmatrix:
        sb = sb.tocsr()

    sol = np.kron(a, b)
    assert_eq(sparse.kron(sa, sb), sol)
    assert_eq(sparse.kron(sa, b), sol)
    assert_eq(sparse.kron(a, sb), sol)

    with pytest.raises(ValueError):
        assert_eq(sparse.kron(a, b), sol)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_kron_scalar(ndim):
    if ndim:
        a_shape = (3, 4, 5)[:ndim]
        sa = sparse.random(a_shape, density=0.5)
        a = sa.todense()
    else:
        sa = a = np.array(6)
    scalar = np.array(5)

    sol = np.kron(a, scalar)
    assert_eq(sparse.kron(sa, scalar), sol)
    assert_eq(sparse.kron(scalar, sa), sol)


def test_gt():
    s = sparse.random((2, 3, 4), density=0.5)
    x = s.todense()

    m = x.mean()
    assert_eq(x > m, s > m)

    m = s.data[2]
    assert_eq(x > m, s > m)
    assert_eq(x >= m, s >= m)


@pytest.mark.parametrize(
    "index",
    [
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
        (0, slice(0, 2)),
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
        # With multi-axis advanced indexing
        ([0, 1],) * 2,
        ([0, 1], [0, 2]),
        ([0, 0, 0], [0, 1, 2], [1, 2, 1]),
        # Pathological - Slices larger than array
        (slice(None, 1000)),
        (slice(None), slice(None, 1000)),
        (slice(None), slice(1000, -1000, -1)),
        (slice(None), slice(1000, -1000, -50)),
        # Pathological - Wrong ordering of start/stop
        (slice(5, 0),),
        (slice(0, 5, -1),),
        (slice(0, 0, None),),
    ],
)
def test_slicing(index):
    s = sparse.random((2, 3, 4), density=0.5)
    x = s.todense()

    assert_eq(x[index], s[index])


@pytest.mark.parametrize(
    "index",
    [
        ([1, 0], 0),
        (1, [0, 2]),
        (0, [1, 0], 0),
        (1, [2, 0], 0),
        (1, [], 0),
        ([True, False], slice(1, None), slice(-2, None)),
        (slice(1, None), slice(-2, None), [True, False, True, False]),
        ([1, 0],),
        (Ellipsis, [2, 1, 3]),
        (slice(None), [2, 1, 2]),
        (1, [2, 0, 1]),
    ],
)
def test_advanced_indexing(index):
    s = sparse.random((2, 3, 4), density=0.5)
    x = s.todense()

    assert_eq(x[index], s[index])


def test_custom_dtype_slicing():
    dt = np.dtype([("part1", np.float64), ("part2", np.int64, (2,)), ("part3", np.int64, (2, 2))])

    x = np.zeros((2, 3, 4), dtype=dt)
    x[1, 1, 1] = (0.64, [4, 2], [[1, 2], [3, 0]])

    s = COO.from_numpy(x)

    assert x[1, 1, 1] == s[1, 1, 1]
    assert x[0, 1, 2] == s[0, 1, 2]

    assert_eq(x["part1"], s["part1"])
    assert_eq(x["part2"], s["part2"])
    assert_eq(x["part3"], s["part3"])


@pytest.mark.parametrize(
    "index",
    [
        (Ellipsis, Ellipsis),
        (1, 1, 1, 1),
        (slice(None),) * 4,
        5,
        -5,
        "foo",
        [True, False, False],
        0.5,
        [0.5],
        {"potato": "kartoffel"},
        ([[0, 1]],),
    ],
)
def test_slicing_errors(index):
    s = sparse.random((2, 3, 4), density=0.5)

    with pytest.raises(IndexError):
        s[index]


def test_concatenate():
    xx = sparse.random((2, 3, 4), density=0.5)
    x = xx.todense()
    yy = sparse.random((5, 3, 4), density=0.5)
    y = yy.todense()
    zz = sparse.random((4, 3, 4), density=0.5)
    z = zz.todense()

    assert_eq(np.concatenate([x, y, z], axis=0), sparse.concatenate([xx, yy, zz], axis=0))

    xx = sparse.random((5, 3, 1), density=0.5)
    x = xx.todense()
    yy = sparse.random((5, 3, 3), density=0.5)
    y = yy.todense()
    zz = sparse.random((5, 3, 2), density=0.5)
    z = zz.todense()

    assert_eq(np.concatenate([x, y, z], axis=2), sparse.concatenate([xx, yy, zz], axis=2))

    assert_eq(np.concatenate([x, y, z], axis=-1), sparse.concatenate([xx, yy, zz], axis=-1))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("func", [sparse.stack, sparse.concatenate])
def test_concatenate_mixed(func, axis):
    s = sparse.random((10, 10), density=0.5)
    d = s.todense()

    with pytest.raises(ValueError):
        func([d, s, s], axis=axis)


def test_concatenate_noarrays():
    with pytest.raises(ValueError):
        sparse.concatenate([])


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4), (5, 2)])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_stack(shape, axis):
    xx = sparse.random(shape, density=0.5)
    x = xx.todense()
    yy = sparse.random(shape, density=0.5)
    y = yy.todense()
    zz = sparse.random(shape, density=0.5)
    z = zz.todense()

    assert_eq(np.stack([x, y, z], axis=axis), sparse.stack([xx, yy, zz], axis=axis))


def test_large_concat_stack():
    data = np.array([1], dtype=np.uint8)
    coords = np.array([[255]], dtype=np.uint8)

    xs = COO(coords, data, shape=(256,), has_duplicates=False, sorted=True)
    x = xs.todense()

    assert_eq(np.stack([x, x]), sparse.stack([xs, xs]))

    assert_eq(np.concatenate((x, x)), sparse.concatenate((xs, xs)))


def test_addition():
    a = sparse.random((2, 3, 4), density=0.5)
    x = a.todense()

    b = sparse.random((2, 3, 4), density=0.5)
    y = b.todense()

    assert_eq(x + y, a + b)
    assert_eq(x - y, a - b)


@pytest.mark.parametrize("scalar", [2, 2.5, np.float32(2.0), np.int8(3)])
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


@pytest.mark.filterwarnings("ignore:divide by zero")
def test_scalar_exponentiation():
    a = sparse.random((2, 3, 4), density=0.5)
    x = a.todense()

    assert_eq(x**2, a**2)
    assert_eq(x**0.5, a**0.5)

    assert_eq(x**-1, a**-1)


def test_create_with_lists_of_tuples():
    L = [((0, 0, 0), 1), ((1, 2, 1), 1), ((1, 1, 1), 2), ((1, 3, 2), 3)]

    s = COO(L, shape=(2, 4, 3))

    x = np.zeros((2, 4, 3), dtype=np.asarray([1, 2, 3]).dtype)
    for ind, value in L:
        x[ind] = value

    assert_eq(s, x)


def test_sizeof():
    x = np.eye(100)
    y = COO.from_numpy(x)

    nb = sys.getsizeof(y)
    assert 400 < nb < x.nbytes / 10


def test_scipy_sparse_interface(rng):
    n = 100
    m = 10
    row = rng.integers(0, n, size=n, dtype=np.uint16)
    col = rng.integers(0, m, size=n, dtype=np.uint16)
    data = np.ones(n, dtype=np.uint8)

    inp = (data, (row, col))

    x = scipy.sparse.coo_matrix(inp, shape=(n, m))
    xx = sparse.COO(inp, shape=(n, m))

    assert_eq(x, xx, check_nnz=False)
    assert_eq(x.T, xx.T, check_nnz=False)
    assert_eq(xx.to_scipy_sparse(), x, check_nnz=False)
    assert_eq(COO.from_scipy_sparse(xx.to_scipy_sparse()), xx, check_nnz=False)

    assert_eq(x, xx, check_nnz=False)
    assert_eq(x.T.dot(x), xx.T.dot(xx), check_nnz=False)
    assert isinstance(x + xx, COO)
    assert isinstance(xx + x, COO)


@pytest.mark.parametrize("scipy_format", ["coo", "csr", "dok", "csc"])
def test_scipy_sparse_interaction(scipy_format):
    x = sparse.random((10, 20), density=0.2).todense()
    sp = getattr(scipy.sparse, scipy_format + "_matrix")(x)
    coo = COO(x)
    assert isinstance(sp + coo, COO)
    assert isinstance(coo + sp, COO)
    assert_eq(sp, coo)


@pytest.mark.parametrize(
    "func",
    [operator.mul, operator.add, operator.sub, operator.gt, operator.lt, operator.ne],
)
def test_op_scipy_sparse(func):
    xs = sparse.random((3, 4), density=0.5)
    y = sparse.random((3, 4), density=0.5).todense()

    ys = scipy.sparse.csr_matrix(y)
    x = xs.todense()

    assert_eq(func(x, y), func(xs, ys))


@pytest.mark.parametrize(
    "func",
    [
        operator.add,
        operator.sub,
        pytest.param(
            operator.mul,
            marks=pytest.mark.xfail(reason="Scipy sparse auto-densifies in this case."),
        ),
        pytest.param(
            operator.gt,
            marks=pytest.mark.xfail(reason="Scipy sparse doesn't support this yet."),
        ),
        pytest.param(
            operator.lt,
            marks=pytest.mark.xfail(reason="Scipy sparse doesn't support this yet."),
        ),
        pytest.param(
            operator.ne,
            marks=pytest.mark.xfail(reason="Scipy sparse doesn't support this yet."),
        ),
    ],
)
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


def test_single_dimension():
    x = COO([1, 3], [1.0, 3.0], shape=(4,))
    assert_eq(x, np.array([0, 1.0, 0, 3.0]))


def test_large_sum(rng):
    n = 500000
    x = rng.integers(0, 10000, size=(n,))
    y = rng.integers(0, 1000, size=(n,))
    z = rng.integers(0, 3, size=(n,))

    data = rng.random(n)

    a = COO((x, y, z), data, shape=(10000, 1000, 3))

    b = a.sum(axis=2)
    assert b.nnz > 100000


def test_add_many_sparse_arrays():
    x = COO({(1, 1): 1}, shape=(2, 2))
    y = sum([x] * 100)
    assert y.nnz < np.prod(y.shape)


def test_caching():
    x = COO({(9, 9, 9): 1}, shape=(10, 10, 10))
    assert x[:].reshape((100, 10)).transpose().tocsr() is not x[:].reshape((100, 10)).transpose().tocsr()

    x = COO({(9, 9, 9): 1}, shape=(10, 10, 10), cache=True)
    assert x[:].reshape((100, 10)).transpose().tocsr() is x[:].reshape((100, 10)).transpose().tocsr()

    x = COO({(1, 1, 1, 1, 1, 1, 1, 2): 1}, shape=(2, 2, 2, 2, 2, 2, 2, 3), cache=True)

    for _ in range(x.ndim):
        x.reshape(x.size)

    assert len(x._cache["reshape"]) < 5


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


@pytest.mark.parametrize(
    "shape, k",
    [((3, 4), 0), ((3, 4, 5), 1), ((4, 2), -1), ((2, 4), -2), ((4, 4), 1000)],
)
def test_triul(shape, k):
    s = sparse.random(shape, density=0.5)
    x = s.todense()

    assert_eq(np.triu(x, k), sparse.triu(s, k))
    assert_eq(np.tril(x, k), sparse.tril(s, k))


def test_empty_reduction():
    x = np.zeros((2, 3, 4), dtype=np.float64)
    xs = COO.from_numpy(x)

    assert_eq(x.sum(axis=(0, 2)), xs.sum(axis=(0, 2)))


@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("density", [0.1, 0.3, 0.5, 0.7])
def test_random_shape(shape, density):
    s = sparse.random(shape, density)

    assert isinstance(s, COO)

    assert s.shape == shape
    expected_nnz = density * np.prod(shape)
    assert np.floor(expected_nnz) <= s.nnz <= np.ceil(expected_nnz)


@pytest.mark.parametrize("shape, nnz", [((1,), 1), ((2,), 0), ((3, 4), 5)])
def test_random_nnz(shape, nnz):
    s = sparse.random(shape, nnz=nnz)

    assert isinstance(s, COO)

    assert s.nnz == nnz


@pytest.mark.parametrize("density, nnz", [(1, 1), (1.01, None), (-0.01, None), (None, 2)])
def test_random_invalid_density_and_nnz(density, nnz):
    with pytest.raises(ValueError):
        sparse.random((1,), density, nnz=nnz)


def test_two_random_unequal():
    s1 = sparse.random((2, 3, 4), 0.3)
    s2 = sparse.random((2, 3, 4), 0.3)

    assert not np.allclose(s1.todense(), s2.todense())


def test_two_random_same_seed(rng):
    state = rng.integers(100)
    s1 = sparse.random((2, 3, 4), 0.3, random_state=state)
    s2 = sparse.random((2, 3, 4), 0.3, random_state=state)

    assert_eq(s1, s2)


@pytest.mark.parametrize(
    "rvs, dtype",
    [
        (None, np.float64),
        (scipy.stats.poisson(25, loc=10).rvs, np.int64),
        (lambda x: np.random.default_rng().choice([True, False], size=x), np.bool_),
    ],
)
@pytest.mark.parametrize("shape", [(2, 4, 5), (20, 40, 50)])
@pytest.mark.parametrize("density", [0.0, 0.01, 0.1, 0.2])
def test_random_rvs(rvs, dtype, shape, density):
    x = sparse.random(shape, density, data_rvs=rvs)
    assert x.shape == shape
    assert x.dtype == dtype


@pytest.mark.parametrize("format", ["coo", "dok"])
def test_random_fv(format, rng):
    fv = rng.random()
    s = sparse.random((2, 3, 4), density=0.5, format=format, fill_value=fv)

    assert s.fill_value == fv


def test_scalar_shape_construction(rng):
    x = rng.random(5)
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

    with pytest.raises(RuntimeError):
        np.array(s)


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
        [(3,), (), (2, 3)],
        [(4, 4), (), ()],
    ],
)
def test_three_arg_where(shapes):
    cs = sparse.random(shapes[0], density=0.5).astype(np.bool_)
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

    for e, a in zip(expected, actual, strict=True):
        assert_eq(e, a, compare_dtype=False)


def test_one_arg_where_dense(rng):
    x = rng.random((2, 3, 4))

    with pytest.raises(ValueError):
        sparse.where(x)


def test_two_arg_where():
    cs = sparse.random((2, 3, 4), density=0.5).astype(np.bool_)
    xs = sparse.random((2, 3, 4), density=0.5)

    with pytest.raises(ValueError):
        sparse.where(cs, xs)


@pytest.mark.parametrize("func", [operator.imul, operator.iadd, operator.isub])
def test_inplace_invalid_shape(func):
    xs = sparse.random((3, 4), density=0.5)
    ys = sparse.random((2, 3, 4), density=0.5)

    with pytest.raises(ValueError):
        func(xs, ys)


def test_nonzero():
    s = sparse.random((2, 3, 4), density=0.5)
    x = s.todense()

    expected = x.nonzero()
    actual = s.nonzero()

    assert isinstance(actual, tuple)
    assert len(expected) == len(actual)

    for e, a in zip(expected, actual, strict=True):
        assert_eq(e, a, compare_dtype=False)


def test_argwhere():
    s = sparse.random((2, 3, 4), density=0.5)
    x = s.todense()

    assert_eq(np.argwhere(s), np.argwhere(x), compare_dtype=False)


@pytest.mark.parametrize("format", ["coo", "dok"])
def test_asformat(format):
    s = sparse.random((2, 3, 4), density=0.5, format="coo")
    s2 = s.asformat(format)

    assert_eq(s, s2)


@pytest.mark.parametrize("format", [sparse.COO, sparse.DOK, scipy.sparse.csr_matrix, np.asarray])
def test_as_coo(format):
    x = format(sparse.random((3, 4), density=0.5, format="coo").todense())

    s1 = sparse.as_coo(x)
    s2 = COO(x)

    assert_eq(x, s1)
    assert_eq(x, s2)


def test_invalid_attrs_error():
    s = sparse.random((3, 4), density=0.5, format="coo")

    with pytest.raises(ValueError):
        sparse.as_coo(s, shape=(2, 3))

    with pytest.raises(ValueError):
        COO(s, shape=(2, 3))

    with pytest.raises(ValueError):
        sparse.as_coo(s, fill_value=0.0)


def test_invalid_iterable_error():
    with pytest.raises(ValueError):
        x = [(3, 4, 5)]
        COO.from_iter(x, shape=(6,))

    with pytest.raises(ValueError):
        x = [((2.3, 4.5), 3.2)]
        COO.from_iter(x, shape=(5,))

    with pytest.raises(TypeError):
        COO.from_iter({(1, 1): 1})


def test_prod_along_axis():
    s1 = sparse.random((10, 10), density=0.1)
    s2 = 1 - s1

    x1 = s1.todense()
    x2 = s2.todense()

    assert_eq(s1.prod(axis=0), x1.prod(axis=0))
    assert_eq(s2.prod(axis=0), x2.prod(axis=0))


class TestRoll:
    # test on 1d array #
    @pytest.mark.parametrize("shift", [0, 2, -2, 20, -20])
    def test_1d(self, shift):
        xs = sparse.random((100,), density=0.5)
        x = xs.todense()
        assert_eq(np.roll(x, shift), sparse.roll(xs, shift))
        assert_eq(np.roll(x, shift), sparse.roll(x, shift))

    # test on 2d array #
    @pytest.mark.parametrize("shift", [0, 2, -2, 20, -20])
    @pytest.mark.parametrize("ax", [None, 0, 1, (0, 1)])
    def test_2d(self, shift, ax):
        xs = sparse.random((10, 10), density=0.5)
        x = xs.todense()
        assert_eq(np.roll(x, shift, axis=ax), sparse.roll(xs, shift, axis=ax))
        assert_eq(np.roll(x, shift, axis=ax), sparse.roll(x, shift, axis=ax))

    # test on rolling multiple axes at once #
    @pytest.mark.parametrize("shift", [(0, 0), (1, -1), (-1, 1), (10, -10)])
    @pytest.mark.parametrize("ax", [(0, 1), (0, 2), (1, 2), (-1, 1)])
    def test_multiaxis(self, shift, ax):
        xs = sparse.random((9, 9, 9), density=0.5)
        x = xs.todense()
        assert_eq(np.roll(x, shift, axis=ax), sparse.roll(xs, shift, axis=ax))
        assert_eq(np.roll(x, shift, axis=ax), sparse.roll(x, shift, axis=ax))

    # test original is unchanged #
    @pytest.mark.parametrize("shift", [0, 2, -2, 20, -20])
    @pytest.mark.parametrize("ax", [None, 0, 1, (0, 1)])
    def test_original_is_copied(self, shift, ax):
        xs = sparse.random((10, 10), density=0.5)
        xc = COO(np.copy(xs.coords), np.copy(xs.data), shape=xs.shape)
        sparse.roll(xs, shift, axis=ax)
        assert_eq(xs, xc)

    # test on empty array #
    def test_empty(self):
        x = np.array([])
        assert_eq(np.roll(x, 1), sparse.roll(sparse.as_coo(x), 1))

    # test error handling #
    @pytest.mark.parametrize(
        "args",
        [
            # iterable shift, but axis not iterable
            ((1, 1), 0),
            # ndim(axis) != 1
            (1, [[0, 1]]),
            # ndim(shift) != 1
            ([[0, 1]], [0, 1]),
            ([[0, 1], [0, 1]], [0, 1]),
        ],
    )
    def test_valerr(self, args):
        x = sparse.random((2, 2, 2), density=1)
        with pytest.raises(ValueError):
            sparse.roll(x, *args)

    @pytest.mark.parametrize("dtype", [np.uint8, np.int8])
    @pytest.mark.parametrize("shift", [300, -300])
    def test_dtype_errors(self, dtype, shift):
        x = sparse.random((5, 5, 5), density=0.2, idx_dtype=dtype)
        with pytest.raises(ValueError):
            sparse.roll(x, shift)

    def test_unsigned_type_error(self):
        x = sparse.random((5, 5, 5), density=0.3, idx_dtype=np.uint8)
        with pytest.raises(ValueError):
            sparse.roll(x, -1)


def test_clip():
    x = np.array([[0, 0, 1, 0, 2], [5, 0, 0, 3, 0]])

    s = sparse.COO.from_numpy(x)

    assert_eq(s.clip(min=1), x.clip(min=1))
    assert_eq(s.clip(max=3), x.clip(max=3))
    assert_eq(s.clip(min=1, max=3), x.clip(min=1, max=3))
    assert_eq(s.clip(min=1, max=3.0), x.clip(min=1, max=3.0))

    assert_eq(np.clip(s, 1, 3), np.clip(x, 1, 3))

    with pytest.raises(ValueError):
        s.clip()

    out = sparse.COO.from_numpy(np.zeros_like(x))
    out2 = s.clip(min=1, max=3, out=out)
    assert out is out2
    assert_eq(out, x.clip(min=1, max=3))


class TestFailFillValue:
    # Check failed fill_value op
    def test_nonzero_fv(self):
        xs = sparse.random((2, 3), density=0.5, fill_value=1)
        ys = sparse.random((3, 4), density=0.5)

        with pytest.raises(ValueError):
            sparse.dot(xs, ys)

    def test_inconsistent_fv(self):
        xs = sparse.random((3, 4), density=0.5, fill_value=1)
        ys = sparse.random((3, 4), density=0.5, fill_value=2)

        with pytest.raises(ValueError):
            sparse.concatenate([xs, ys])


def test_pickle():
    x = sparse.COO.from_numpy([1, 0, 0, 0, 0]).reshape((5, 1))
    # Enable caching and add some data to it
    x.enable_caching()
    x.T  # noqa: B018
    assert x._cache is not None
    # Pickle sends data but not cache
    x2 = pickle.loads(pickle.dumps(x))
    assert_eq(x, x2)
    assert x2._cache is None


@pytest.mark.parametrize("deep", [True, False])
def test_copy(deep):
    x = sparse.COO.from_numpy([1, 0, 0, 0, 0]).reshape((5, 1))
    # Enable caching and add some data to it
    x.enable_caching()
    x.T  # noqa: B018
    assert x._cache is not None

    x2 = x.copy(deep)
    assert_eq(x, x2)
    assert (x2.data is x.data) is not deep
    assert (x2.coords is x.coords) is not deep
    assert x2._cache is None


@pytest.mark.parametrize("ndim", [2, 3, 4, 5])
def test_initialization(ndim, rng):
    shape = [10] * ndim
    shape[1] *= 2
    shape = tuple(shape)

    coords = rng.integers(10, size=(ndim, 20))
    data = rng.random(20)
    COO(coords, data=data, shape=shape)

    with pytest.raises(ValueError, match="data length"):
        COO(coords, data=data[:5], shape=shape)
    with pytest.raises(ValueError, match="shape of `coords`"):
        coords = rng.integers(10, size=(1, 20))
        COO(coords, data=data, shape=shape)


@pytest.mark.parametrize("N, M", [(4, None), (4, 10), (10, 4), (0, 10)])
def test_eye(N, M):
    m = M or N
    for k in [0, N - 2, N + 2, m - 2, m + 2, np.iinfo(np.intp).min]:
        assert_eq(sparse.eye(N, M=M, k=k), np.eye(N, M=M, k=k))
        assert_eq(sparse.eye(N, M=M, k=k, dtype="i4"), np.eye(N, M=M, k=k, dtype="i4"))


@pytest.mark.parametrize("from_", [np.int8, np.int64, np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("to", [np.int8, np.int64, np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("casting", ["no", "safe", "same_kind"])
def test_can_cast(from_, to, casting):
    assert sparse.can_cast(sparse.zeros((2, 2), dtype=from_), to, casting=casting) == np.can_cast(
        np.zeros((2, 2), dtype=from_), to, casting=casting
    )
    assert sparse.can_cast(from_, to, casting=casting) == np.can_cast(from_, to, casting=casting)


@pytest.mark.parametrize("funcname", ["ones", "zeros"])
def test_ones_zeros(funcname):
    sp_func = getattr(sparse, funcname)
    np_func = getattr(np, funcname)

    assert_eq(sp_func(5), np_func(5))
    assert_eq(sp_func((5, 4)), np_func((5, 4)))
    assert_eq(sp_func((5, 4), dtype="i4"), np_func((5, 4), dtype="i4"))
    assert_eq(sp_func((5, 4), dtype=None), np_func((5, 4), dtype=None))


@pytest.mark.parametrize("funcname", ["ones_like", "zeros_like"])
def test_ones_zeros_like(funcname):
    sp_func = getattr(sparse, funcname)
    np_func = getattr(np, funcname)

    x = np.ones((5, 5), dtype="i8")

    assert_eq(sp_func(x), np_func(x))
    assert_eq(sp_func(x, dtype="f8"), np_func(x, dtype="f8"))
    assert_eq(sp_func(x, dtype=None), np_func(x, dtype=None))
    assert_eq(sp_func(x, shape=(2, 2)), np_func(x, shape=(2, 2)))


def test_full():
    assert_eq(sparse.full(5, 9), np.full(5, 9))
    assert_eq(sparse.full(5, 9, dtype="f8"), np.full(5, 9, dtype="f8"))
    assert_eq(sparse.full((5, 4), 9.5), np.full((5, 4), 9.5))
    assert_eq(sparse.full((5, 4), 9.5, dtype="i4"), np.full((5, 4), 9.5, dtype="i4"))


def test_full_like():
    x = np.zeros((5, 5), dtype="i8")
    assert_eq(sparse.full_like(x, 9.5), np.full_like(x, 9.5))
    assert_eq(sparse.full_like(x, 9.5, dtype="f8"), np.full_like(x, 9.5, dtype="f8"))
    assert_eq(sparse.full_like(x, 9.5, shape=(2, 2)), np.full_like(x, 9.5, shape=(2, 2)))


@pytest.mark.parametrize(
    "x",
    [
        np.array([1, 2, 0, 0, 0]),
        np.array([1 + 2j, 2 - 1j, 0, 1, 0]),
        np.array(["a", "b", "c"]),
    ],
)
def test_complex_methods(x):
    s = sparse.COO.from_numpy(x)
    assert_eq(s.imag, x.imag)
    assert_eq(s.real, x.real)

    if np.issubdtype(s.dtype, np.number):
        assert_eq(s.conj(), x.conj())


def test_np_matrix(rng):
    x = rng.random((10, 1)).view(type=np.matrix)
    s = sparse.COO.from_numpy(x)

    assert_eq(x, s)


def test_out_dtype():
    a = sparse.eye(5, dtype="float32")
    b = sparse.eye(5, dtype="float64")

    assert np.positive(a, out=b).dtype == np.positive(a.todense(), out=b.todense()).dtype
    assert (
        np.positive(a, out=b, dtype="float64").dtype == np.positive(a.todense(), out=b.todense(), dtype="float64").dtype
    )


@contextlib.contextmanager
def auto_densify():
    "For use in tests only! Not threadsafe."
    import os
    from importlib import reload

    os.environ["SPARSE_AUTO_DENSIFY"] = "1"
    reload(sparse.numba_backend._settings)
    yield
    del os.environ["SPARSE_AUTO_DENSIFY"]
    reload(sparse.numba_backend._settings)


def test_setting_into_numpy_slice():
    actual = np.zeros((5, 5))
    s = sparse.COO(data=[1, 1], coords=(2, 4), shape=(5,))
    # This calls s.__array__(dtype('float64')) which means that __array__
    # must accept a positional argument. If not this will raise, of course,
    # TypeError: __array__() takes 1 positional argument but 2 were given
    with auto_densify():
        actual[:, 0] = s

    # Might as well check the content of the result as well.
    expected = np.zeros((5, 5))
    expected[:, 0] = s.todense()
    assert_eq(actual, expected)

    # Without densification, setting is unsupported.
    with pytest.raises(RuntimeError):
        actual[:, 0] = s


def test_successful_densification():
    s = sparse.random((3, 4, 5), density=0.5)
    with auto_densify():
        x = np.array(s)

    assert isinstance(x, np.ndarray)
    assert_eq(s, x)


def test_failed_densification():
    s = sparse.random((3, 4, 5), density=0.5)
    with pytest.raises(RuntimeError):
        np.array(s)


def test_warn_on_too_dense():
    import os
    from importlib import reload

    os.environ["SPARSE_WARN_ON_TOO_DENSE"] = "1"
    reload(sparse.numba_backend._settings)

    with pytest.warns(RuntimeWarning):
        sparse.random((3, 4, 5), density=1.0)

    del os.environ["SPARSE_WARN_ON_TOO_DENSE"]
    reload(sparse.numba_backend._settings)


def test_prune_coo():
    coords = np.array([[0, 1, 2, 3]])
    data = np.array([1, 0, 1, 2])
    s1 = COO(coords, data, shape=(4,))
    s2 = COO(coords, data, shape=(4,), prune=True)
    assert s2.nnz == 3

    # Densify s1 because it isn't canonical
    assert_eq(s1.todense(), s2, check_nnz=False)


def test_diagonal():
    a = sparse.random((4, 4), density=0.5)

    assert_eq(sparse.diagonal(a, offset=0), np.diagonal(a.todense(), offset=0))
    assert_eq(sparse.diagonal(a, offset=1), np.diagonal(a.todense(), offset=1))
    assert_eq(sparse.diagonal(a, offset=2), np.diagonal(a.todense(), offset=2))

    a = sparse.random((4, 5, 4, 6), density=0.5)

    assert_eq(
        sparse.diagonal(a, offset=0, axis1=0, axis2=2),
        np.diagonal(a.todense(), offset=0, axis1=0, axis2=2),
    )

    assert_eq(
        sparse.diagonal(a, offset=1, axis1=0, axis2=2),
        np.diagonal(a.todense(), offset=1, axis1=0, axis2=2),
    )

    assert_eq(
        sparse.diagonal(a, offset=2, axis1=0, axis2=2),
        np.diagonal(a.todense(), offset=2, axis1=0, axis2=2),
    )


def test_diagonalize():
    assert_eq(sparse.diagonalize(np.ones(3)), sparse.eye(3))

    assert_eq(
        sparse.diagonalize(scipy.sparse.coo_matrix(np.eye(3))),
        sparse.diagonalize(sparse.eye(3)),
    )

    # inverse of diagonal
    b = sparse.random((4, 3, 2), density=0.5)
    b_diag = sparse.diagonalize(b, axis=1)

    assert_eq(b, sparse.diagonal(b_diag, axis1=1, axis2=3).transpose([0, 2, 1]))


RESULT_TYPE_DTYPES = [
    "i1",
    "i2",
    "i4",
    "i8",
    "u1",
    "u2",
    "u4",
    "u8",
    "f4",
    "f8",
    "c8",
    "c16",
    object,
]


@pytest.mark.parametrize("t1", RESULT_TYPE_DTYPES)
@pytest.mark.parametrize("t2", RESULT_TYPE_DTYPES)
@pytest.mark.parametrize(
    "func",
    [
        sparse.result_type,
        pytest.param(
            np.result_type,
            marks=pytest.mark.skipif(not NEP18_ENABLED, reason="NEP18 is not enabled"),
        ),
    ],
)
@pytest.mark.parametrize("data", [1, [1]])  # Not the same outputs!
def test_result_type(t1, t2, func, data):
    a = np.array(data, dtype=t1)
    b = np.array(data, dtype=t2)
    expect = np.result_type(a, b)
    assert func(a, sparse.COO(b)) == expect
    assert func(sparse.COO(a), b) == expect
    assert func(sparse.COO(a), sparse.COO(b)) == expect
    assert func(a.dtype, sparse.COO(b)) == np.result_type(a.dtype, b)
    assert func(sparse.COO(a), b.dtype) == np.result_type(a, b.dtype)


@pytest.mark.parametrize("in_shape", [(5, 5), 62, (3, 3, 3)])
def test_flatten(in_shape):
    s = sparse.random(in_shape, density=0.5)
    x = s.todense()

    a = s.flatten()
    e = x.flatten()

    assert_eq(e, a)


def test_asnumpy():
    s = sparse.COO(data=[1], coords=[2], shape=(5,))
    assert_eq(sparse.asnumpy(s), s.todense())
    assert_eq(sparse.asnumpy(s, dtype=np.float64), np.asarray(s.todense(), dtype=np.float64))
    a = np.array([1, 2, 3])
    # Array passes through with no copying.
    assert sparse.asnumpy(a) is a


@pytest.mark.parametrize("shape1", [(2,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("shape2", [(2,), (2, 3), (2, 3, 4)])
def test_outer(shape1, shape2):
    s1 = sparse.random(shape1, density=0.5)
    s2 = sparse.random(shape2, density=0.5)

    x1 = s1.todense()
    x2 = s2.todense()

    assert_eq(sparse.outer(s1, s2), np.outer(x1, x2))
    assert_eq(np.multiply.outer(s1, s2), np.multiply.outer(x1, x2))


def test_scalar_list_init():
    a = sparse.COO([], [], ())
    b = sparse.COO([], [1], ())

    assert a.todense() == 0
    assert b.todense() == 1


def test_raise_on_nd_data():
    s1 = sparse.random((2, 3, 4), density=0.5)
    with pytest.raises(ValueError):
        sparse.COO(s1.coords, s1.data[:, None], shape=(2, 3, 4))


def test_astype_casting():
    s1 = sparse.random((2, 3, 4), density=0.5)
    with pytest.raises(TypeError):
        s1.astype(dtype=np.int64, casting="safe")


def test_astype_no_copy():
    s1 = sparse.random((2, 3, 4), density=0.5)
    s2 = s1.astype(s1.dtype, copy=False)
    assert s1 is s2


def test_coo_valerr():
    a = np.arange(300)
    with pytest.raises(ValueError):
        COO.from_numpy(a, idx_dtype=np.int8)


def test_random_idx_dtype():
    with pytest.raises(ValueError):
        sparse.random((300,), density=0.1, format="coo", idx_dtype=np.int8)


def test_html_for_size_zero():
    arr = sparse.COO.from_numpy(np.array(()))
    ground_truth = "<table><tbody>"
    ground_truth += '<tr><th style="text-align: left">Format</th><td style="text-align: left">coo</td></tr>'
    ground_truth += '<tr><th style="text-align: left">Data Type</th><td style="text-align: left">float64</td></tr>'
    ground_truth += '<tr><th style="text-align: left">Shape</th><td style="text-align: left">(0,)</td></tr>'
    ground_truth += '<tr><th style="text-align: left">nnz</th><td style="text-align: left">0</td></tr>'
    ground_truth += '<tr><th style="text-align: left">Density</th><td style="text-align: left">nan</td></tr>'
    ground_truth += '<tr><th style="text-align: left">Read-only</th><td style="text-align: left">True</td></tr>'
    ground_truth += '<tr><th style="text-align: left">Size</th><td style="text-align: left">0</td></tr>'
    ground_truth += '<tr><th style="text-align: left">Storage ratio</th><td style="text-align: left">nan</td></tr>'
    ground_truth += "</tbody></table>"

    table = html_table(arr)
    assert table == ground_truth


@pytest.mark.parametrize(
    "pad_width",
    [
        2,
        (2, 1),
        ((2), (1)),
        ((1, 2), (4, 5), (7, 8)),
    ],
)
@pytest.mark.parametrize("constant_values", [0, 1, 150, np.nan])
def test_pad_valid(pad_width, constant_values):
    y = sparse.random((50, 50, 3), density=0.15, fill_value=constant_values)
    x = y.todense()
    xx = np.pad(x, pad_width=pad_width, constant_values=constant_values)
    yy = np.pad(y, pad_width=pad_width, constant_values=constant_values)
    assert_eq(xx, yy)


@pytest.mark.parametrize(
    "pad_width",
    [
        ((2, 1), (5, 7)),
    ],
)
@pytest.mark.parametrize("constant_values", [150, 2, (1, 2)])
def test_pad_invalid(pad_width, constant_values, fill_value=0):
    y = sparse.random((50, 50, 3), density=0.15)
    with pytest.raises(ValueError):
        np.pad(y, pad_width, constant_values=constant_values)


@pytest.mark.parametrize("val", [0, 5])
def test_scalar_from_numpy(val):
    x = np.int64(val)
    s = sparse.COO.from_numpy(x)
    assert s.nnz == 0
    assert_eq(x, s)


def test_scalar_elemwise(rng):
    s1 = sparse.random((), density=0.5)
    x2 = rng.random(2)

    x1 = s1.todense()

    assert_eq(s1 * x2, x1 * x2)


def test_array_as_shape():
    coords = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    data = [10, 20, 30, 40, 50]

    sparse.COO(coords, data, shape=np.array((5, 5)))


@pytest.mark.parametrize(
    "arr",
    [np.array([[0, 3, 0], [1, 2, 0]]), np.array([[[0, 0], [1, 0]], [[5, 0], [0, -3]]])],
)
@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("mode", [(sparse.argmax, np.argmax), (sparse.argmin, np.argmin)])
def test_argmax_argmin(arr, axis, keepdims, mode):
    sparse_func, np_func = mode

    s_arr = sparse.COO.from_numpy(arr)

    result = sparse_func(s_arr, axis=axis, keepdims=keepdims).todense()
    expected = np_func(arr, axis=axis, keepdims=keepdims)

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1, 2])
@pytest.mark.parametrize("mode", [(sparse.argmax, np.argmax), (sparse.argmin, np.argmin)])
def test_argmax_argmin_3D(axis, mode):
    sparse_func, np_func = mode

    s_arr = sparse.zeros(shape=(1000, 550, 3), format="dok")
    s_arr[100, 100, 0] = 3
    s_arr[100, 100, 1] = 3
    s_arr[100, 99, 0] = -2
    s_arr = s_arr.to_coo()

    result = sparse_func(s_arr, axis=axis).todense()
    expected = np_func(s_arr.todense(), axis=axis)

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize("func", [sparse.argmax, sparse.argmin])
def test_argmax_argmin_constraint(func):
    s = sparse.COO.from_numpy(np.full((2, 2), 2), fill_value=2)

    with pytest.raises(ValueError, match="`axis=2` is out of bounds for array of dimension 2."):
        func(s, axis=2)


@pytest.mark.parametrize("config", [(np.inf, "isinf"), (np.nan, "isnan")])
def test_isinf_isnan(config):
    obj, func_name = config

    arr = np.array([[1, 1, obj], [-obj, 1, 1]])
    s = sparse.COO.from_numpy(arr)

    result = getattr(s, func_name)().todense()
    expected = getattr(np, func_name)(arr)

    np.testing.assert_equal(result, expected)


class TestSqueeze:
    eye_arr = np.eye(2).reshape(1, 2, 1, 2)

    @pytest.mark.parametrize(
        "arr_and_axis",
        [
            (eye_arr, None),
            (eye_arr, 0),
            (eye_arr, 2),
            (eye_arr, (0, 2)),
            (np.zeros((5,)), None),
        ],
    )
    def test_squeeze(self, arr_and_axis):
        arr, axis = arr_and_axis

        s_arr = sparse.COO.from_numpy(arr)

        result_1 = sparse.squeeze(s_arr, axis=axis).todense()
        result_2 = s_arr.squeeze(axis=axis).todense()
        expected = np.squeeze(arr, axis=axis)

        np.testing.assert_equal(result_1, result_2)
        np.testing.assert_equal(result_1, expected)

    def test_squeeze_validation(self):
        s_arr = sparse.COO.from_numpy(np.eye(3))

        with pytest.raises(IndexError, match="tuple index out of range"):
            s_arr.squeeze(3)

        with pytest.raises(ValueError, match="Invalid axis parameter: `1.1`."):
            s_arr.squeeze(1.1)

        with pytest.raises(ValueError, match="Specified axis `0` has a size greater than one: 3"):
            s_arr.squeeze(0)


class TestUnique:
    arr = np.array([[0, 0, 1, 5, 3, 0], [1, 0, 4, 0, 3, 0], [0, 1, 0, 1, 1, 0]], dtype=np.int64)
    arr_empty = np.zeros((5, 5))
    arr_full = np.arange(1, 10)

    @pytest.mark.parametrize("arr", [arr, arr_empty, arr_full])
    @pytest.mark.parametrize("fill_value", [-1, 0, 1])
    def test_unique_counts(self, arr, fill_value):
        s_arr = sparse.COO.from_numpy(arr, fill_value)

        result_values, result_counts = sparse.unique_counts(s_arr)
        expected_values, expected_counts = np.unique(arr, return_counts=True)

        np.testing.assert_equal(result_values, expected_values)
        np.testing.assert_equal(result_counts, expected_counts)

    @pytest.mark.parametrize("arr", [arr, arr_empty, arr_full])
    @pytest.mark.parametrize("fill_value", [-1, 0, 1])
    def test_unique_values(self, arr, fill_value):
        s_arr = sparse.COO.from_numpy(arr, fill_value)

        result = sparse.unique_values(s_arr)
        expected = np.unique(arr)

        np.testing.assert_equal(result, expected)

    @pytest.mark.parametrize("func", [sparse.unique_counts, sparse.unique_values])
    def test_input_validation(self, func):
        with pytest.raises(ValueError, match="Input must be an instance of SparseArray"):
            func(self.arr)


@pytest.mark.parametrize("axis", [-1, 0, 1, 2, 3])
def test_expand_dims(axis):
    arr = np.arange(24).reshape((2, 3, 4))
    s_arr = sparse.COO.from_numpy(arr)

    result = sparse.expand_dims(s_arr, axis=axis)
    expected = np.expand_dims(arr, axis=axis)

    np.testing.assert_equal(result.todense(), expected)


@pytest.mark.parametrize(
    "arr",
    [
        np.array([[0, 0, 1, 5, 3, 0], [1, 0, 4, 0, 3, 0], [0, 1, 0, 1, 1, 0]], dtype=np.int64),
        np.array([[[2, 0], [0, 5]], [[1, 0], [4, 0]], [[0, 1], [0, -1]]], dtype=np.float64),
        np.arange(3, 10),
    ],
)
@pytest.mark.parametrize("fill_value", [-1, 0, 1, 3])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize(
    "stable", [False, pytest.param(True, marks=pytest.mark.xfail(reason="Numba doesn't support `stable=True`."))]
)
def test_sort(arr, fill_value, axis, descending, stable):
    if axis >= arr.ndim:
        return

    s_arr = sparse.COO.from_numpy(arr, fill_value)

    kind = "mergesort" if stable else "quicksort"

    result = sparse.sort(s_arr, axis=axis, descending=descending, stable=stable)
    expected = -np.sort(-arr, axis=axis, kind=kind) if descending else np.sort(arr, axis=axis, kind=kind)

    np.testing.assert_equal(result.todense(), expected)
    # make sure no inplace changes happened
    np.testing.assert_equal(s_arr.todense(), arr)


@pytest.mark.parametrize("fill_value", [-1, 0, 1])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_only_fill_value(fill_value, descending):
    arr = np.full((3, 3), fill_value=fill_value)
    s_arr = sparse.COO.from_numpy(arr, fill_value)

    result = sparse.sort(s_arr, axis=0, descending=descending)
    expected = np.sort(arr, axis=0)

    np.testing.assert_equal(result.todense(), expected)


@pytest.mark.parametrize("axis", [None, -1, 0, 1, 2, (0, 1), (2, 0)])
def test_flip(axis):
    arr = np.arange(24).reshape((2, 3, 4))
    s_arr = sparse.COO.from_numpy(arr)

    result = sparse.flip(s_arr, axis=axis)
    expected = np.flip(arr, axis=axis)

    np.testing.assert_equal(result.todense(), expected)


@pytest.mark.parametrize("fill_value", [-1, 0, 1, 3])
@pytest.mark.parametrize(
    "indices,axis",
    [
        (
            [1],
            0,
        ),
        ([2, 1], 1),
        ([1, 2, 3], 2),
        ([2, 3], -1),
        ([5, 3, 7, 8], None),
    ],
)
def test_take(fill_value, indices, axis):
    arr = np.arange(24).reshape((2, 3, 4))

    s_arr = sparse.COO.from_numpy(arr, fill_value)

    result = sparse.take(s_arr, np.array(indices), axis=axis)
    expected = np.take(arr, indices, axis)

    np.testing.assert_equal(result.todense(), expected)


@pytest.mark.parametrize("ndim", [2, 3, 4, 5])
@pytest.mark.parametrize("density", [0.0, 0.1, 0.25, 1.0])
def test_matrix_transpose(ndim, density):
    shape = tuple(range(2, 34)[:ndim])
    xs = sparse.random(shape, density=density)
    xd = xs.todense()

    transpose_axes = list(range(ndim))
    transpose_axes[-2:] = transpose_axes[-2:][::-1]

    expected = np.transpose(xd, axes=transpose_axes)
    actual = sparse.matrix_transpose(xs)

    assert_eq(actual, expected)
    assert_eq(xs.mT, expected)


@pytest.mark.parametrize(
    ("shape1", "shape2", "axis"),
    [
        ((2, 3, 4), (3, 4), -2),
        ((3, 4), (2, 3, 4), -1),
        ((3, 1, 4), (3, 2, 4), 2),
        ((1, 3, 4), (3, 4), -2),
        ((3, 4, 1), (3, 4, 2), 0),
        ((3, 1), (3, 4), -2),
        ((1, 4), (3, 4), 1),
    ],
)
@pytest.mark.parametrize("density", [0.0, 0.1, 0.25, 1.0])
@pytest.mark.parametrize("is_complex", [False, True])
def test_vecdot(shape1, shape2, axis, density, rng, is_complex):
    def data_rvs(size):
        data = rng.random(size)
        if is_complex:
            data = data + rng.random(size) * 1j
        return data

    s1 = sparse.random(shape1, density=density, data_rvs=data_rvs)
    s2 = sparse.random(shape2, density=density, data_rvs=data_rvs)

    x1 = s1.todense()
    x2 = s2.todense()

    def np_vecdot(x1, x2, /, *, axis=-1):
        if np.issubdtype(x1.dtype, np.complexfloating):
            x1 = np.conjugate(x1)

        return np.sum(x1 * x2, axis=axis)

    actual = sparse.vecdot(s1, s2, axis=axis)
    assert s1.dtype == s2.dtype == actual.dtype
    expected = np_vecdot(x1, x2, axis=axis)
    np.testing.assert_allclose(actual.todense(), expected)


@pytest.mark.parametrize(
    ("shape1", "shape2", "axis"),
    [
        ((2, 3, 4), (3, 4), 0),
        ((3, 4), (2, 3, 4), 0),
        ((3, 1, 4), (3, 2, 4), -2),
        ((1, 3, 4), (3, 4), -3),
        ((3, 4, 1), (3, 4, 2), -1),
        ((3, 1), (3, 4), 1),
        ((1, 4), (3, 4), -2),
    ],
)
def test_vecdot_invalid_axis(shape1, shape2, axis):
    s1 = sparse.random(shape1, density=0.5)
    s2 = sparse.random(shape2, density=0.5)

    with pytest.raises(ValueError, match=r"Shapes must match along"):
        sparse.vecdot(s1, s2, axis=axis)


@pytest.mark.parametrize(
    ("func", "args", "kwargs"),
    [
        (sparse.eye, (5,), {}),
        (sparse.zeros, ((5,)), {}),
        (sparse.ones, ((5,)), {}),
        (sparse.full, ((5,), 5), {}),
        (sparse.empty, ((5,)), {}),
        (sparse.full_like, (5,), {}),
        (sparse.ones_like, (), {}),
        (sparse.zeros_like, (), {}),
        (sparse.empty_like, (), {}),
        (sparse.asarray, (), {}),
    ],
)
def test_invalid_device(func, args, kwargs):
    if func.__name__.endswith("_like") or func is sparse.asarray:
        like = sparse.random((5, 5), density=0.5)
        args = (like,) + args

    with pytest.raises(ValueError, match="Device must be"):
        func(*args, device="invalid_device", **kwargs)


def test_device():
    s = sparse.random((5, 5), density=0.5)
    data = getattr(s, "data", None)
    device = getattr(data, "device", "cpu")

    assert s.device == device


def test_to_device():
    s = sparse.random((5, 5), density=0.5)
    s2 = s.to_device(s.device)

    assert s is s2


def test_to_invalid_device():
    s = sparse.random((5, 5), density=0.5)
    with pytest.raises(ValueError, match=r"Only .* is supported."):
        s.to_device("invalid_device")
