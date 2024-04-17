import operator

import sparse
from sparse import COO
from sparse.numba_backend._compressed import GCXS
from sparse.numba_backend._utils import assert_eq, assert_gcxs_slicing, default_rng

import pytest

import numpy as np
import scipy.sparse
import scipy.stats


@pytest.mark.parametrize(
    "a_shape,b_shape,axes",
    [
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
    ],
)
@pytest.mark.parametrize(
    "a_format, b_format",
    [("coo", "coo"), ("coo", "gcxs"), ("gcxs", "coo"), ("gcxs", "gcxs")],
)
def test_tensordot(a_shape, b_shape, axes, a_format, b_format):
    sa = sparse.random(a_shape, density=0.5, format=a_format)
    sb = sparse.random(b_shape, density=0.5, format=b_format)

    a = sa.todense()
    b = sb.todense()

    a_b = np.tensordot(a, b, axes)

    # tests for return_type=None
    sa_sb = sparse.tensordot(sa, sb, axes)
    sa_b = sparse.tensordot(sa, b, axes)
    a_sb = sparse.tensordot(a, sb, axes)

    assert_eq(a_b, sa_sb)
    assert_eq(a_b, sa_b)
    assert_eq(a_b, a_sb)
    if all(isinstance(arr, COO) for arr in [sa, sb]):
        assert isinstance(sa_sb, COO)
    else:
        assert isinstance(sa_sb, GCXS)
    assert isinstance(sa_b, np.ndarray)
    assert isinstance(a_sb, np.ndarray)

    # tests for return_type=COO
    sa_b = sparse.tensordot(sa, b, axes, return_type=COO)
    a_sb = sparse.tensordot(a, sb, axes, return_type=COO)

    assert_eq(a_b, sa_b)
    assert_eq(a_b, a_sb)
    assert isinstance(sa_b, COO)
    assert isinstance(a_sb, COO)

    # tests form return_type=GCXS
    sa_b = sparse.tensordot(sa, b, axes, return_type=GCXS)
    a_sb = sparse.tensordot(a, sb, axes, return_type=GCXS)

    assert_eq(a_b, sa_b)
    assert_eq(a_b, a_sb)
    assert isinstance(sa_b, GCXS)
    assert isinstance(a_sb, GCXS)

    # tests for return_type=np.ndarray
    sa_sb = sparse.tensordot(sa, sb, axes, return_type=np.ndarray)

    assert_eq(a_b, sa_sb)
    assert isinstance(sa_sb, np.ndarray)


def test_tensordot_empty():
    x1 = np.empty((0, 0, 0))
    x2 = np.empty((0, 0, 0))
    s1 = sparse.COO.from_numpy(x1)
    s2 = sparse.COO.from_numpy(x2)

    assert_eq(np.tensordot(x1, x2), sparse.tensordot(s1, s2))


def test_tensordot_valueerror():
    x1 = sparse.COO(np.array(1))
    x2 = sparse.COO(np.array(1))

    with pytest.raises(ValueError):
        x1 @ x2


def gen_kwargs(format):
    from sparse.numba_backend._utils import convert_format

    format = convert_format(format)
    if format == "gcxs":
        return [{"compressed_axes": c} for c in [(0,), (1,)]]

    return [{}]


def gen_for_format(format):
    return [(format, g) for g in gen_kwargs(format)]


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ((3, 1, 6, 5), (2, 1, 4, 5, 6)),
        ((2, 1, 4, 5, 6), (3, 1, 6, 5)),
        ((1, 1, 5), (3, 5, 6)),
        ((3, 4, 5), (1, 5, 6)),
        ((3, 4, 5), (3, 5, 6)),
        ((3, 4, 5), (5, 6)),
        ((4, 5), (5, 6)),
        ((5,), (5, 6)),
        ((4, 5), (5,)),
        ((5,), (5,)),
        ((3, 4), (1, 2, 4, 3)),
    ],
)
@pytest.mark.parametrize(
    "a_format, a_kwargs",
    [*gen_for_format("coo"), *gen_for_format("gcxs")],
)
@pytest.mark.parametrize(
    "b_format, b_kwargs",
    [*gen_for_format("coo"), *gen_for_format("gcxs")],
)
def test_matmul(a_shape, b_shape, a_format, b_format, a_kwargs, b_kwargs):
    if len(a_shape) == 1:
        a_kwargs = {}
    if len(b_shape) == 1:
        b_kwargs = {}
    sa = sparse.random(a_shape, density=0.5, format=a_format, **a_kwargs)
    sb = sparse.random(b_shape, density=0.5, format=b_format, **b_kwargs)

    a = sa.todense()
    b = sb.todense()

    assert_eq(np.matmul(a, b), sparse.matmul(sa, sb))
    assert_eq(sparse.matmul(sa, b), sparse.matmul(a, sb))
    assert_eq(np.matmul(a, b), sparse.matmul(sa, sb))

    if a.ndim == 2 or b.ndim == 2:
        assert_eq(
            np.matmul(a, b),
            sparse.matmul(
                scipy.sparse.coo_matrix(a) if a.ndim == 2 else sa,
                scipy.sparse.coo_matrix(b) if b.ndim == 2 else sb,
            ),
        )

    if hasattr(operator, "matmul"):
        assert_eq(operator.matmul(a, b), operator.matmul(sa, sb))


def test_matmul_errors():
    with pytest.raises(ValueError):
        sa = sparse.random((3, 4, 5, 6), 0.5)
        sb = sparse.random((3, 6, 5, 6), 0.5)
        sparse.matmul(sa, sb)


@pytest.mark.parametrize(
    "a, b",
    [
        (
            sparse.GCXS.from_numpy(default_rng.choice([0, np.nan, 2], size=[100, 100], p=[0.99, 0.001, 0.009])),
            sparse.random((100, 100), density=0.01),
        ),
        (
            sparse.COO.from_numpy(default_rng.choice([0, np.nan, 2], size=[100, 100], p=[0.99, 0.001, 0.009])),
            sparse.random((100, 100), density=0.01),
        ),
        (
            sparse.GCXS.from_numpy(default_rng.choice([0, np.nan, 2], size=[100, 100], p=[0.99, 0.001, 0.009])),
            scipy.sparse.random(100, 100),
        ),
        (
            default_rng.choice([0, np.nan, 2], size=[100, 100], p=[0.99, 0.001, 0.009]),
            sparse.random((100, 100), density=0.01),
        ),
    ],
)
def test_matmul_nan_warnings(a, b):
    with pytest.warns(RuntimeWarning):
        a @ b


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ((1, 4, 5), (3, 5, 6)),
        ((3, 4, 5), (1, 5, 6)),
        ((3, 4, 5), (3, 5, 6)),
        ((3, 4, 5), (5, 6)),
        ((4, 5), (5, 6)),
        ((5,), (5, 6)),
        ((4, 5), (5,)),
        ((5,), (5,)),
    ],
)
@pytest.mark.parametrize(
    "a_format, a_kwargs",
    [*gen_for_format("coo"), *gen_for_format("gcxs")],
)
@pytest.mark.parametrize(
    "b_format, b_kwargs",
    [*gen_for_format("coo"), *gen_for_format("gcxs")],
)
def test_dot(a_shape, b_shape, a_format, b_format, a_kwargs, b_kwargs):
    if len(a_shape) == 1:
        a_kwargs = {}
    if len(b_shape) == 1:
        b_kwargs = {}
    sa = sparse.random(a_shape, density=0.5, format=a_format, **a_kwargs)
    sb = sparse.random(b_shape, density=0.5, format=b_format, **b_kwargs)

    a = sa.todense()
    b = sb.todense()

    e = np.dot(a, b)
    assert_eq(e, sa.dot(sb))
    assert_eq(e, sparse.dot(sa, sb))
    assert_eq(e, sparse.dot(a, sb))
    assert_eq(e, sparse.dot(a, sb))

    # Basic equivalences
    e = operator.matmul(a, b)
    assert_eq(e, operator.matmul(sa, sb))
    assert_eq(e, operator.matmul(a, sb))
    assert_eq(e, operator.matmul(sa, b))


@pytest.mark.parametrize(
    "a_dense, b_dense, o_type",
    [
        (False, False, sparse.SparseArray),
        (False, True, np.ndarray),
        (True, False, np.ndarray),
    ],
)
def test_dot_type(a_dense, b_dense, o_type):
    a = sparse.random((3, 4), density=0.8)
    b = sparse.random((4, 5), density=0.8)

    if a_dense:
        a = a.todense()

    if b_dense:
        b = b.todense()

    assert isinstance(sparse.dot(a, b), o_type)


@pytest.mark.xfail
def test_dot_nocoercion():
    sa = sparse.random((3, 4, 5), density=0.5)
    sb = sparse.random((5, 6), density=0.5)

    a = sa.todense()
    b = sb.todense()

    la = a.tolist()
    lb = b.tolist()

    if hasattr(operator, "matmul"):
        # Operations with naive collection (list)
        assert_eq(operator.matmul(la, b), operator.matmul(la, sb))
        assert_eq(operator.matmul(a, lb), operator.matmul(sa, lb))


dot_formats = [
    lambda x: x.asformat("coo"),
    lambda x: x.asformat("gcxs"),
    lambda x: x.todense(),
]


@pytest.mark.parametrize("format1", dot_formats)
@pytest.mark.parametrize("format2", dot_formats)
def test_small_values(format1, format2):
    s1 = format1(sparse.COO(coords=[[0, 10]], data=[3.6e-100, 7.2e-009], shape=(20,)))
    s2 = format2(sparse.COO(coords=[[0, 0], [4, 28]], data=[3.8e-25, 4.5e-225], shape=(20, 50)))

    def dense_convertor(x):
        return x.todense() if isinstance(x, sparse.SparseArray) else x

    x1, x2 = dense_convertor(s1), dense_convertor(s2)

    assert_eq(x1 @ x2, s1 @ s2)


dot_dtypes = [np.complex64, np.complex128]


@pytest.mark.parametrize("dtype1", dot_dtypes)
@pytest.mark.parametrize("dtype2", dot_dtypes)
@pytest.mark.parametrize("format1", dot_formats)
@pytest.mark.parametrize("format2", dot_formats)
@pytest.mark.parametrize("ndim1", (1, 2))
@pytest.mark.parametrize("ndim2", (1, 2))
def test_complex(dtype1, dtype2, format1, format2, ndim1, ndim2):
    s1 = format1(sparse.random((20,) * ndim1, density=0.5).astype(dtype1))
    s2 = format2(sparse.random((20,) * ndim2, density=0.5).astype(dtype2))

    def dense_convertor(x):
        return x.todense() if isinstance(x, sparse.SparseArray) else x

    x1, x2 = dense_convertor(s1), dense_convertor(s2)

    assert_eq(x1 @ x2, s1 @ s2)


@pytest.mark.parametrize("dtype1", dot_dtypes)
@pytest.mark.parametrize("dtype2", dot_dtypes)
@pytest.mark.parametrize("ndim1", (1, 2))
@pytest.mark.parametrize("ndim2", (1, 2))
def test_dot_dense(dtype1, dtype2, ndim1, ndim2):
    a = sparse.random((20,) * ndim1, density=0.5).astype(dtype1).todense()
    b = sparse.random((20,) * ndim2, density=0.5).astype(dtype2).todense()

    assert_eq(sparse.dot(a, b), np.dot(a, b))
    assert_eq(sparse.matmul(a, b), np.matmul(a, b))
    if ndim1 == 2 and ndim2 == 2:
        assert_eq(sparse.tensordot(a, b), np.tensordot(a, b))


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [((3, 4, 5), (5, 6)), ((2, 8, 6), (6, 3))],
)
def test_dot_GCXS_slicing(a_shape, b_shape):
    sa = sparse.random(shape=a_shape, density=1, format="gcxs")
    sb = sparse.random(shape=b_shape, density=1, format="gcxs")

    a = sa.todense()
    b = sb.todense()

    # tests dot
    sa_sb = sparse.dot(sa, sb)
    a_b = np.dot(a, b)

    assert_gcxs_slicing(sa_sb, a_b)


@pytest.mark.parametrize(
    "a_shape,b_shape,axes",
    [
        [(3, 4, 5), (4, 3), (1, 0)],
        [(3, 4), (5, 4, 3), (1, 1)],
        [(5, 9), (9, 5, 6), (0, 1)],
    ],
)
def test_tensordot_GCXS_slicing(a_shape, b_shape, axes):
    sa = sparse.random(shape=a_shape, density=1, format="gcxs")
    sb = sparse.random(shape=b_shape, density=1, format="gcxs")

    a = sa.todense()
    b = sb.todense()

    sa_sb = sparse.tensordot(sa, sb, axes)
    a_b = np.tensordot(a, b, axes)

    assert_gcxs_slicing(sa_sb, a_b)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        [(1, 1, 5), (3, 5, 6)],
        [(3, 4, 5), (1, 5, 6)],
        [(3, 4, 5), (3, 5, 6)],
        [(3, 4, 5), (5, 6)],
    ],
)
def test_matmul_GCXS_slicing(a_shape, b_shape):
    sa = sparse.random(shape=a_shape, density=1, format="gcxs")
    sb = sparse.random(shape=b_shape, density=1, format="gcxs")

    a = sa.todense()
    b = sb.todense()

    sa_sb = sparse.matmul(sa, sb)
    a_b = np.matmul(a, b)

    assert_gcxs_slicing(sa_sb, a_b)
