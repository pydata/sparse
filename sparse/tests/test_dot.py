import numpy as np
import pytest
import scipy.sparse
import scipy.stats

import operator
import sparse
from sparse._compressed import GCXS
from sparse import COO
from sparse._utils import assert_eq


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
    "a_format, b_format",
    [("coo", "coo"), ("coo", "gcxs"), ("gcxs", "coo"), ("gcxs", "gcxs")],
)
@pytest.mark.parametrize(
    "a_comp_axes, b_comp_axes", [([0], [0]), ([0], [1]), ([1], [0]), ([1], [1])]
)
def test_matmul(a_shape, b_shape, a_format, b_format, a_comp_axes, b_comp_axes):
    if a_format == "coo" or len(a_shape) == 1:
        a_comp_axes = None
    if b_format == "coo" or len(b_shape) == 1:
        b_comp_axes = None
    sa = sparse.random(
        a_shape, density=0.5, format=a_format, compressed_axes=a_comp_axes
    )
    sb = sparse.random(
        b_shape, density=0.5, format=b_format, compressed_axes=b_comp_axes
    )

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
    "a_format, b_format",
    [("coo", "coo"), ("coo", "gcxs"), ("gcxs", "coo"), ("gcxs", "gcxs")],
)
@pytest.mark.parametrize(
    "a_comp_axes, b_comp_axes", [([0], [0]), ([0], [1]), ([1], [0]), ([1], [1])]
)
def test_dot(a_shape, b_shape, a_format, b_format, a_comp_axes, b_comp_axes):
    if a_format == "coo" or len(a_shape) == 1:
        a_comp_axes = None
    if b_format == "coo" or len(b_shape) == 1:
        b_comp_axes = None
    sa = sparse.random(
        a_shape, density=0.5, format=a_format, compressed_axes=a_comp_axes
    )
    sb = sparse.random(
        b_shape, density=0.5, format=b_format, compressed_axes=b_comp_axes
    )

    a = sa.todense()
    b = sb.todense()

    assert_eq(a.dot(b), sa.dot(sb))
    assert_eq(np.dot(a, b), sparse.dot(sa, sb))
    assert_eq(sparse.dot(sa, b), sparse.dot(a, sb))
    assert_eq(np.dot(a, b), sparse.dot(sa, sb))

    # Basic equivalences
    assert_eq(operator.matmul(a, b), operator.matmul(sa, sb))
    # Test that COO's and np.array's combine correctly
    # Not possible due to https://github.com/numpy/numpy/issues/9028
    # assert_eq(eval("a @ sb"), eval("sa @ b"))


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
    s2 = format2(
        sparse.COO(coords=[[0, 0], [4, 28]], data=[3.8e-25, 4.5e-225], shape=(20, 50))
    )

    dense_convertor = lambda x: x.todense() if isinstance(x, sparse.SparseArray) else x
    x1, x2 = dense_convertor(s1), dense_convertor(s2)

    assert_eq(x1 @ x2, s1 @ s2)


dot_dtypes = [np.complex64, np.complex128]


@pytest.mark.parametrize("dtype1", dot_dtypes)
@pytest.mark.parametrize("dtype2", dot_dtypes)
@pytest.mark.parametrize("format1", dot_formats)
@pytest.mark.parametrize("format2", dot_formats)
def test_complex(dtype1, dtype2, format1, format2):
    s1 = format1(sparse.random((20,), density=0.5).astype(dtype1))
    s2 = format2(sparse.random((20,), density=0.5).astype(dtype2))

    dense_convertor = lambda x: x.todense() if isinstance(x, sparse.SparseArray) else x
    x1, x2 = dense_convertor(s1), dense_convertor(s2)

    assert_eq(x1 @ x2, s1 @ s2)
