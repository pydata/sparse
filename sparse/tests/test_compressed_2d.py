import numpy as np
from numpy.core.numeric import indices
import pytest
from hypothesis import settings, given, strategies as st
import scipy.sparse
from scipy.sparse import data
from scipy.sparse.construct import random
import scipy.stats

import sparse
from sparse import COO
from sparse._compressed.compressed import GCXS, CSR, CSC
from sparse._utils import assert_eq
from _utils import gen_sparse_random, gen_sparse_random_from


@pytest.fixture(scope="module", params=[CSR, CSC])
def cls(request):
    return request.param


@pytest.fixture(scope="module", params=["f8", "f4", "i8", "i4"])
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def random_sparse(cls, dtype):
    if np.issubdtype(dtype, np.integer):

        def data_rvs(n):
            return np.random.randint(-1000, 1000, n)

    else:
        data_rvs = None
    return cls(sparse.random((20, 30), density=0.25, data_rvs=data_rvs).astype(dtype))


@pytest.fixture(scope="module")
def random_sparse_small(cls, dtype):
    if np.issubdtype(dtype, np.integer):

        def data_rvs(n):
            return np.random.randint(-10, 10, n)

    else:
        data_rvs = None
    return cls(
        sparse.random((20, 30, 40), density=0.25, data_rvs=data_rvs).astype(dtype)
    )


def test_repr(random_sparse):
    cls = type(random_sparse).__name__

    str_repr = repr(random_sparse)
    assert cls in str_repr


def test_bad_constructor_input(cls):
    with pytest.raises(ValueError, match=r".*shape.*"):
        cls(arg="hello world")


@given(n=st.sampled_from([0, 1, 3]))
def test_bad_nd_input(cls, n):
    a = np.ones(shape=tuple(5 for _ in range(n)))
    with pytest.raises(ValueError, match=f"{n}-d"):
        cls(a)


@settings(deadline=None)
@given(gcxs=gen_sparse_random_from((20, 30), density=0.25))
def test_from_sparse(cls, gcxs):
    result = cls(gcxs)

    assert_eq(result, gcxs)


@settings(deadline=None)
@given(
    scipy_type=st.sampled_from(["coo", "csr", "csc", "lil"]),
    CLS=st.sampled_from([CSR, CSC, GCXS]),
)
def test_from_scipy_sparse(scipy_type, CLS, dtype):
    orig = scipy.sparse.random(20, 30, density=0.2, format=scipy_type, dtype=dtype)
    ref = COO.from_scipy_sparse(orig)
    result = CLS.from_scipy_sparse(orig)

    assert_eq(ref, result)

    result_via_init = CLS(orig)

    assert_eq(ref, result_via_init)


@settings(deadline=None)
@given(
    cls_str=st.sampled_from(["coo", "dok", "csr", "csc", "gcxs"]),
    rs=gen_sparse_random((20, 30), density=0.25),
)
def test_to_sparse(cls_str, rs):
    result = rs.asformat(cls_str)

    assert_eq(rs, result)


@settings(deadline=None)
@given(
    copy=st.sampled_from([True, False]), rs=gen_sparse_random((20, 30), density=0.25)
)
def test_transpose(copy, rs):
    from operator import is_, is_not

    t = rs.transpose(copy=copy)
    tt = t.transpose(copy=copy)

    # Check if a copy was made
    if copy:
        check = is_not
    else:
        check = is_

    assert check(rs.data, t.data)
    assert check(rs.indices, t.indices)
    assert check(rs.indptr, t.indptr)

    assert rs.shape == t.shape[::-1]

    assert_eq(rs, tt)
    assert type(rs) == type(tt)


@given(rs=gen_sparse_random((20, 30), density=0.25))
def test_transpose_error(rs):
    with pytest.raises(ValueError):
        rs.transpose(axes=1)
