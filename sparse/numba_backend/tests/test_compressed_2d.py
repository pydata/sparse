import sparse
from sparse import COO
from sparse.numba_backend._compressed.compressed import CSC, CSR, GCXS
from sparse.numba_backend._utils import assert_eq

import pytest

import numpy as np
import scipy.sparse
import scipy.stats


@pytest.fixture(scope="module", params=[CSR, CSC])
def cls(request):
    return request.param


@pytest.fixture(scope="module", params=["f8", "f4", "i8", "i4"])
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def random_sparse(cls, dtype, rng):
    if np.issubdtype(dtype, np.integer):

        def data_rvs(n):
            return rng.integers(-1000, 1000, n)

    else:
        data_rvs = None
    return cls(sparse.random((20, 30), density=0.25, data_rvs=data_rvs).astype(dtype))


@pytest.fixture(scope="module")
def random_sparse_small(cls, dtype, rng):
    if np.issubdtype(dtype, np.integer):

        def data_rvs(n):
            return rng.integers(-10, 10, n)

    else:
        data_rvs = None
    return cls(sparse.random((20, 20), density=0.25, data_rvs=data_rvs).astype(dtype))


def test_repr(random_sparse):
    cls = type(random_sparse).__name__

    str_repr = repr(random_sparse)
    assert cls in str_repr


def test_bad_constructor_input(cls):
    with pytest.raises(ValueError, match=r".*shape.*"):
        cls(arg="hello world")


@pytest.mark.parametrize("n", [0, 1, 3])
def test_bad_nd_input(cls, n):
    a = np.ones(shape=tuple(5 for _ in range(n)))
    with pytest.raises(ValueError, match=f"{n}-d"):
        cls(a)


@pytest.mark.parametrize("source_type", ["gcxs", "coo"])
def test_from_sparse(cls, source_type):
    gcxs = sparse.random((20, 30), density=0.25, format=source_type)
    result = cls(gcxs)

    assert_eq(result, gcxs)


@pytest.mark.parametrize("scipy_type", ["coo", "csr", "csc", "lil"])
@pytest.mark.parametrize("CLS", [CSR, CSC, GCXS])
def test_from_scipy_sparse(scipy_type, CLS, dtype):
    orig = scipy.sparse.random(20, 30, density=0.2, format=scipy_type, dtype=dtype)
    ref = COO.from_scipy_sparse(orig)
    result = CLS.from_scipy_sparse(orig)

    assert_eq(ref, result)

    result_via_init = CLS(orig)

    assert_eq(ref, result_via_init)


@pytest.mark.parametrize("cls_str", ["coo", "dok", "csr", "csc", "gcxs"])
def test_to_sparse(cls_str, random_sparse):
    result = random_sparse.asformat(cls_str)

    assert_eq(random_sparse, result)


@pytest.mark.parametrize("copy", [True, False])
def test_transpose(random_sparse, copy):
    from operator import is_, is_not

    t = random_sparse.transpose(copy=copy)
    tt = t.transpose(copy=copy)

    # Check if a copy was made
    check = is_not if copy else is_

    assert check(random_sparse.data, t.data)
    assert check(random_sparse.indices, t.indices)
    assert check(random_sparse.indptr, t.indptr)

    assert random_sparse.shape == t.shape[::-1]

    assert_eq(random_sparse, tt)
    assert type(random_sparse) is type(tt)

    assert_eq(random_sparse.transpose(axes=(0, 1)), random_sparse)
    assert_eq(random_sparse.transpose(axes=(1, 0)), t)
    with pytest.raises(ValueError, match="Invalid transpose axes"):
        random_sparse.transpose(axes=0)


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_mT_fill_value(format):
    fv = 1.0
    arr = sparse.full((10, 20), fill_value=fv, format=format)
    assert_eq(arr.mT, sparse.full((20, 10), fill_value=fv))


def test_transpose_error(random_sparse):
    with pytest.raises(ValueError):
        random_sparse.transpose(axes=1)


def test_matmul(random_sparse_small):
    arr = random_sparse_small.todense()

    actual = random_sparse_small @ random_sparse_small
    expected = arr @ arr

    assert_eq(actual, expected)
