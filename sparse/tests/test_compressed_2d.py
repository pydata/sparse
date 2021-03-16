import numpy as np
import pytest
import scipy.sparse
from scipy.sparse.construct import random
import scipy.stats

import sparse
from sparse import COO
from sparse._compressed.compressed import GCXS, CSR, CSC
from sparse._utils import assert_eq


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


def test_foo(random_sparse):
    assert isinstance(random_sparse, (CSC, CSR))
