import importlib
import operator
import os

import sparse

import pytest

import scipy.sparse as sps

DENSITY = 0.001


def get_test_id(side):
    return f"{side=}"


@pytest.fixture(params=[100, 500, 1000], ids=get_test_id)
def elemwise_args(request, rng, max_size):
    side = request.param
    if side**2 >= max_size:
        pytest.skip()
    s1_sps = sps.random(side, side, format="csr", density=DENSITY, random_state=rng) * 10
    s1_sps.sum_duplicates()
    s2_sps = sps.random(side, side, format="csr", density=DENSITY, random_state=rng) * 10
    s2_sps.sum_duplicates()
    return s1_sps, s2_sps


@pytest.fixture(params=[operator.add, operator.mul, operator.gt])
def elemwise_function(request):
    return request.param


@pytest.fixture(params=["SciPy", "Numba", "Finch"])
def backend_name(request):
    return request.param


@pytest.fixture
def backend_setup(backend_name):
    os.environ[sparse._ENV_VAR_NAME] = backend_name
    importlib.reload(sparse)
    yield sparse, backend_name
    del os.environ[sparse._ENV_VAR_NAME]
    importlib.reload(sparse)


@pytest.fixture
def sparse_arrays(elemwise_args, backend_setup):
    s1_sps, s2_sps = elemwise_args
    sparse, backend_name = backend_setup

    if backend_name == "SciPy":
        s1 = s1_sps
        s2 = s2_sps
    elif backend_name == "Numba":
        s1 = sparse.asarray(s1_sps)
        s2 = sparse.asarray(s2_sps)
    elif backend_name == "Finch":
        s1 = sparse.asarray(s1_sps.asformat("csc"), format="csc")
        s2 = sparse.asarray(s2_sps.asformat("csc"), format="csc")

    return s1, s2


def test_elemwise(benchmark, elemwise_function, sparse_arrays):
    s1, s2 = sparse_arrays

    elemwise_function(s1, s2)

    @benchmark
    def bench():
        elemwise_function(s1, s2)
