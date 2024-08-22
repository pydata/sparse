import importlib
import itertools
import operator
import os

import sparse

import pytest

import numpy as np
import scipy.sparse as sps

DENSITY = 0.001


def get_test_id(side):
    return f"{side=}"


@pytest.fixture(params=[100, 500, 1000], ids=get_test_id)
def elemwise_args(request, seed, max_size):
    side = request.param
    if side**2 >= max_size:
        pytest.skip()
    rng = np.random.default_rng(seed=seed)
    s1_sps = sps.random(side, side, format="csr", density=DENSITY, random_state=rng) * 10
    s1_sps.sum_duplicates()
    s2_sps = sps.random(side, side, format="csr", density=DENSITY, random_state=rng) * 10
    s2_sps.sum_duplicates()
    return s1_sps, s2_sps


def get_elemwise_id(param):
    f, backend = param
    return f"{f=}-{backend=}"


@pytest.fixture(
    params=itertools.product([operator.add, operator.mul, operator.gt], ["SciPy", "Numba", "Finch"]),
    scope="function",
    ids=get_elemwise_id,
)
def backend(request):
    f, backend = request.param
    os.environ[sparse._ENV_VAR_NAME] = backend
    importlib.reload(sparse)
    yield f, sparse, backend
    del os.environ[sparse._ENV_VAR_NAME]
    importlib.reload(sparse)


def test_elemwise(benchmark, backend, elemwise_args):
    s1_sps, s2_sps = elemwise_args
    f, sparse, backend = backend

    if backend == "SciPy":
        s1 = s1_sps
        s2 = s2_sps
    elif backend == "Numba":
        s1 = sparse.asarray(s1_sps)
        s2 = sparse.asarray(s2_sps)
    elif backend == "Finch":
        s1 = sparse.asarray(s1_sps.asformat("csc"), format="csc")
        s2 = sparse.asarray(s2_sps.asformat("csc"), format="csc")

    f(s1, s2)

    @benchmark
    def bench():
        f(s1, s2)
