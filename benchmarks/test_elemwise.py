import itertools
import operator

import sparse

import pytest

import numpy as np

import os
import importlib

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


@pytest.mark.parametrize("f", [operator.add, operator.mul, operator.ge])
@pytest.mark.parametrize("backend", ["SciPy", "Numba", "Finch"])
def test_elemwise(benchmark, f, backend, elemwise_args):
    
    if backend in ["Numba", "Finch"]:
        os.environ[sparse._ENV_VAR_NAME] = backend
        importlib.reload(sparse)
    
    s1_sps, s2_sps = elemwise_args
    
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