import itertools
import operator

import sparse

import pytest

import numpy as np

DENSITY = 0.01


def side_ids(side):
    return f"{side=}"


@pytest.mark.parametrize("side", [100, 500, 1000], ids=side_ids)
def test_matmul(benchmark, side, seed, max_size):
    if side**2 >= max_size:
        pytest.skip()
    rng = np.random.default_rng(seed=seed)
    x = sparse.random((side, side), density=DENSITY, random_state=rng)
    y = sparse.random((side, side), density=DENSITY, random_state=rng)

    x @ y  # Numba compilation

    @benchmark
    def bench():
        x @ y


def id_of_test(param):
    side, rank = param
    return f"{side=}-{rank=}"


@pytest.fixture(params=itertools.product([100, 500, 1000], [1, 2, 3, 4]), ids=id_of_test)
def elemwise_args(request, seed, max_size):
    side, rank = request.param
    if side**rank >= max_size:
        pytest.skip()
    rng = np.random.default_rng(seed=seed)
    shape = (side,) * rank
    x = sparse.random(shape, density=DENSITY, random_state=rng)
    y = sparse.random(shape, density=DENSITY, random_state=rng)
    return x, y


@pytest.mark.parametrize("f", [operator.add, operator.mul])
def test_elemwise(benchmark, f, elemwise_args):
    x, y = elemwise_args
    f(x, y)

    @benchmark
    def bench():
        f(x, y)


@pytest.fixture(params=[100, 500, 1000], ids=side_ids)
def elemwise_broadcast_args(request, seed, max_size):
    side = request.param
    if side**2 >= max_size:
        pytest.skip()
    rng = np.random.default_rng(seed=seed)
    x = sparse.random((side, 1, side), density=DENSITY, random_state=rng)
    y = sparse.random((side, side), density=DENSITY, random_state=rng)
    return x, y


@pytest.mark.parametrize("f", [operator.add, operator.mul])
def test_elemwise_broadcast(benchmark, f, elemwise_broadcast_args):
    x, y = elemwise_broadcast_args
    f(x, y)

    @benchmark
    def bench():
        f(x, y)


@pytest.fixture(params=itertools.product([100, 500, 1000], [1, 2, 3]), ids=id_of_test)
def indexing_args(request, seed, max_size):
    side, rank = request.param
    if side**3 >= max_size:
        pytest.skip()
    rng = np.random.default_rng(seed=seed)
    shape = (side,) * rank
    x = sparse.random(shape, density=DENSITY, random_state=rng)
    return x


def test_index_scalar(benchmark, indexing_args):
    x = indexing_args
    side = x.shape[0]
    ndim = len(x.shape)

    x[(side // 2,) * ndim]  # Numba compilation

    @benchmark
    def bench():
        x[(side // 2,) * ndim]


def test_index_slice(benchmark, indexing_args):
    x = indexing_args
    side = x.shape[0]
    rank = len(x.shape)

    x[(slice(side // 2),) * rank]  # Numba compilation

    @benchmark
    def bench():
        x[(slice(side // 2),) * rank]


def test_index_fancy(benchmark, indexing_args, seed):
    x = indexing_args
    side = x.shape[0]
    rank = len(x.shape)
    rng = np.random.default_rng(seed=seed)
    index = rng.integers((side // 2,) * rank)

    x[index]  # Numba compilation

    @benchmark
    def bench():
        x[index]
