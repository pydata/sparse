import itertools
import operator

import sparse

import pytest

import numpy as np

DENSITY = 0.01


def side_ids(side):
    return f"{side=}"


@pytest.mark.parametrize("side", [100, 500, 1000], ids=side_ids)
def test_matmul(benchmark, side, seed):
    if side**2 >= 2**26:
        pytest.skip()
    rng = np.random.default_rng(seed=seed)
    x = sparse.random((side, side), density=DENSITY, random_state=rng)
    y = sparse.random((side, side), density=DENSITY, random_state=rng)

    x @ y  # Numba compilation

    @benchmark
    def bench():
        x @ y


def elemwise_test_name(param):
    side, rank = param
    return f"{side=}-{rank=}"


@pytest.fixture(params=itertools.product([100, 500, 1000], [1, 2, 3, 4]), ids=elemwise_test_name)
def elemwise_args(request, seed):
    side, rank = request.param
    if side**rank >= 2**26:
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
def elemwise_broadcast_args(request, seed):
    side = request.param
    if side**2 >= 2**26:
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


@pytest.fixture(params=[100, 500, 1000], ids=side_ids)
def indexing_args(request, seed):
    side = request.param
    if side**3 >= 2**26:
        pytest.skip()
    rng = np.random.default_rng(seed=seed)

    return sparse.random((side, side, side), density=DENSITY, random_state=rng)


def test_index_scalar(benchmark, indexing_args):
    x = indexing_args
    side = x.shape[0]

    x[side // 2, side // 2, side // 2]  # Numba compilation

    @benchmark
    def bench():
        x[side // 2, side // 2, side // 2]


def test_index_slice(benchmark, indexing_args):
    x = indexing_args
    side = x.shape[0]

    x[: side // 2]  # Numba compilation

    @benchmark
    def bench():
        x[: side // 2]


def test_index_slice2(benchmark, indexing_args):
    x = indexing_args
    side = x.shape[0]

    x[: side // 2, : side // 2]  # Numba compilation

    @benchmark
    def bench():
        x[: side // 2, : side // 2]


def test_index_slice3(benchmark, indexing_args):
    x = indexing_args
    side = x.shape[0]

    x[: side // 2, : side // 2, : side // 2]  # Numba compilation

    @benchmark
    def bench():
        x[: side // 2, : side // 2, : side // 2]


def test_index_fancy(benchmark, indexing_args, seed):
    x = indexing_args
    side = x.shape[0]
    rng = np.random.default_rng(seed=seed)
    index = rng.integers(0, side, side // 2)

    x[index]  # Numba compilation

    @benchmark
    def bench():
        x[index]
