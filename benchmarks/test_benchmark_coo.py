import itertools
import operator

import sparse

import pytest

import numpy as np

from .utils import DENSITY, SEED


def side_ids(side):
    return f"{side=}"


@pytest.mark.parametrize("side", [100, 500, 1000], ids=side_ids)
def test_matmul(benchmark, side):
    rng = np.random.default_rng(seed=SEED)
    x = sparse.random((side, side), density=DENSITY, random_state=rng)
    y = sparse.random((side, side), density=DENSITY, random_state=rng)

    x @ y  # Numba compilation

    @benchmark
    def bench():
        x @ y


def elemwise_test_name(param):
    side, rank = param
    return f"{side=}-{rank=}"


@pytest.fixture(scope="module", params=itertools.product([100, 500, 1000], [1, 2, 3, 4]), ids=elemwise_test_name)
def elemwise_args(request):
    side, rank = request.param
    if side**rank >= 2**26:
        pytest.skip()
    rng = np.random.default_rng(seed=SEED)
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


@pytest.fixture(scope="module", params=[100, 500, 1000], ids=side_ids)
def elemwise_broadcast_args(request):
    side = request.param
    rng = np.random.default_rng(seed=SEED)
    if side**side >= 2**26:
        pytest.skip()
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


@pytest.fixture(scope="module", params=[100, 500, 1000], ids=side_ids)
def indexing_args(request):
    side = request.param
    rng = np.random.default_rng(seed=SEED)
    if side**side >= 2**26:
        pytest.skip()

    x = sparse.random((side, side, side), density=DENSITY, random_state=rng)

    return x


def test_index_scalar(benchmark, indexing_args):
    x = indexing_args
    side = x.shape([0])

    x[5]  # Numba compilation

    @benchmark
    def bench():
        x[side // 2, side // 2, side // 2]


def test_index_slice(benchmark, indexing_args):
    x = indexing_args
    side = x.shape([0])

    x[5]  # Numba compilation

    @benchmark
    def bench():
        x[: side // 2]


def test_index_slice2(benchmark, indexing_args):
    x = indexing_args
    side = x.shape([0])

    x[5]  # Numba compilation

    @benchmark
    def bench():
        x[: side // 2, : side // 2]


def test_index_slice3(benchmark, indexing_args):
    x = indexing_args
    side = x.shape([0])

    x[5]  # Numba compilation

    @benchmark
    def bench():
        x[: side // 2, : side // 2, : side // 2]


def test_index_fancy(benchmark, indexing_args):
    x = indexing_args
    side = x.shape([0])
    rng = np.random.default_rng(seed=SEED)
    index = rng.integers(0, side, side // 2)

    x[index]  # Numba compilation

    @benchmark
    def bench():
        x[index]
