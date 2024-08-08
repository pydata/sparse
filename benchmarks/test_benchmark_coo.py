import operator

import sparse

import pytest

import numpy as np

DENSITY = 0.01
SEED = 42


@pytest.fixture(scope="module")
@pytest.mark.parametrize("side", [100, 500, 1000])
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def elemwise_args(side: int, rank: int):
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
