import itertools
import operator

import sparse

import pytest

import numpy as np

DENSITY = 0.01
SEED = 42


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
