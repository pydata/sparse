import itertools
import operator

import sparse

import pytest

import numpy as np

DENSITY = 0.01


def side_ids(side):
    return f"{side=}"


def get_matmul_ids(params):
    side, format = params
    return f"{side=}-{format=}"


@pytest.fixture(params=itertools.product([100, 500, 1000], ["coo", "gcxs"]), ids=get_matmul_ids)
def test_matmul(benchmark, side, format, seed, max_size):
    if side**2 >= max_size:
        pytest.skip()
    rng = np.random.default_rng(seed=seed)
    x = sparse.random((side, side), density=DENSITY, format=format, random_state=rng)
    y = sparse.random((side, side), density=DENSITY, format=format, random_state=rng)

    x @ y  # Numba compilation

    @benchmark
    def bench():
        x @ y


def get_test_id(param):
    side, rank, format = param
    return f"{side=}-{rank=}"  # -{format=}"


@pytest.fixture(params=itertools.product([100, 500, 1000], [1, 2, 3, 4], ["coo", "gcxs"]), ids=get_test_id)
def elemwise_args(request, seed, max_size):
    side, rank, format = request.param
    if side**rank >= max_size:
        pytest.skip()
    rng = np.random.default_rng(seed=seed)
    shape = (side,) * rank
    x = sparse.random(shape, density=DENSITY, format=format, random_state=rng)
    y = sparse.random(shape, density=DENSITY, format=format, random_state=rng)
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


@pytest.fixture(params=itertools.product([100, 500, 1000], [1, 2, 3], ["coo", "gcxs"]), ids=get_test_id)
def indexing_args(request, seed, max_size):
    side, rank, format = request.param
    if side**rank >= max_size:
        pytest.skip()
    rng = np.random.default_rng(seed=seed)
    shape = (side,) * rank

    return sparse.random(shape, density=DENSITY, format=format, random_state=rng)


def test_index_scalar(benchmark, indexing_args):
    x = indexing_args
    side = x.shape[0]
    rank = x.ndim

    x[(side // 2,) * rank]  # Numba compilation

    @benchmark
    def bench():
        x[(side // 2,) * rank]


def test_index_slice(benchmark, indexing_args):
    x = indexing_args
    side = x.shape[0]
    rank = x.ndim

    x[(slice(side // 2),) * rank]  # Numba compilation

    @benchmark
    def bench():
        x[(slice(side // 2),) * rank]


def test_index_fancy(benchmark, indexing_args, seed):
    x = indexing_args
    side = x.shape[0]
    rng = np.random.default_rng(seed=seed)
    index = rng.integers(0, side, size=(side // 2,))

    x[index]  # Numba compilation

    @benchmark
    def bench():
        x[index]


def get_densemul_id(param):
    compressed_axis, n_vectors = param
    return f"{compressed_axis=}-{n_vectors}"


@pytest.fixture(params=itertools.product([0, 1], [1, 20, 100]), ids=get_densemul_id)
def densemul_args(request, seed):
    compressed_axis, n_vectors = request.param

    rng = np.random.default_rng(seed=seed)
    n = 10000
    x = sparse.random((n, n), density=DENSITY / 10, format="gcxs", random_state=rng).change_compressed_axes(
        (compressed_axis,)
    )
    t = rng.random((n, n_vectors))

    return x, t


def test_gcxs_dot_ndarray(benchmark, densemul_args):
    x, t = densemul_args

    # Numba compilation
    x @ t

    @benchmark
    def bench():
        x @ t


def test_ndarray_dot_gcxs(benchmark, densemul_args):
    x, t = densemul_args

    u = t.T

    # Numba compilation
    u @ x

    @benchmark
    def bench():
        u @ x
