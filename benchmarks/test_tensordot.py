import itertools

import sparse

import pytest

import numpy as np

DENSITY = 0.01


def get_sides_ids(param):
    m, n, p, q = param
    return f"{m=}-{n=}-{p=}-{q=}"


@pytest.fixture(
    params=itertools.product([200, 500, 1000], [200, 500, 1000], [200, 500, 1000], [200, 500, 1000]), ids=get_sides_ids
)
def sides(request):
    m, n, p, q = request.param
    return m, n, p, q


# def axis(request):
#   ([0, 1], [0, 2]) (m n) (m p) numpy
#   ([0, 1], [0, 2]) (m n) (m p) sparse
#
#   ([0, 1], [0, 2]) (m n) (m p) numpy
#   ([0, 1], [0, 2]) (m n) (m p) sparse
#
#   ([0, 1], [0, 1]) (m n) (m n) numpy
#   ([0, 1], [0, 1]) (m n) (m n) sparse


# @pytest.fixture(params=([(0,1,0,2), (0,1,0,2), (0,1,0,1)]))
#                         (m,n,m,p) (m,n,m,p) (m,n,m,n)


def get_tensor_ids(param):
    left_index, right_index, left_format, right_format = param
    return f"{left_index=}-{right_index=}-{left_format=}-{right_format=}"


@pytest.fixture(params=([(1, 2, "dense", "coo"), (1, 2, "coo", "coo"), (1, 1, "coo", "dense")]), ids=get_tensor_ids)
def tensordot_args(request, sides, seed, max_size):
    left_index, right_index, left_format, right_format = request.param
    m, n, p, q = sides
    if m * n >= max_size or n * p >= max_size:
        pytest.skip()
    rng = np.random.default_rng(seed=seed)

    t = rng.random((m, n))

    if left_format == "dense":
        left_tensor = t
    if left_format == "coo":
        left_tensor = sparse.random((m, n, p, q), density=DENSITY, format=left_format, random_state=rng)
    if right_format == "coo":
        right_tensor = sparse.random((m, n, p, q), density=DENSITY, format=right_format, random_state=rng)
    if right_format == "dense":
        right_tensor = t

    return left_index, right_index, left_tensor, right_tensor


@pytest.mark.parametrize("return_type", [np.ndarray, sparse.COO])
def test_tensordot(benchmark, return_type, tensordot_args):
    left_index, right_index, left_tensor, right_tensor = tensordot_args

    sparse.tensordot(left_tensor, right_tensor, axes=([0, left_index], [0, right_index]), return_type=return_type)

    @benchmark
    def bench():
        sparse.tensordot(left_tensor, right_tensor, axes=([0, left_index], [0, right_index]), return_type=return_type)
