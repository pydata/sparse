import itertools

import sparse

import pytest

import numpy as np

DENSITY = 0.01


def get_sides_ids(param):
    m, n, p, q = param
    return f"{m=}-{n=}-{p=}-{q=}"


@pytest.fixture(
    params=itertools.product([10, 50], [10, 20], [20, 50], [10, 50]),
    ids=get_sides_ids,
    scope="function",
)
def sides(request):
    m, n, p, q = request.param
    return m, n, p, q


def get_tensor_ids(param):
    left_index, right_index, left_format, right_format = param
    return f"{left_index=}-{right_index=}-{left_format=}-{right_format=}"


@pytest.fixture(
    params=([(1, 2, "dense", "coo"), (1, 2, "coo", "coo"), (1, 1, "coo", "dense")]),
    ids=get_tensor_ids,
    scope="function",
)
def tensordot_args(request, sides, seed, max_size):
    m, n, p, q = sides
    if m * n * p * q >= max_size:
        pytest.skip()
    left_index, right_index, left_format, right_format = request.param
    rng = np.random.default_rng(seed=seed)

    t = rng.random((m, n))

    if left_format == "dense" and right_format == "coo":
        left_tensor = t
        right_tensor = sparse.random((m, p, n, q), density=DENSITY, format=right_format, random_state=rng)

    if left_format == "coo" and right_format == "coo":
        left_tensor = sparse.random((m, p), density=DENSITY, format=left_format, random_state=rng)
        right_tensor = sparse.random((m, n, p, q), density=DENSITY, format=right_format, random_state=rng)

    if left_format == "coo" and right_format == "dense":
        left_tensor = sparse.random((m, n, p, q), density=DENSITY, format=left_format, random_state=rng)
        right_tensor = t

    return left_index, right_index, left_tensor, right_tensor


@pytest.mark.parametrize("return_type", [np.ndarray, sparse.COO])
def test_tensordot(benchmark, return_type, tensordot_args):
    left_index, right_index, left_tensor, right_tensor = tensordot_args

    sparse.tensordot(left_tensor, right_tensor, axes=([0, left_index], [0, right_index]), return_type=return_type)

    @benchmark
    def bench():
        sparse.tensordot(left_tensor, right_tensor, axes=([0, left_index], [0, right_index]), return_type=return_type)
