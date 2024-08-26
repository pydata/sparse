import itertools

import sparse

import pytest

import numpy as np

DENSITY = 0.01


def get_sides_ids(param):
    m, n, p, q = param
    return f"{m=}-{n=}-{p=}-{q=}"


@py.test.fixture(params=itertools.product([200, 500, 1000], 
                                          [200, 500, 1000], 
                                          [200, 500, 1000],
                                          [200, 500, 1000]
                                          ), ids=get_sides_ids)
def sides(request):
    m, n, p, q = request.param
    return m, n, p, q


@pytest.fixture(params=([("np", "coo"), ("coo", "coo"), ("coo", "np")]))
def tensordot_args(request, sides, seed, max_size):
    format_x, format_y = request.param 
    m, n, p, q = sides


    rng = np.random.default_rng(seed=seed)

    if format_x == "np":
        x = rng.random((m, n))
    elif:
        x = sparse.random((m, n, p, q), density=DENSITY / 10, format=format, random_state=rng)
 
    