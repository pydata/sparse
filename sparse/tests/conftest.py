import sparse

import pytest

import numpy as np


@pytest.fixture(scope="session", params=[sparse.BackendType.Numba, sparse.BackendType.Finch])
def backend(request):
    with sparse.Backend(backend=request.param):
        yield request.param


@pytest.fixture(scope="module")
def graph():
    return np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
        ]
    )
