import importlib
import os

import pytest

import numpy as np


@pytest.fixture(scope="session", params=["Numba", "Finch"])
def backend(request):
    import sparse

    os.environ[sparse._ENV_VAR_NAME] = request.param
    importlib.reload(sparse)

    yield sparse.BackendType[request.param]

    os.environ[sparse._ENV_VAR_NAME] = sparse.BackendType.Numba.value
    importlib.reload(sparse)


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
