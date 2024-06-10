import os

import sparse

import pytest

import numpy as np


@pytest.fixture(scope="session")
def backend():
    yield sparse.BackendType[os.environ[sparse._ENV_VAR_NAME]]


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
