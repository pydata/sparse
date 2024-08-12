import sparse

import pytest

import numpy as np


@pytest.fixture(scope="session")
def backend():
    yield sparse._BACKEND


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
