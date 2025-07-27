import pytest

import numpy as np


@pytest.fixture(scope="function")
def rng(scope="session"):
    return np.random.default_rng(seed=42)


@pytest.fixture
def max_size(scope="session"):
    return 2**26
