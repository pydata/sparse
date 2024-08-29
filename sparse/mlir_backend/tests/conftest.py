import pytest

import numpy as np


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
