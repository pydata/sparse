import pytest


@pytest.fixture
def seed(scope="session"):
    return 42


@pytest.fixture
def max_size(scope="session"):
    return 2**26
