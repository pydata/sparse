import pytest


@pytest.fixture
def seed(scope="session"):
    return 42
