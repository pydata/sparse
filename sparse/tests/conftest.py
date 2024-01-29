import pytest


@pytest.fixture(scope="session")
def rng():
    from sparse._utils import default_rng

    return default_rng
