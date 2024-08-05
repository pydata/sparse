import pytest


@pytest.fixture(scope="session", autouse=True)
def add_doctest_modules(doctest_namespace):
    import sparse

    import numpy as np

    if sparse.BackendType.Numba != sparse.BACKEND:
        pytest.skip()

    doctest_namespace["np"] = np
    doctest_namespace["sparse"] = sparse
