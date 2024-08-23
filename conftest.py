import pytest


@pytest.fixture(scope="session", autouse=True)
def add_doctest_modules(doctest_namespace):
    import sparse

    import numpy as np

    if sparse._BackendType.Numba != sparse._BACKEND:
        pass  # TODO: pytest.skip() skips Finch and MLIR tests

    doctest_namespace["np"] = np
    doctest_namespace["sparse"] = sparse
