import pathlib

import sparse

import pytest


@pytest.fixture(scope="session", autouse=True)
def add_doctest_modules(doctest_namespace):
    import sparse

    import numpy as np

    doctest_namespace["np"] = np
    doctest_namespace["sparse"] = sparse


def pytest_ignore_collect(collection_path: pathlib.Path, config: pytest.Config) -> bool | None:
    if "numba_backend" in collection_path.parts and sparse._BackendType.Numba != sparse._BACKEND:
        return True

    if "mlir_backend" in collection_path.parts and sparse._BackendType.MLIR != sparse._BACKEND:
        return True

    if "finch_backend" in collection_path.parts and sparse._BackendType.Finch != sparse._BACKEND:
        return True

    return None
