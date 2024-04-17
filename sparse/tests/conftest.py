import sparse

import pytest


@pytest.fixture(scope="session", params=[sparse.BackendType.Numba, sparse.BackendType.Finch])
def backend(request):
    with sparse.Backend(backend=request.param):
        yield request.param
