import sparse

import pytest


@pytest.fixture(scope="session", params=[sparse.BackendType.PyData, sparse.BackendType.Finch])
def backend(request):
    with sparse.Backend(backend=request.param):
        yield request.param
