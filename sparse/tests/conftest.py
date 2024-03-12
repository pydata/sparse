import sparse

import pytest


@pytest.fixture(scope="session", params=[sparse.BackendType.pydata, sparse.BackendType.finch])
def backend(request):
    with sparse.Backend(backend=request.param):
        yield request.param
