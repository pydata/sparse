import sparse

import pytest

if sparse._BackendType.Numba != sparse._BACKEND:
    pytest.skip("skipping Numba tests", allow_module_level=True)
