import pytest

import sparse

if sparse._BACKEND != sparse._BackendType.Numba:
    pytest.skip("skipping Numba tests", allow_module_level=True)
