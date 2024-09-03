import pytest

import sparse

if sparse._BACKEND == sparse._BackendType.MLIR:
    pytest.skip("skipping backend tests", allow_module_level=True)
