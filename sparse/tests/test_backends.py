import sparse

import pytest

import numpy as np


def test_backend_contex_manager(backend):
    if backend == sparse.BackendType.finch:
        with pytest.raises(NotImplementedError):
            sparse.COO.from_numpy(np.eye(5))
    else:
        sparse.COO.from_numpy(np.eye(5))
