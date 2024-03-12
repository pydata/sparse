import pytest
import numpy as np

import sparse


@pytest.mark.parametrize("backend", [sparse.BackendType.pydata, sparse.BackendType.finch])
def test_backend_contex_manager(backend):

    sparse.COO.from_numpy(np.eye(5))

    with sparse.Backend(backend=backend):

        try:
            sparse.COO.from_numpy(np.eye(5))
        except NotImplementedError:
            pass

    sparse.COO.from_numpy(np.eye(5))
