import sparse

import pytest

import numpy as np
import scipy.sparse as sp


def test_backend_contex_manager(backend):
    if backend == sparse.BackendType.Finch:
        with pytest.raises(NotImplementedError):
            sparse.COO.from_numpy(np.eye(5))
    else:
        sparse.COO.from_numpy(np.eye(5))


def test_finch_backend():
    np_eye = np.eye(5)
    sp_arr = sp.csr_matrix(np_eye)

    with sparse.Backend(backend=sparse.BackendType.Finch):
        finch_dense = sparse.Tensor(np_eye)

        assert np.shares_memory(finch_dense.todense(), np_eye)

        finch_arr = sparse.Tensor(sp_arr)

        np.testing.assert_equal(finch_arr.todense(), np_eye)

        transposed = sparse.permute_dims(finch_arr, (1, 0))

        np.testing.assert_equal(transposed.todense(), np_eye.T)
