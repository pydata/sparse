import sparse

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_equal


def test_backend_contex_manager(backend):
    rng = np.random.default_rng(0)
    x = sparse.random((100, 10, 100), density=0.01, random_state=rng)
    y = sparse.random((100, 10, 100), density=0.01, random_state=rng)

    if backend == sparse.BackendType.Finch:
        import finch

        def storage():
            return finch.Storage(finch.Dense(finch.SparseList(finch.SparseList(finch.Element(0.0)))), order="C")

        x = x.to_device(storage())
        y = y.to_device(storage())
    else:
        x.asformat("gcxs")
        y.asformat("gcxs")

    z = x + y
    result = sparse.sum(z)
    assert result.shape == ()


def test_finch_backend():
    np_eye = np.eye(5)
    sp_arr = sp.csr_matrix(np_eye)

    with sparse.Backend(backend=sparse.BackendType.Finch):
        import finch

        finch_dense = finch.Tensor(np_eye)

        assert np.shares_memory(finch_dense.todense(), np_eye)

        finch_arr = finch.Tensor(sp_arr)

        assert_equal(finch_arr.todense(), np_eye)

        transposed = sparse.permute_dims(finch_arr, (1, 0))

        assert_equal(transposed.todense(), np_eye.T)

        @sparse.compiled
        def my_fun(tns1, tns2):
            tmp = sparse.add(tns1, tns2)
            return sparse.sum(tmp, axis=0)

        result = my_fun(finch_dense, finch_arr)

        assert_equal(result.todense(), np.sum(2 * np_eye, axis=0))
