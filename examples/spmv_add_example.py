import importlib
import os

import sparse

from utils import benchmark

import numpy as np
import scipy.sparse as sps

LEN = 100000
DENSITY = 0.000001
ITERS = 3
rng = np.random.default_rng(0)


if __name__ == "__main__":
    print("SpMv_add Example:\n")

    A_sps = sps.random(LEN - 10, LEN, format="csc", density=DENSITY, random_state=rng) * 10
    x_sps = rng.random((LEN, 1)) * 10
    y_sps = rng.random((LEN - 10, 1)) * 10

    # ======= Finch =======
    os.environ[sparse._ENV_VAR_NAME] = "Finch"
    importlib.reload(sparse)

    A = sparse.asarray(A_sps)
    x = sparse.asarray(np.array(x_sps, order="C"))
    y = sparse.asarray(np.array(y_sps, order="C"))

    @sparse.compiled()
    def spmv_finch(A, x, y):
        return sparse.sum(A[:, None, :] * sparse.permute_dims(x, (1, 0))[None, :, :], axis=-1) + y

    # Compile & Benchmark
    result_finch = benchmark(spmv_finch, args=[A, x, y], info="Finch", iters=ITERS)

    # ======= Numba =======
    os.environ[sparse._ENV_VAR_NAME] = "Numba"
    importlib.reload(sparse)

    A = sparse.asarray(A_sps, format="csc")
    x = x_sps
    y = y_sps

    def spmv_numba(A, x, y):
        return A @ x + y

    # Compile & Benchmark
    result_numba = benchmark(spmv_numba, args=[A, x, y], info="Numba", iters=ITERS)

    # ======= SciPy =======
    def spmv_scipy(A, x, y):
        return A @ x + y

    A = A_sps
    x = x_sps
    y = y_sps

    # Compile & Benchmark
    result_scipy = benchmark(spmv_scipy, args=[A, x, y], info="SciPy", iters=ITERS)

    np.testing.assert_allclose(result_numba, result_scipy)
    np.testing.assert_allclose(result_finch.todense(), result_numba)
    np.testing.assert_allclose(result_finch.todense(), result_scipy)
