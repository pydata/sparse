import importlib
import os
import time

import sparse

import numpy as np
import scipy.sparse as sps

LEN = 100000
DENSITY = 0.000001
ITERS = 3
rng = np.random.default_rng(0)


def benchmark(func, info, args):
    print(info)
    start = time.time()
    for _ in range(ITERS):
        func(*args)
    elapsed = time.time() - start
    print(f"Took {elapsed / ITERS} s.\n")


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

    @sparse.compiled
    def spmv_finch(A, x, y):
        return sparse.sum(A[:, None, :] * sparse.permute_dims(x, (1, 0))[None, :, :], axis=-1) + y

    # Compile
    result_finch = spmv_finch(A, x, y)
    assert sparse.nonzero(result_finch)[0].size > 5
    # Benchmark
    benchmark(spmv_finch, info="Finch", args=[A, x, y])

    # ======= Numba =======
    os.environ[sparse._ENV_VAR_NAME] = "Numba"
    importlib.reload(sparse)

    A = sparse.asarray(A_sps, format="csc")
    x = x_sps
    y = y_sps

    def spmv_numba(A, x, y):
        return A @ x + y

    # Compile
    result_numba = spmv_numba(A, x, y)
    assert sparse.nonzero(result_numba)[0].size > 5
    # Benchmark
    benchmark(spmv_numba, info="Numba", args=[A, x, y])

    # ======= SciPy =======
    def spmv_scipy(A, x, y):
        return A @ x + y

    A = A_sps
    x = x_sps
    y = y_sps

    result_scipy = spmv_scipy(A, x, y)
    # Benchmark
    benchmark(spmv_scipy, info="SciPy", args=[A, x, y])

    np.testing.assert_allclose(result_numba, result_scipy)
    np.testing.assert_allclose(result_finch.todense(), result_numba)
    np.testing.assert_allclose(result_finch.todense(), result_scipy)
