import importlib
import os

import sparse

from utils import benchmark

import numpy as np
import scipy.sparse as sps

LEN = 100000
DENSITY = 0.00001
ITERS = 3
rng = np.random.default_rng(0)


if __name__ == "__main__":
    print("Matmul Example:\n")

    a_sps = sps.random(LEN, LEN - 10, format="csr", density=DENSITY, random_state=rng) * 10
    a_sps.sum_duplicates()
    b_sps = sps.random(LEN - 10, LEN, format="csr", density=DENSITY, random_state=rng) * 10
    b_sps.sum_duplicates()

    # ======= Finch =======
    os.environ[sparse._ENV_VAR_NAME] = "Finch"
    importlib.reload(sparse)

    a = sparse.asarray(a_sps)
    b = sparse.asarray(b_sps)

    @sparse.compiled()
    def sddmm_finch(a, b):
        return a @ b

    # Compile & Benchmark
    result_finch = benchmark(sddmm_finch, args=[a, b], info="Finch", iters=ITERS)

    # ======= Numba =======
    os.environ[sparse._ENV_VAR_NAME] = "Numba"
    importlib.reload(sparse)

    a = sparse.asarray(a_sps)
    b = sparse.asarray(b_sps)

    def sddmm_numba(a, b):
        return a @ b

    # Compile & Benchmark
    result_numba = benchmark(sddmm_numba, args=[a, b], info="Numba", iters=ITERS)

    # ======= SciPy =======
    def sddmm_scipy(a, b):
        return a @ b

    a = a_sps
    b = b_sps

    # Compile & Benchmark
    result_scipy = benchmark(sddmm_scipy, args=[a, b], info="SciPy", iters=ITERS)

    # np.testing.assert_allclose(result_numba.todense(), result_scipy.toarray())
    # np.testing.assert_allclose(result_finch.todense(), result_numba.todense())
    # np.testing.assert_allclose(result_finch.todense(), result_scipy.toarray())
