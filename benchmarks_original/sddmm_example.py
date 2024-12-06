import importlib
import os

import sparse

from utils import benchmark

import numpy as np
import scipy.sparse as sps

LEN = 10000
DENSITY = 0.00001
ITERS = 3
rng = np.random.default_rng(0)


if __name__ == "__main__":
    print("SDDMM Example:\n")

    a_sps = rng.random((LEN, LEN - 10)) * 10
    b_sps = rng.random((LEN - 10, LEN)) * 10
    s_sps = sps.random(LEN, LEN, format="coo", density=DENSITY, random_state=rng) * 10
    s_sps.sum_duplicates()

    # ======= Finch =======
    os.environ[sparse._ENV_VAR_NAME] = "Finch"
    importlib.reload(sparse)

    s = sparse.asarray(s_sps)
    a = sparse.asarray(np.array(a_sps, order="F"))
    b = sparse.asarray(np.array(b_sps, order="C"))

    @sparse.compiled()
    def sddmm_finch(s, a, b):
        return sparse.sum(
            s[:, :, None] * (a[:, None, :] * sparse.permute_dims(b, (1, 0))[None, :, :]),
            axis=-1,
        )

    # Compile & Benchmark
    result_finch = benchmark(sddmm_finch, args=[s, a, b], info="Finch", iters=ITERS)

    # ======= Numba =======
    os.environ[sparse._ENV_VAR_NAME] = "Numba"
    importlib.reload(sparse)

    s = sparse.asarray(s_sps)
    a = a_sps
    b = b_sps

    def sddmm_numba(s, a, b):
        return s * (a @ b)

    # Compile & Benchmark
    result_numba = benchmark(sddmm_numba, args=[s, a, b], info="Numba", iters=ITERS)

    # ======= SciPy =======
    def sddmm_scipy(s, a, b):
        return s.multiply(a @ b)

    s = s_sps.asformat("csr")
    a = a_sps
    b = b_sps

    # Compile & Benchmark
    result_scipy = benchmark(sddmm_scipy, args=[s, a, b], info="SciPy", iters=ITERS)

    np.testing.assert_allclose(result_numba.todense(), result_scipy.toarray())
    np.testing.assert_allclose(result_finch.todense(), result_numba.todense())
    np.testing.assert_allclose(result_finch.todense(), result_scipy.toarray())
