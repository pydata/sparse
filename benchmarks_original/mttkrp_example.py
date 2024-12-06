import importlib
import os

import sparse

from utils import benchmark

import numpy as np

I_ = 1000
J_ = 25
K_ = 1000
L_ = 100
DENSITY = 0.0001
ITERS = 3
rng = np.random.default_rng(0)


if __name__ == "__main__":
    print("MTTKRP Example:\n")

    B_sps = sparse.random((I_, K_, L_), density=DENSITY, random_state=rng) * 10
    D_sps = rng.random((L_, J_)) * 10
    C_sps = rng.random((K_, J_)) * 10

    # ======= Finch =======
    os.environ[sparse._ENV_VAR_NAME] = "Finch"
    importlib.reload(sparse)

    B = sparse.asarray(B_sps.todense(), format="csf")
    D = sparse.asarray(np.array(D_sps, order="F"))
    C = sparse.asarray(np.array(C_sps, order="F"))

    @sparse.compiled()
    def mttkrp_finch(B, D, C):
        return sparse.sum(B[:, :, :, None] * D[None, None, :, :] * C[None, :, None, :], axis=(1, 2))

    # Compile & Benchmark
    result_finch = benchmark(mttkrp_finch, args=[B, D, C], info="Finch", iters=ITERS)

    # ======= Numba =======
    os.environ[sparse._ENV_VAR_NAME] = "Numba"
    importlib.reload(sparse)

    B = sparse.asarray(B_sps, format="gcxs")
    D = D_sps
    C = C_sps

    def mttkrp_numba(B, D, C):
        return sparse.sum(B[:, :, :, None] * D[None, None, :, :] * C[None, :, None, :], axis=(1, 2))

    # Compile & Benchmark
    result_numba = benchmark(mttkrp_numba, args=[B, D, C], info="Numba", iters=ITERS)

    np.testing.assert_allclose(result_finch.todense(), result_numba.todense())
