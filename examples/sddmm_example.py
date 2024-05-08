import time

import sparse

import numpy as np
import scipy.sparse as sps

LEN = 10000
DENSITY = 0.0001
ITERS = 5
rng = np.random.default_rng(0)


def benchmark(func, info, args):
    print(info)
    start = time.time()
    for _ in range(ITERS):
        func(*args)
    elapsed = time.time() - start
    print(f"Took {elapsed / ITERS} s.\n")


if __name__ == "__main__":
    a_sps = rng.random((LEN, LEN - 10)) * 10
    b_sps = rng.random((LEN, LEN - 10)) * 10
    s_sps = sps.random(LEN, LEN, format="coo", density=DENSITY, random_state=rng) * 10
    s_sps.sum_duplicates()

    # Finch
    with sparse.Backend(backend=sparse.BackendType.Finch):
        s = sparse.asarray(s_sps)
        a = sparse.asarray(np.array(a_sps, order="F"))
        b = sparse.asarray(np.array(b_sps, order="F"))

        @sparse.compiled
        def sddmm_finch(s, a, b):
            return sparse.sum(s[:, :, None] * (a[:, None, :] * b[None, :, :]), axis=-1)

        # Compile
        result_finch = sddmm_finch(s, a, b)
        assert sparse.nonzero(result_finch)[0].size > 5
        # Benchmark
        benchmark(sddmm_finch, info="Finch", args=[s, a, b])

    # Numba
    with sparse.Backend(backend=sparse.BackendType.Numba):
        s = sparse.asarray(s_sps)
        a = a_sps
        b = b_sps

        def sddmm_numba(s, a, b):
            return s * (a @ b.T)

        # Compile
        result_numba = sddmm_numba(s, a, b)
        assert sparse.nonzero(result_numba)[0].size > 5
        # Benchmark
        benchmark(sddmm_numba, info="Numba", args=[s, a, b])

    # SciPy
    def sddmm_scipy(s, a, b):
        return s.multiply(a @ b.T)

    s = s_sps.asformat("csr")
    a = a_sps
    b = b_sps

    result_scipy = sddmm_scipy(s, a, b)
    # Benchmark
    benchmark(sddmm_scipy, info="SciPy", args=[s, a, b])

    np.testing.assert_allclose(result_numba.todense(), result_scipy.toarray())
    np.testing.assert_allclose(result_finch.todense(), result_numba.todense())
    np.testing.assert_allclose(result_finch.todense(), result_scipy.toarray())
