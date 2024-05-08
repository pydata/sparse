import time

import sparse

import numpy as np
import scipy.sparse as sps

LEN = 1000
DENSITY = 0.0001
ITERS = 3
rng = np.random.default_rng()


def benchmark(func, info, args):
    print(info)
    start = time.time()
    for i in range(ITERS):
        print(f"iter: {i}")
        func(*args)
    elapsed = time.time() - start
    print(f"Took {elapsed / ITERS} s.\n")


if __name__ == "__main__":
    # Finch
    with sparse.Backend(backend=sparse.BackendType.Finch):
        s = sparse.random((LEN, LEN), density=DENSITY, random_state=rng)
        a = sparse.asarray(np.array(rng.random((LEN, LEN)), order="F"))
        b = sparse.asarray(np.array(rng.random((LEN, LEN)), order="F"))

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
        s = sparse.random((LEN, LEN), density=DENSITY, random_state=rng)
        a = rng.random((LEN, LEN))
        b = rng.random((LEN, LEN))

        def sddmm_numba(s, a, b):
            return sparse.sum(s[:, :, None] * (a[:, None, :] * b[None, :, :]), axis=-1)

        # Compile
        result_numba = sddmm_numba(s, a, b)
        assert sparse.nonzero(result_numba)[0].size > 5
        # Benchmark
        benchmark(sddmm_numba, info="Numba", args=[s, a, b])

    # SciPy
    def sddmm_scipy(s, a, b):
        return s * (a @ b)

    s = sps.random_array((LEN, LEN), format="coo", density=DENSITY, random_state=rng)
    a = rng.random((LEN, LEN))
    b = rng.random((LEN, LEN))

    # Benchmark
    benchmark(sddmm_scipy, info="SciPy", args=[s, a, b])
