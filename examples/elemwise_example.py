import importlib
import operator
import os

import sparse

from utils import benchmark

import numpy as np
import scipy.sparse as sps

LEN = 10000
DENSITY = 0.001
ITERS = 3
rng = np.random.default_rng(0)


if __name__ == "__main__":
    print("Elementwise Example:\n")

    for func_name in ["multiply", "add", "greater_equal"]:
        print(f"{func_name} benchmark:\n")

        s1_sps = sps.random(LEN, LEN, format="csr", density=DENSITY, random_state=rng) * 10
        s1_sps.sum_duplicates()
        s2_sps = sps.random(LEN, LEN, format="csr", density=DENSITY, random_state=rng) * 10
        s2_sps.sum_duplicates()

        # ======= Finch =======
        os.environ[sparse._ENV_VAR_NAME] = "Finch"
        importlib.reload(sparse)

        s1 = sparse.asarray(s1_sps.asformat("csc"), format="csc")
        s2 = sparse.asarray(s2_sps.asformat("csc"), format="csc")

        func = getattr(sparse, func_name)

        # Compile & Benchmark
        result_finch = benchmark(func, args=[s1, s2], info="Finch", iters=ITERS)

        # ======= Numba =======
        os.environ[sparse._ENV_VAR_NAME] = "Numba"
        importlib.reload(sparse)

        s1 = sparse.asarray(s1_sps)
        s2 = sparse.asarray(s2_sps)

        func = getattr(sparse, func_name)

        # Compile & Benchmark
        result_numba = benchmark(func, args=[s1, s2], info="Numba", iters=ITERS)

        # ======= SciPy =======
        s1 = s1_sps
        s2 = s2_sps

        if func_name == "multiply":
            func, args = s1.multiply, [s2]
        elif func_name == "add":
            func, args = operator.add, [s1, s2]
        elif func_name == "greater_equal":
            func, args = operator.ge, [s1, s2]

        # Compile & Benchmark
        result_scipy = benchmark(func, args=args, info="SciPy", iters=ITERS)

        np.testing.assert_allclose(result_numba.todense(), result_scipy.toarray())
        np.testing.assert_allclose(result_finch.todense(), result_numba.todense())
        np.testing.assert_allclose(result_finch.todense(), result_scipy.toarray())
