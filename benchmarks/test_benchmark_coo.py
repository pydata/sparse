import sparse

import numpy as np


def test_matmul(benchmark):
    rng = np.random.default_rng(seed=42)
    x = sparse.random((3, 3), density=0.01, random_state=rng)
    y = sparse.random((3, 3), density=0.01, random_state=rng)

    x @ y  # Numba compilation

    @benchmark
    def test_matmul():
        x @ y
