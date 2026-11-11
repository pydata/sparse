import sparse

import pytest

import numpy as np


@pytest.mark.parametrize("columns", [1, 2, 4, 8, 64, 256])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("density", [0.001, 0.01])
def test_coo_dense_matmul(benchmark, rng, columns, order, density):
    a = sparse.random((1000, 1000), density=density, random_state=rng)
    b = np.array(rng.random((1000, columns)), order=order)
    a @ b  # Compile before measuring.

    @benchmark
    def bench():
        a @ b
