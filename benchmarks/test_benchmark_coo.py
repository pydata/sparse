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


class ElemwiseSuite:
    def setup(self):
        rng = np.random.default_rng(0)
        self.x = sparse.random((100, 100, 100), density=0.01, random_state=rng)
        self.y = sparse.random((100, 100, 100), density=0.01, random_state=rng)

        self.x + self.y  # Numba compilation
        self.x * self.y  # Numba compilation

    def time_add(self):
        self.x + self.y

    def time_mul(self):
        self.x * self.y


class ElemwiseBroadcastingSuite:
    def setup(self):
        rng = np.random.default_rng(0)
        self.x = sparse.random((100, 1, 100), density=0.01, random_state=rng)
        self.y = sparse.random((100, 100), density=0.01, random_state=rng)

        self.x + self.y  # Numba compilation
        self.x * self.y  # Numba compilation

    def time_add(self):
        self.x + self.y

    def time_mul(self):
        self.x * self.y


class IndexingSuite:
    def setup(self):
        rng = np.random.default_rng(0)
        self.index = rng.integers(0, 100, 50)
        self.x = sparse.random((100, 100, 100), density=0.01, random_state=rng)

        # Numba compilation
        self.x[5]
        self.x[self.index]

    def time_index_scalar(self):
        self.x[5, 5, 5]

    def time_index_slice(self):
        self.x[:50]

    def time_index_slice2(self):
        self.x[:50, :50]

    def time_index_slice3(self):
        self.x[:50, :50, :50]

    def time_index_fancy(self):
        self.x[self.index]
