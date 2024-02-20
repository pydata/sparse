import sparse

import numpy as np


class MatrixMultiplySuite:
    def setup(self):
        rng = np.random.default_rng(0)
        self.x = sparse.random((100, 100), density=0.01, format="gcxs", random_state=rng)
        self.y = sparse.random((100, 100), density=0.01, format="gcxs", random_state=rng)

        self.x @ self.y  # Numba compilation

    def time_matmul(self):
        self.x @ self.y


class ElemwiseSuite:
    def setup(self):
        rng = np.random.default_rng(0)
        self.x = sparse.random((100, 100, 100), density=0.01, format="gcxs", random_state=rng)
        self.y = sparse.random((100, 100, 100), density=0.01, format="gcxs", random_state=rng)

        self.x + self.y  # Numba compilation

    def time_add(self):
        self.x + self.y

    def time_mul(self):
        self.x * self.y


class ElemwiseBroadcastingSuite:
    def setup(self):
        rng = np.random.default_rng(0)
        self.x = sparse.random((100, 1, 100), density=0.01, format="gcxs", random_state=rng)
        self.y = sparse.random((100, 100), density=0.01, format="gcxs", random_state=rng)

    def time_add(self):
        self.x + self.y

    def time_mul(self):
        self.x * self.y


class IndexingSuite:
    def setup(self):
        rng = np.random.default_rng(0)
        self.index = rng.integers(0, 100, 50)
        self.x = sparse.random((100, 100, 100), density=0.01, format="gcxs", random_state=rng)

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


class DenseMultiplySuite:
    params = ([0, 1], [1, 20, 100])
    param_names = ["compressed axis", "n_vectors"]

    def setup(self, compressed_axis, n_vecs):
        rng = np.random.default_rng(1337)
        n = 10000
        x = sparse.random((n, n), density=0.001, format="gcxs", random_state=rng).change_compressed_axes(
            (compressed_axis,)
        )
        self.x = x
        self.t = rng.random((n, n_vecs))
        self.u = rng.random((n_vecs, n))

        # Numba compilation
        self.x @ self.t
        self.u @ self.x

    def time_gcxs_dot_ndarray(self, *args):
        self.x @ self.t

    def time_ndarray_dot_gcxs(self, *args):
        self.u @ self.x
