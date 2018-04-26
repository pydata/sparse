import numpy as np

import sparse


class ElemwiseSuite(object):
    def setup(self):
        np.random.seed(0)
        self.x = sparse.random((100, 100, 100), density=0.01)
        self.y = sparse.random((100, 100, 100), density=0.01)

        self.x + self.y  # Numba compilation

    def time_add(self):
        self.x + self.y

    def time_mul(self):
        self.x * self.y

    def time_index(self):
        self.x[5]


class ElemwiseBroadcastingSuite(object):
    def setup(self):
        np.random.seed(0)
        self.x = sparse.random((100, 1, 100), density=0.01)
        self.y = sparse.random((100, 100), density=0.01)

    def time_add(self):
        self.x + self.y

    def time_mul(self):
        self.x * self.y


class IndexingSuite(object):
    def setup(self):
        np.random.seed(0)
        self.x = sparse.random((100, 100, 100), density=0.01)
        self.x[5]  # Numba compilation

    def time_index_scalar(self):
        self.x[5]

    def time_index_slice(self):
        self.x[:50]

    def time_index_slice2(self):
        self.x[:50, :50]

    def time_index_slice3(self):
        self.x[:50, :50, :50]
