import numpy as np

import sparse


class ElemwiseSuite(object):
    def setup(self):
        np.random.seed(0)
        self.x = sparse.random((100, 100, 100), density=0.01)
        self.y = sparse.random((100, 100, 100), density=0.01)

        self.x.sum_duplicates()
        self.y.sum_duplicates()

    def time_add(self):
        self.x + self.y

    def time_mul(self):
        self.x * self.y

    def time_index(self):
        for i in range(100):
            self.x[i]


class ElemwiseBroadcastingSuite(object):
    def setup(self):
        np.random.seed(0)
        self.x = sparse.random((100, 1, 100), density=0.01)
        self.y = sparse.random((100, 100), density=0.01)

        self.x.sum_duplicates()
        self.y.sum_duplicates()

    def time_add(self):
        self.x + self.y

    def time_mul(self):
        self.x * self.y
