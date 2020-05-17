import numpy as np
import sparse


class TensordotSuite:
    """
    Performance comparison for returntype="sparse".
    """

    def setup(self):
        np.random.seed(0)
        self.n = np.random.random((100, 100))
        self.s = sparse.random((100, 100, 100, 100), density=0.01)

    def time_dense(self):
        sparse.tensordot(self.n, self.s, axes=([0, 1], [0, 2]))

    def time_sparse(self):
        sparse.tensordot(self.n, self.s, axes=([0, 1], [0, 2]), return_type=sparse.COO)
