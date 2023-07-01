import numpy as np

import sparse


class TensordotSuiteDenseSparse:
    """
    Performance comparison for returntype=COO vs returntype=np.ndarray.
    tensordot(np.ndarray, COO)
    """

    def setup(self):
        np.random.seed(0)
        self.n = np.random.random((100, 100))
        self.s = sparse.random((100, 100, 100, 100), density=0.01)

    def time_dense(self):
        sparse.tensordot(self.n, self.s, axes=([0, 1], [0, 2]))

    def time_sparse(self):
        sparse.tensordot(self.n, self.s, axes=([0, 1], [0, 2]), return_type=sparse.COO)


class TensordotSuiteSparseSparse:
    """
    Performance comparison for returntype=COO vs returntype=np.ndarray.
    tensordot(COO, COO)
    """

    def setup(self):
        np.random.seed(0)
        self.s1 = sparse.random((100, 100), density=0.01)
        self.s2 = sparse.random((100, 100, 100, 100), density=0.01)

    def time_dense(self):
        sparse.tensordot(
            self.s1,
            self.s2,
            axes=([0, 1], [0, 2]),
            return_type=np.ndarray,
        )

    def time_sparse(self):
        sparse.tensordot(self.s1, self.s2, axes=([0, 1], [0, 2]))


class TensordotSuiteSparseDense:
    """
    Performance comparison for returntype=COO vs returntype=np.ndarray.
    tensordot(COO, np.ndarray)
    """

    def setup(self):
        np.random.seed(0)
        self.s = sparse.random((100, 100, 100, 100), density=0.01)
        self.n = np.random.random((100, 100))

    def time_dense(self):
        sparse.tensordot(self.s, self.n, axes=([0, 1], [0, 1]))

    def time_sparse(self):
        sparse.tensordot(self.s, self.n, axes=([0, 1], [0, 1]), return_type=sparse.COO)
