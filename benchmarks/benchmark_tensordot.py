import os

import sparse

import numpy as np


# class TensordotSuiteDenseSparse:
#     """
#     Performance comparison for returntype=COO vs returntype=np.ndarray.
#     tensordot(np.ndarray, COO)
#     """

#     def setup(self):
#         rng = np.random.default_rng(0)
#         self.n = rng.random((100, 100))
#         self.s = sparse.random((100, 100, 100, 100), density=0.01, random_state=rng)

#     def time_dense(self):
#         sparse.tensordot(self.n, self.s, axes=([0, 1], [0, 2]))

#     def time_sparse(self):
#         sparse.tensordot(self.n, self.s, axes=([0, 1], [0, 2]), return_type=sparse.COO)


class TensordotSuiteSparseSparse:
    """
    Performance comparison for returntype=COO vs returntype=np.ndarray.
    tensordot(COO, COO)
    """
    timeout = 120.0

    def setup(self):

        random_kwargs = {"density": 0.01, "random_state": 42}
        if os.environ[sparse._ENV_VAR_NAME] == "PyData":
            random_kwargs["format"] = "gcxs"

        self.s1 = sparse.random((100, 100), **random_kwargs)
        self.s2 = sparse.random((100, 100, 100, 100), **random_kwargs)

        if os.environ[sparse._ENV_VAR_NAME] == "Finch":
            import finch
            self.s1 = self.s1.to_device(
                finch.Storage(finch.Dense(finch.SparseList(finch.Element(0.0))), order=self.s1.get_order())
            )
            self.s2 = self.s2.to_device(
                finch.Storage(
                    finch.Dense(finch.SparseList(finch.SparseList(finch.SparseList(finch.Element(0.0))))),
                    order=self.s2.get_order(),
                )
            )


    # def time_dense(self):
    #     sparse.tensordot(self.s1, self.s2, axes=([0, 1], [0, 2]), return_type=np.ndarray)

    def time_sparse(self):
        sparse.tensordot(self.s1, self.s2, axes=([0, 1], [0, 2]))


# class TensordotSuiteSparseDense:
#     """
#     Performance comparison for returntype=COO vs returntype=np.ndarray.
#     tensordot(COO, np.ndarray)
#     """

#     def setup(self):
#         rng = np.random.default_rng(0)
#         self.s = sparse.random((100, 100, 100, 100), density=0.01, random_state=rng)
#         self.n = rng.random((100, 100))

#     def time_dense(self):
#         sparse.tensordot(self.s, self.n, axes=([0, 1], [0, 1]))

#     def time_sparse(self):
#         sparse.tensordot(self.s, self.n, axes=([0, 1], [0, 1]), return_type=sparse.COO)
