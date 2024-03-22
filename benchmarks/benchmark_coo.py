# import os

# import sparse

# import numpy as np


# class ElemwiseSuite:
#     timeout = 120.0
#     warmup_time = 5.0
#     repeat = 5

#     def setup(self):
#         self.x = sparse.random((100, 100, 100), density=0.05, random_state=42)
#         self.y = sparse.random((100, 100, 100), density=0.05, random_state=42)

#         if os.environ[sparse._ENV_VAR_NAME] == "Finch":
#             import finch

#             storage = finch.Storage(finch.Dense(finch.SparseList(finch.SparseList(finch.Element(0.0)))))
#             self.x = self.x.to_device(storage)
#             self.y = self.y.to_device(storage)

#         self.x + self.y  # compilation
#         self.x * self.y  # compilation

#     def time_add(self):
#         self.x + self.y

#     def time_mul(self):
#         self.x * self.y


# class ElemwiseBroadcastingSuite:
#     timeout = 120.0
#     warmup_time = 5.0
#     repeat = 5

#     def setup(self):
#         self.x = sparse.random((1, 100, 100), density=0.01, random_state=42)
#         self.y = sparse.random((100, 100), density=0.01, random_state=42)

#         self.x + self.y  # compilation
#         self.x * self.y  # compilation

#     def time_add(self):
#         self.x + self.y

#     def time_mul(self):
#         self.x * self.y


# class IndexingSuite:
#     def setup(self):
#         rng = np.random.default_rng(0)
#         self.index = rng.integers(0, 100, 50)
#         self.x = sparse.random((100, 100, 100), density=0.01, random_state=rng)

#         # Numba compilation
#         self.x[5]
#         # self.x[self.index]

#     def time_index_scalar(self):
#         self.x[5, 5, 5]

#     def time_index_slice(self):
#         self.x[:50]

#     def time_index_slice2(self):
#         self.x[:50, :50]

#     def time_index_slice3(self):
#         self.x[:50, :50, :50]

#     # def time_index_fancy(self):
#     #     self.x[self.index]
