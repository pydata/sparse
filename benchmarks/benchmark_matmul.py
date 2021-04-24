import numpy as np
import sparse


class Matmul_Sparse:
    params = (["coo", "gcxs"], [0.01, 0.33, 0.5, 1.0])

    def setup(self, p, dens_arg):
        np.random.seed(0)
        self.x = sparse.random((100, 100), density=0.01, format=p)
        self.y = sparse.random((100, 100), density=dens_arg, format=p)

        self.x @ self.y

    def time_matmul(self, p, dens_arg):
        self.x @ self.y
