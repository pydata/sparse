import sparse

import numpy as np


class Matmul_Sparse:
    params = (["coo", "gcxs"], [0, 1, None])

    def setup(self, p, dens_arg):
        rng = np.random.default_rng(0)
        self.x = sparse.random((100, 100), density=0.01, format=p, random_state=rng)
        self.y = sparse.random((100, 100), density=0.01, format=p, random_state=rng)

        if dens_arg == 0:
            self.x = self.x.todense()
        elif dens_arg == 1:
            self.y = self.y.todense()

        self.x @ self.y

    def time_matmul(self, p, dens_arg):
        self.x @ self.y
