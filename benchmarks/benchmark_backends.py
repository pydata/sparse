import sparse

import numpy as np

from .utils import SkipNotImplemented

TIMEOUT: float = 300.0
BACKEND: sparse.BackendType = sparse.backend_var.get()


class Tensordot:
    timeout = TIMEOUT

    def setup(self):
        rng = np.random.default_rng(0)

        random_kwargs = {"density": 0.01, "random_state": rng}
        if sparse.BackendType.Numba == BACKEND:
            random_kwargs["format"] = "gcxs"

        self.s1 = sparse.random((100, 10), **random_kwargs)
        self.s2 = sparse.random((100, 100, 10), **random_kwargs)

        if sparse.BackendType.Finch == BACKEND:
            import finch

            self.s1 = self.s1.to_device(
                finch.Storage(finch.Dense(finch.SparseList(finch.Element(0.0))), order=self.s1.get_order())
            )
            self.s2 = self.s2.to_device(
                finch.Storage(
                    finch.Dense(finch.SparseList(finch.SparseList(finch.Element(0.0)))),
                    order=self.s2.get_order(),
                )
            )

        sparse.tensordot(self.s1, self.s2, axes=([0, 1], [0, 2]))  # compilation

    def time_tensordot(self):
        sparse.tensordot(self.s1, self.s2, axes=([0, 1], [0, 2]))


class SpMv:
    timeout = TIMEOUT
    # NOTE: https://github.com/willow-ahrens/Finch.jl/issues/488
    params = [[True, False], [(10, 0.01)]]  # (1000, 0.01), (1_000_000, 1e-05)
    param_names = ["lazy_mode", "size_and_density"]

    def setup(self, lazy_mode, size_and_density):
        rng = np.random.default_rng(0)
        size, density = size_and_density

        random_kwargs = {"density": density, "random_state": rng}
        if sparse.BackendType.Numba == BACKEND:
            random_kwargs["format"] = "gcxs"

        self.M = sparse.random((size, size), **random_kwargs)
        # NOTE: Once https://github.com/willow-ahrens/Finch.jl/issues/487 is fixed change to (size, 1).
        self.v1 = rng.normal(size=(size, 2))
        self.v2 = rng.normal(size=(size, 2))

        if sparse.BackendType.Finch == BACKEND:
            import finch

            self.M = self.M.to_device(
                finch.Storage(finch.Dense(finch.SparseList(finch.Element(0.0))), order=self.M.get_order())
            )
            self.v1 = finch.Tensor(self.v1)
            self.v2 = finch.Tensor(self.v2)
            if lazy_mode:

                @sparse.compiled
                def fn(tns1, tns2, tns3):
                    return tns1 @ tns2 + tns3
            else:

                def fn(tns1, tns2, tns3):
                    return tns1 @ tns2 + tns3

        elif sparse.BackendType.Numba == BACKEND:
            if lazy_mode:
                raise SkipNotImplemented("Numba doesn't have lazy mode")

            def fn(tns1, tns2, tns3):
                return tns1 @ tns2 + tns3

        else:
            raise Exception(f"Invalid backend: {BACKEND}")

        self.fn = fn
        self.fn(self.M, self.v1, self.v2)

    def time_spmv(self, lazy_mode, size_and_density):
        self.fn(self.M, self.v1, self.v2)


class Elemwise:
    timeout = TIMEOUT

    def setup(self):
        rng = np.random.default_rng(0)

        random_kwargs = {"density": 0.01, "random_state": rng}
        if sparse.BackendType.Numba == BACKEND:
            random_kwargs["format"] = "gcxs"

        self.s1 = sparse.random((100, 10), **random_kwargs)
        self.s2 = sparse.random((100, 100, 10), **random_kwargs)

        if sparse.BackendType.Finch == BACKEND:
            import finch

            self.s1 = self.s1.to_device(
                finch.Storage(finch.Dense(finch.SparseList(finch.Element(0.0))), order=self.s1.get_order())
            )
            self.s2 = self.s2.to_device(
                finch.Storage(
                    finch.Dense(finch.SparseList(finch.SparseList(finch.Element(0.0)))), order=self.s2.get_order()
                )
            )

        self.s1 + self.s2
        self.s1 * self.s2

    def time_add(self):
        self.s1 + self.s2

    def time_mul(self):
        self.s1 * self.s2


# class SDDMM:
#     timeout = TIMEOUT

#     def setup():
#         pass

# class Reductions:
#     timeout = TIMEOUT

#     def setup():
#         pass
