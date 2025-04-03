import os
from typing import Any

import graphblas as gb
import graphblas_algorithms as ga

import numpy as np
import scipy.sparse as sps
from numpy.testing import assert_allclose

os.environ["SPARSE_BACKEND"] = "Finch"
import sparse

# select namespace
xp = sparse  # np jnp
Array = Any


def converged(xprev: Array, x: Array, N: int, tol: float) -> bool:
    err = xp.sum(xp.abs(x - xprev))
    return err < xp.asarray(N * tol)


class Graph:
    def __init__(self, A: Array):
        assert A.ndim == 2 and A.shape[0] == A.shape[1]
        self.N = A.shape[0]
        self.A = A


@sparse.compiled()
def kernel(hprev: Array, A: Array, N: int, tol: float) -> tuple[Array, Array, Array]:
    a = hprev.mT @ A
    h = A @ a.mT
    h = h / xp.max(h)
    conv = converged(hprev, h, N, tol)
    return h, a, conv


def hits_finch(G: Graph, max_iter: int = 100, tol: float = 1e-8, normalized: bool = True) -> tuple[Array, Array]:
    N = G.N
    if N == 0:
        return xp.asarray([]), xp.asarray([])

    h = xp.full((N, 1), 1.0 / N)
    A = xp.asarray(G.A)

    for _ in range(max_iter):
        hprev = h
        a = hprev.mT @ A
        h = A @ a.mT
        h = h / xp.max(h)
        if converged(hprev, h, N, tol):
            break
        # alternatively these lines can be compiled
        # h, a, conv = kernel(h, A, N, tol)
    else:
        raise Exception("Didn't converge")

    if normalized:
        h = h / xp.sum(xp.abs(h))
        a = a / xp.sum(xp.abs(a))
    return h, a


if __name__ == "__main__":
    coords = (np.array([0, 0, 1, 2, 2, 3]), np.array([1, 3, 0, 0, 1, 2]))
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    A = sps.coo_array((data, coords))
    G = Graph(A)

    h_finch, a_finch = hits_finch(G)

    print(h_finch, a_finch)

    M = gb.io.from_scipy_sparse(A)
    G = ga.Graph(M)
    h_gb, a_gb = ga.hits(G)

    assert_allclose(h_finch.todense().ravel(), h_gb.to_dense())
    assert_allclose(a_finch.todense().ravel(), a_gb.to_dense())
