import importlib
import os
import time

import sparse

import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_scipy

import numpy as np
import scipy.sparse as sp


def pagerank(G, alpha=0.85, max_iter=100, tol=1e-6) -> dict:
    N = len(G)
    if N == 0:
        return {}

    alpha = sparse.asarray(alpha)
    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, dtype=float, format="csc")
    A = sparse.asarray(A)
    S = sparse.sum(A, axis=1)
    S = sparse.where(sparse.asarray(0.0) != S, sparse.asarray(1.0) / S, S)

    # TODO: spdiags https://github.com/willow-ahrens/Finch.jl/issues/499
    Q = sparse.asarray(sp.csc_array(sp.spdiags(S.todense(), 0, *A.shape)))
    A = Q @ A

    # initial vector
    x = sparse.full((1, N), fill_value=1.0 / N)

    # personalization vector
    p = sparse.full((1, N), fill_value=1.0 / N)

    # Dangling nodes
    dangling_weights = p

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x_dangling = sparse.where(S[None, :] == sparse.asarray(0.0), x, sparse.asarray(0.0))
        x = (
            alpha * (x @ A + sparse.asarray(sparse.sum(x_dangling)) * dangling_weights)
            + (sparse.asarray(1) - alpha) * p
        )
        # check convergence, l1 norm
        err = sparse.sum(sparse.abs(x - xlast))
        if err < N * tol:
            return dict(zip(nodelist, map(float, x[0, :]), strict=False))

    raise nx.PowerIterationFailedConvergence(max_iter)


if __name__ == "__main__":
    G = nx.DiGraph(nx.path_graph(4))
    ITERS = 3

    os.environ[sparse._ENV_VAR_NAME] = "Finch"
    importlib.reload(sparse)

    # compile
    pagerank(G)
    print("compiled")

    # finch
    start = time.time()
    for i in range(ITERS):
        print(f"finch iter: {i}")
        pr = pagerank(G)
    elapsed = time.time() - start
    print(f"Finch took {elapsed / ITERS} s.")

    # scipy
    start = time.time()
    for i in range(ITERS):
        print(f"scipy iter: {i}")
        scipy_pr = _pagerank_scipy(G)
    elapsed = time.time() - start
    print(f"SciPy took {elapsed / ITERS} s.")

    np.testing.assert_almost_equal(list(pr.values()), list(scipy_pr.values()))
    print(f"finch: {pr}")
    print(f"scipy: {scipy_pr}")
