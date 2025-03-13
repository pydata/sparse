import importlib
import os

import sparse

import networkx as nx
from utils import benchmark

import numpy as np

ITERS = 3


if __name__ == "__main__":
    print("Counting Triangles Example:\n")

    G = nx.gnp_random_graph(n=200, p=0.2)

    # ======= Finch =======
    os.environ[sparse._ENV_VAR_NAME] = "Finch"
    importlib.reload(sparse)

    a_sps = nx.to_scipy_sparse_array(G)
    a = sparse.asarray(a_sps)

    @sparse.compiled()
    def count_triangles_finch(a):
        return sparse.sum(a @ a * a) / sparse.asarray(6)

    # Compile & Benchmark
    result_finch = benchmark(count_triangles_finch, args=[a], info="Finch", iters=ITERS)

    # ======= SciPy =======
    def count_triangles_scipy(a):
        return (a @ a * a).sum() / 6

    a = nx.to_scipy_sparse_array(G)

    # Compile & Benchmark
    result_scipy = benchmark(count_triangles_scipy, args=[a], info="SciPy", iters=ITERS)

    # ======= NetworkX =======
    def count_triangles_networkx(a):
        return sum(nx.triangles(a).values()) / 3

    a = G

    # Compile & Benchmark
    result_networkx = benchmark(count_triangles_networkx, args=[a], info="NetworkX", iters=ITERS)

    np.testing.assert_equal(result_finch.todense(), result_scipy)
    np.testing.assert_equal(result_finch.todense(), result_networkx)
    assert result_networkx == result_scipy
