# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: sparse
#     language: python
#     name: python3
# ---

# # Using with SciPy
# ## Import

# +
import sparse

import numpy as np
import scipy.sparse as sps

# -

# ## Create Arrays

rng = np.random.default_rng(42)
M = 1_000
DENSITY = 0.01
a = sparse.random((M, M), density=DENSITY, format="csc")
identity = sparse.eye(M, format="csc")

# ## Invert and verify matrix
# This showcases the `scipy.sparse.linalg` integration.

a_inv = sps.linalg.spsolve(a, identity)
np.testing.assert_array_almost_equal((a_inv @ a).todense(), identity.todense())

# ## Calculate the graph distances
# This showcases the `scipy.sparse.csgraph` integration.

sps.csgraph.bellman_ford(sparse.eye(5, k=1) + sparse.eye(5, k=-1), return_predecessors=False)
