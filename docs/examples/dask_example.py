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

# # Using with Dask
# ## Import

# +
import sparse

import dask.array as da

import numpy as np

# -

# ## Create Arrays
#
# Here, we create two random sparse arrays and move them to Dask.

# +
rng = np.random.default_rng(42)
M, N = 10_000, 10_000
DENSITY = 0.0001
a = sparse.random((M, N), density=DENSITY)
b = sparse.random((M, N), density=DENSITY)

a_dask = da.from_array(a, chunks=1000)
b_dask = da.from_array(b, chunks=1000)
# -

# As we can see in the "data type" section, each chunk of the Dask array is still sparse.

a_dask  # noqa: B018

# # Compute and check results
# As we can see, what we get out of Dask matches what we get out of `sparse`.

assert sparse.all(a + b == (a_dask + b_dask).compute())
