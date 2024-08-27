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

# # Multiple Formats with Finch
# ## Import
# Let's set the backend and import `sparse`.

# +
import os

os.environ["SPARSE_BACKEND"] = "Finch"

import sparse

import numpy as np

# -


# ## Perform Operations
# Let's create two arrays.

rng = np.random.default_rng(42)  # Seed for reproducibility
a = sparse.random((3, 3), density=1 / 6, random_state=rng)
b = sparse.random((3, 3), density=1 / 6, random_state=rng)

# Now let's matrix multiply them.

c = a @ b

# And view the result as a (dense) NumPy array.

c_dense = c.todense()

# Now let's do the same for other formats, and compare the results.

for format in ["coo", "csr", "csc", "dense"]:
    af = sparse.asarray(a, format=format)
    bf = sparse.asarray(b, format=format)
    cf = af @ bf
    np.testing.assert_array_equal(c_dense, cf.todense())
