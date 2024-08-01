# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: sparse
#     language: python
#     name: python3
# ---

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

c.todense()
