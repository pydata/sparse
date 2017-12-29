Sparse Multidimensional Arrays
==============================

|Build Status|

This implements sparse multidimensional arrays on top of NumPy and
Scipy.sparse.  It generalizes the scipy.sparse.coo_matrix_ layout but extends
beyond just rows and columns to an arbitrary number of dimensions.

The original motivation is for machine learning algorithms, but it is
intended for somewhat general use.

This Supports
--------------

-  NumPy ufuncs (where zeros are preserved)
-  Binary operations with other :code:`COO` objects, where zeros are preserved.
-  Binary operations with Scipy sparse matrices, where zeros are preserved.
-  Binary operations with scalars, where zeros are preserved.
-  Broadcasting binary operations and :code:`broadcast_to`.
-  Reductions (sum, max, min, prod, ...)
-  Reshape
-  Transpose
-  Tensordot
-  triu, tril
-  Slicing with integers, lists, and slices (with no step value)
-  Concatenation and stacking

This may yet support
--------------------

A "does not support" list is hard to build because it is infinitely long.
However the following things are in scope, relatively doable, and not yet built
(help welcome).

-  Incremental buliding of arrays and inplace updates
-  More operations supported by Numpy :code:`ndarray`s, such as :code:`argmin` and :code:`argmax`.
-  Array building functions such as :code:`eye`, :code:`spdiags`. See `building sparse matrices`_.
-  Linear algebra operations such as :code:`inv`, :code:`norm` and :code:`solve`. See scipy.sparse.linalg_.

There are no plans to support
-----------------------------

-  Parallel computing (though Dask.array may use this in the future)

Example
-------

::

   pip install sparse

.. code-block:: python

   import numpy as np
   n = 1000
   ndims = 4
   nnz = 1000000
   coords = np.random.randint(0, n - 1, size=(ndims, nnz))
   data = np.random.random(nnz)

   import sparse
   x = sparse.COO(coords, data, shape=((n,) * ndims))
   x
   # <COO: shape=(1000, 1000, 1000, 1000), dtype=float64, nnz=1000000>

   x.nbytes
   # 16000000

   y = sparse.tensordot(x, x, axes=((3, 0), (1, 2)))

   y
   # <COO: shape=(1000, 1000, 1000, 1000), dtype=float64, nnz=1001588>

   z = y.sum(axis=(0, 1, 2))
   z
   # <COO: shape=(1000,), dtype=float64, nnz=999>

   z.todense()
   # array([ 244.0671803 ,  246.38455787,  243.43383158,  256.46068737,
   #         261.18598416,  256.36439011,  271.74177584,  238.56059193,
   #         ...


How does this work?
-------------------

Scipy.sparse implements decent 2-d sparse matrix objects for the standard
layouts, notably for our purposes
`CSR, CSC, and COO <https://en.wikipedia.org/wiki/Sparse_matrix>`_.  However it
doesn't include support for sparse arrays of greater than 2 dimensions.

This library extends the COO layout, which stores the row index, column index,
and value of every element:

=== === ====
row col data
=== === ====
  0   0   10
  0   2   13
  1   3    9
  3   8   21
=== === ====

It is straightforward to extend the COO layout to an arbitrary number of
dimensions:

==== ==== ==== === ====
dim1 dim2 dim3 ... data
==== ==== ==== === ====
  0    0     0   .   10
  0    0     3   .   13
  0    2     2   .    9
  3    1     4   .   21
==== ==== ==== === ====

This makes it easy to *store* a multidimensional sparse array, but we still
need to reimplement all of the array operations like transpose, reshape,
slicing, tensordot, reductions, etc., which can be quite challenging in
general.

Fortunately in many cases we can leverage the existing SciPy.sparse algorithms
if we can intelligently transpose and reshape our multi-dimensional array into
an appropriate 2-d sparse matrix, perform a modified sparse matrix
operation, and then reshape and transpose back.  These reshape and transpose
operations can all be done at numpy speeds by modifying the arrays of
coordinates.  After scipy.sparse runs its operations (coded in C) then we can
convert back to using the same path of reshapings and transpositions in
reverse.

This approach is not novel; it has been around in the multidimensional array
community for a while.  It is also how some operations in numpy work.  For example
the ``numpy.tensordot`` function performs transposes and reshapes so that it can
use the ``numpy.dot`` function for matrix multiplication which is backed by
fast BLAS implementations.  The ``sparse.tensordot`` code is very slight
modification of ``numpy.tensordot``, replacing ``numpy.dot`` with
``scipy.sprarse.csr_matrix.dot``.


LICENSE
-------

This is licensed under New BSD-3

.. _scipy.sparse.coo_matrix: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
.. _building sparse matrices: https://docs.scipy.org/doc/scipy/reference/sparse.html#functions
.. _scipy.sparse.linalg: https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
.. |Build Status| image:: https://travis-ci.org/mrocklin/sparse.svg?branch=master
   :target: https://travis-ci.org/mrocklin/sparse
