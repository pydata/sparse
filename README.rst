Sparse Multidimensional Arrays
==============================

This implements sparse multidimensional arrays on top of NumPy and
Scipy.sparse.  It generalizes the scipy.sparse.coo_matrix_ layout but extends
beyond just rows and columns to an arbitrary number of dimensions.

The original motivation is for maching learning algorithms, but it is general
use.

This Supports
--------------

-  Reshape
-  Transpose
-  Tensordot
-  Reductions (sum, max)
-  Slicing with integers, lists, and slices (with no step value)

This may yet support
--------------------

A "does not support" list is hard to build because it is infinitely long.
However the following things are in scope, relatively doable, and not yet built
(help welcome).

-  Arithmetic like add
-  NumPy ufunc support (where zero is maintained)
-  Concatenation and stacking
-  Smooth interaction with numpy arrays, scipy.sparse arrays, etc. for binary
   operations

Example
-------

.. code-block:: python

   In [1]: import numpy as np
   In [2]: n = 1000
   In [3]: ndims = 4
   In [4]: nnz = 1000000
   In [5]: coords = np.random.randint(0, n - 1, size=(nnz, ndims))
   In [6]: data = np.random.random(nnz)

   In [7]: import sparse
   In [8]: x = sparse.COO(coords, data, shape=((n,) * ndims))
   In [9]: x
   Out[9]: <COO: shape=(1000, 1000, 1000, 1000), dtype=float64, nnz=1000000>

   In [10]: x.nbytes
   Out[10]: 40000000

   In [11]: %time y = sparse.tensordot(x, x, axes=((3, 0), (1, 2)))
   CPU times: user 1.52 s, sys: 20 ms, total: 1.54 s
   Wall time: 1.54 s

   In [12]: y
   Out[12]: <COO: shape=(1000, 1000, 1000, 1000), dtype=float64, nnz=1001588>

   In [13]: %time z = y.sum(axis=(0, 1, 2))
   CPU times: user 408 ms, sys: 408 ms, total: 816 ms
   Wall time: 818 ms

   In [14]: z
   Out[14]: <COO: shape=(1000,), dtype=float64, nnz=999>

   In [15]: z.todense()
   Out[15]:
   array([ 244.0671803 ,  246.38455787,  243.43383158,  256.46068737,
           261.18598416,  256.36439011,  271.74177584,  238.56059193,
           ...


LICENSE
-------

This is licensed under New BSD-3

.. _scipy.sparse.coo_matrix: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
