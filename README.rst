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

LICENSE
-------

This is licensed under New BSD-3

.. _scipy.sparse.coo_matrix: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
