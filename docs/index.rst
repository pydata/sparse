Sparse
======

.. image:: logo.png
   :alt: Logo
   :align: center
   :width: 20em

This implements sparse arrays of arbitrary dimension on top of :obj:`numpy` and :obj:`scipy.sparse`.
It generalizes the :obj:`scipy.sparse.coo_matrix` and :obj:`scipy.sparse.dok_matrix` layouts,
but extends beyond just rows and columns to an arbitrary number of dimensions.

Additionally, this project maintains compatibility with the :obj:`numpy.ndarray` interface
rather than the :obj:`numpy.matrix` interface used in :obj:`scipy.sparse`

These differences make this project useful in certain situations
where scipy.sparse matrices are not well suited,
but it should not be considered a full replacement.
It lacks layouts that are not easily generalized like CSR/CSC
and depends on scipy.sparse for some computations.


Motivation
----------

Sparse arrays, or arrays that are mostly empty or filled with zeros,
are common in many scientific applications.
To save space we often avoid storing these arrays in traditional dense formats,
and instead choose different data structures.
Our choice of data structure can significantly affect our storage and computational
costs when working with these arrays.


Design
------

The main data structure in this library follows the
`Coordinate List (COO) <https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)>`_
layout for sparse matrices, but extends it to multiple dimensions.

The COO layout, which stores the row index, column index, and value of every element:

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
slicing, tensordot, reductions, etc., which can be challenging in general.

Fortunately in many cases we can leverage the existing :obj:`scipy.sparse`
algorithms if we can intelligently transpose and reshape our multi-dimensional
array into an appropriate 2-d sparse matrix, perform a modified sparse matrix
operation, and then reshape and transpose back.  These reshape and transpose
operations can all be done at numpy speeds by modifying the arrays of
coordinates.  After scipy.sparse runs its operations (often written in C) then
we can convert back to using the same path of reshapings and transpositions in
reverse.

LICENSE
-------

This library is licensed under BSD-3

.. toctree::
   :maxdepth: 3
   :hidden:

   install
   quickstart
   construct
   operations
   generated/sparse
   roadmap
   contributing
   changelog

.. _scipy.sparse: https://docs.scipy.org/doc/scipy/reference/sparse.html
