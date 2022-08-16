Sparse
======

.. raw:: html
	:file: logo.svg

This implements sparse arrays of arbitrary dimension on top of :obj:`numpy` and :obj:`scipy.sparse`.
It generalizes the :obj:`scipy.sparse.coo_matrix` and :obj:`scipy.sparse.dok_matrix` layouts,
but extends beyond just rows and columns to an arbitrary number of dimensions.

Additionally, this project maintains compatibility with the :obj:`numpy.ndarray` interface
rather than the :obj:`numpy.matrix` interface used in :obj:`scipy.sparse`

These differences make this project useful in certain situations
where scipy.sparse matrices are not well suited,
but it should not be considered a full replacement.
The data structures in pydata/sparse complement and can
be used in conjunction with the fast linear algebra routines
inside scipy.sparse. A format conversion or copy may be required.


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

This library also includes several other data structures. Similar to COO,
the `Dictionary of Keys (DOK) <https://en.wikipedia.org/wiki/Sparse_matrix#Dictionary_of_keys_(DOK)>`_
format for sparse matrices generalizes well to an arbitrary number of dimensions.
DOK is well-suited for writing and mutating. Most other operations are not supported for DOK.
A common workflow may involve writing an array with DOK and then converting to another
format for other operations.

The `Compressed Sparse Row/Column (CSR/CSC) <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)>`_
formats are widely used in scientific computing are now supported
by pydata/sparse. The CSR/CSC formats excel at compression and mathematical operations.
While these formats are restricted to two dimensions, pydata/sparse supports
the GCXS sparse array format, based on
`GCRS/GCCS from <https://ieeexplore.ieee.org/abstract/document/7237032/similar#similar>`_
which generalizes CSR/CSC to n-dimensional arrays.
Like their two-dimensional CSR/CSC counterparts, GCXS arrays compress well.
Whereas the storage cost of COO depends heavily on the number of dimensions of the array,
the number of dimensions only minimally affects the storage cost of GCXS arrays,
which results in favorable compression ratios across many use cases.

Together these formats cover a wide array of applications of sparsity.
Additionally, with each format complying with the :obj:`numpy.ndarray` interface and
following the appropriate dispatching protocols,
pydata/sparse arrays can interact with other array libraries and seamlessly
take part in pydata-ecosystem-based workflows.

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
   conduct

.. _scipy.sparse: https://docs.scipy.org/doc/scipy/reference/sparse.html
