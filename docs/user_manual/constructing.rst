.. currentmodule:: sparse

Constructing :obj:`COO` arrays
==============================

From coordinates and data
-------------------------
This is the preferred way of constructing :obj:`COO` arrays. The
constructor for :obj:`COO` (see :obj:`COO.__init__`) can create these
objects from two main variables: :code:`coords` and :code:`data`.

:code:`coords` contains the indices where the data is nonzero, and
:code:`data` contains the data corresponding to those indices. For
example, the following code will generate a :math:`5 \times 5`
identity matrix:

.. code-block:: python

   coords = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
   data = [1, 1, 1, 1, 1]
   s = COO(coords, data)

In general :code:`coords` should be a :code:`(ndim, nnz)` shaped
array. Each row of :code:`coords` contains one dimension of the
desired sparse array, and each column contains the index
corresponding to that nonzero element. :code:`data` contains
the nonzero elements of the array corresponding to the indices
in :code:`coords`. Its shape should be :code:`(nnz,)`

You can, and should, pass in :obj:`numpy.ndarray` objects for
:code:`coords` and :code:`data`.

In this case, the shape of the resulting array was determined from
the maximum index in each dimension. If the array extends beyond
the maximum index in :code:`coords`, you should supply a shape
explicitly. For example, if we did the following without the
:code:`shape` keyword argument, it would result in a
:math:`4 \times 5` matrix, but maybe we wanted one that was actually
:math:`5 \times 5`.

.. code-block:: python

   coords = [[0, 3, 2, 1], [4, 1, 2, 0]]
   data = [1, 4, 2, 1]
   s = COO(coords, data, shape=(5, 5))

From :obj:`scipy.sparse.spmatrix` objects
-----------------------------------------
To construct :obj:`COO` array from :obj:`scipy.sparse.spmatrix`
objects, you can use the :obj:`COO.from_scipy_sparse` method. As an
example, if :code:`x` is a :obj:`scipy.sparse.spmatrix`, you can
do the following to get an equivalent :obj:`COO` array:

.. code-block:: python

   s = COO.from_scipy_sparse(x)

From :obj:`numpy.ndarray` objects
---------------------------------
To construct :obj:`COO` arrays from :obj:`numpy.ndarray`
objects, you can use the :obj:`COO.from_numpy` method. As an
example, if :code:`x` is a :obj:`numpy.ndarray`, you can
do the following to get an equivalent :obj:`COO` array:

.. code-block:: python

   s = COO.from_numpy(x)

Generating random :obj:`COO` objects
------------------------------------
The :obj:`sparse.random` method can be used to create random
:obj:`COO` arrays. For example, the following will generate
a :math:`10 \times 10` matrix with :math:`10` nonzero entries,
each in the interval :math:`[0, 1)`.

.. code-block:: python

   s = sparse.random((10, 10), density=0.1)
