.. currentmodule:: sparse

Construct Sparse Arrays
=======================

From coordinates and data
-------------------------
You can construct :obj:`COO` arrays from coordinates and value data.

The :code:`coords` parameter contains the indices where the data is nonzero,
and the :code:`data` parameter contains the data corresponding to those indices.
For example, the following code will generate a :math:`5 \times 5` diagonal
matrix:

.. code-block:: python

   >>> import sparse

   >>> coords = [[0, 1, 2, 3, 4],
   ...           [0, 1, 2, 3, 4]]
   >>> data = [10, 20, 30, 40, 50]
   >>> s = sparse.COO(coords, data, shape=(5, 5))

   >>> s.todense()
   array([[10,  0,  0,  0,  0],
          [ 0, 20,  0,  0,  0],
          [ 0,  0, 30,  0,  0],
          [ 0,  0,  0, 40,  0],
          [ 0,  0,  0,  0, 50]])

In general :code:`coords` should be a :code:`(ndim, nnz)` shaped
array. Each row of :code:`coords` contains one dimension of the
desired sparse array, and each column contains the index
corresponding to that nonzero element. :code:`data` contains
the nonzero elements of the array corresponding to the indices
in :code:`coords`. Its shape should be :code:`(nnz,)`.

If ``data`` is the same across all the coordinates, it can be passed
in as a scalar. For example, the following produces the :math:`4 \times 4`
identity matrix:

.. code-block:: python

   >>> import sparse

   >>> coords = [[0, 1, 2, 3],
   ...           [0, 1, 2, 3]]
   >>> data = 1
   >>> s = sparse.COO(coords, data, shape=(4, 4))

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

:obj:`COO` arrays support arbitrary fill values. Fill values are the "default"
value, or value to not store. This can be given a value other than zero. For
example, the following builds a (bad) representation of a :math:`2 \times 2`
identity matrix. Note that not all operations are supported for operations
with nonzero fill values.

.. code-block:: python

   coords = [[0, 1], [1, 0]]
   data = [0, 0]
   s = COO(coords, data, fill_value=1)

From :doc:`Scipy sparse matrices <reference/generated/scipy.sparse.spmatrix>`
-----------------------------------------------------------------------------
To construct :obj:`COO` array from :obj:`spmatrix <scipy.sparse.spmatrix>`
objects, you can use the :obj:`COO.from_scipy_sparse` method. As an
example, if :code:`x` is a :obj:`scipy.sparse.spmatrix`, you can
do the following to get an equivalent :obj:`COO` array:

.. code-block:: python

   s = COO.from_scipy_sparse(x)

From :doc:`Numpy arrays <reference/generated/numpy.ndarray>`
------------------------------------------------------------
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

Building :obj:`COO` Arrays from :obj:`DOK` Arrays
-------------------------------------------------
It's possible to build :obj:`COO` arrays from :obj:`DOK` arrays, if it is not
easy to construct the :code:`coords` and :obj:`data` in a simple way. :obj:`DOK`
arrays provide a simple builder interface to build :obj:`COO` arrays, but at
this time, they can do little else.

You can get started by defining the shape (and optionally, datatype) of the
:obj:`DOK` array. If you do not specify a dtype, it is inferred from the value
dictionary or is set to :code:`dtype('float64')` if that is not present.

.. code-block:: python

   s = DOK((6, 5, 2))
   s2 = DOK((2, 3, 4), dtype=np.uint8)

After this, you can build the array by assigning arrays or scalars to elements
or slices of the original array. Broadcasting rules are followed.

.. code-block:: python

   s[1:3, 3:1:-1] = [[6, 5]]

DOK arrays also support fancy indexing assignment if and only if all dimensions are indexed.

.. code-block:: python

   s[[0, 2], [2, 1], [0, 1]] = 5
   s[[0, 3], [0, 4], [0, 1]] = [1, 5]

Alongside indexing assignment and retrieval, :obj:`DOK` arrays support any arbitrary broadcasting function
to any number of arguments where the arguments can be :obj:`SparseArray` objects, :obj:`scipy.sparse.spmatrix`
objects, or :obj:`numpy.ndarrays`. 

.. code-block:: python

   x = sparse.random((10, 10), 0.5, format="dok")
   y = sparse.random((10, 10), 0.5, format="dok")
   sparse.elemwise(np.add, x, y)

:obj:`DOK` arrays also support standard ufuncs and operators, including comparison operators,
in combination with other objects implementing the `numpy` `ndarray.__array_ufunc__` method. For example,
the following code will perform elementwise equality comparison on the two arrays
and return a new boolean :obj:`DOK` array.

.. code-block:: python

   x = sparse.random((10, 10), 0.5, format="dok")
   y = np.random.random((10, 10))
   x == y

:obj:`DOK` arrays are returned from elemwise functions and standard ufuncs if and only if all 
:obj:`SparseArray` objects are obj:`DOK` arrays. Otherwise, a :obj:`COO` array or dense array are returned.

At the end, you can convert the :obj:`DOK` array to a :obj:`COO` arrays.

.. code-block:: python

   s3 = COO(s)

In addition, it is possible to access single elements and slices of the :obj:`DOK` array
using normal Numpy indexing, as well as fancy indexing if and only if all dimensions are indexed.
Slicing and fancy indexing will always return a new DOK array.

.. code-block:: python

   s[1, 2, 1]  # 5
   s[5, 1, 1]  # 0
   s[[0, 3], [0, 4], [0, 1]] # <DOK: shape=(2,), dtype=float64, nnz=2, fill_value=0.0>

.. _converting:

Converting :obj:`COO` objects to other Formats
----------------------------------------------
:obj:`COO` arrays can be converted to :doc:`Numpy arrays <reference/generated/numpy.ndarray>`,
or to some :obj:`spmatrix <scipy.sparse.spmatrix>` subclasses via the following
methods:

* :obj:`COO.todense`: Converts to a :obj:`numpy.ndarray` unconditionally.
* :obj:`COO.maybe_densify`: Converts to a :obj:`numpy.ndarray` based on
   certain constraints.
* :obj:`COO.to_scipy_sparse`: Converts to a :obj:`scipy.sparse.coo_matrix` if
   the array is two dimensional.
* :obj:`COO.tocsr`: Converts to a :obj:`scipy.sparse.csr_matrix` if
   the array is two dimensional.
* :obj:`COO.tocsc`: Converts to a :obj:`scipy.sparse.csc_matrix` if
   the array is two dimensional.
