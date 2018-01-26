Operations on :obj:`COO` arrays
===============================

Operators
---------

:obj:`COO` objects support a number of operations. They interact with scalars,
:obj:`numpy.ndarray` objects, other :obj:`COO` objects, and
:obj:`scipy.sparse.spmatrix` objects, all following standard Python and Numpy
conventions.

For example, the following Numpy expression produces equivalent
results for both Numpy arrays, COO arrays, or a mix of the two:

.. code-block:: python

   np.log(X.dot(beta.T) + 1)

However some operations are not supported, like inplace operation,
operations that implicitly cause dense structures,
or numpy functions that are not yet implemented for sprase arrays

.. code-block:: python

   x += y     # inplace operations not supported
   x + 1      # operations that produce dense results not supported
   np.svd(x)  # sparse svd not implemented


This page describes those valid operations, and their limitations.

.. _auto-densification:


Element-wise Operations
-----------------------
:obj:`COO` arrays support a variety of element-wise operations. However, as
with operators, operations that map zero to a nonzero value are not supported.

To illustrate, the following are all possible, and will produce another
:obj:`COO` array:

.. code-block:: python

   np.abs(x)
   np.sin(x)
   np.sqrt(x)
   np.conj(x)
   np.expm1(x)
   np.log1p(x)

However, the following are all unsupported and will raise a :obj:`ValueError`:

.. code-block:: python

   np.exp(x)
   np.cos(x)
   np.log(x)

Notice that you can apply any unary or binary :obj:`numpy.ufunc` to :obj:`COO`
arrays, and :obj:`numpy.ndarray` objects and scalars and it will work so
long as the result is not dense. When applying to :obj:`numpy.ndarray` objects,
we check that operating on the array with zero would always produce a zero.

:obj:`COO.elemwise`
~~~~~~~~~~~~~~~~~~~
This function allows you to apply any arbitrary unary or binary function where
the first object is :obj:`COO`, and the second is a scalar, :obj:`COO`, or
a :obj:`numpy.ndarray`. For example, the following will add two
:obj:`COO` objects:

.. code-block:: python

   x.elemwise(np.add, y)


Auto-Densification
~~~~~~~~~~~~~~~~~~
Operations that would result in dense matrices, such as binary
operations with :obj:`numpy.ndarray` objects or certain operations with
scalars are not allowed and will raise a :obj:`ValueError`. For example,
all of the following will raise a :obj:`ValueError`. Here, :code:`x` and
:code:`y` are :obj:`COO` objects.

.. code-block:: python

   x == y
   x + 5
   x == 0
   x != 5
   x / y

However, all of the following are valid operations.

.. code-block:: python

   x + 0
   x != y
   x + y
   x == 5
   5 * x
   x / 7.3
   x != 0

If densification is needed, it must be explicit. In other words, you must call
:obj:`COO.todense` on the :obj:`COO` object. If both operands are :obj:`COO`,
both must be densified.


Operations with :obj:`numpy.ndarray`
------------------------------------
Certain operations with :obj:`numpy.ndarray` are also supported. For example,
the following are all allowed if :code:`x` is a :obj:`numpy.ndarray` and
:code:`(x == 0).all()` evaluates to :code:`True`:

.. code-block:: python

   x + y
   x - y

The following is true so long as there are no infinities or NaNs in :code:`x`:

.. code-block:: python

   x * y

In general, if operating on the :code:`numpy.ndarray` with a zero would produce
all-zeros then the operation is supported.


Operations with :obj:`scipy.sparse.spmatrix`
--------------------------------------------
Certain operations with :obj:`scipy.sparse.spmatrix` are also supported.
For example, the following are all allowed if :code:`y` is a :obj:`scipy.sparse.spmatrix`:

.. code-block:: python

   x + y
   x - y
   x * y
   x > y
   x < y

In general, if operating on a :code:`scipy.sparse.spmatrix` is the same as operating
on :obj:`COO`, as long as it is to the right of the operator.

.. note:: Results are not guaranteed if :code:`x` is a :obj:`scipy.sparse.spmatrix`.
   For this reason, we recommend that all Scipy sparse matrices should be explicitly
   converted to :obj:`COO` before any operations.


Broadcasting
------------
All binary operators support :obj:`broadcasting <numpy.doc.broadcasting>`.
This means that (under certain conditions) you can perform binary operations
on arrays with unequal shape. Namely, when the shape is missing a dimension,
or when a dimension is :code:`1`. For example, performing a binary operation
on two :obj:`COO` arrays with shapes :code:`(4,)` and :code:`(5, 1)` yields
an object of shape :code:`(5, 4)`. The same happens with arrays of shape
:code:`(1, 4)` and :code:`(5, 1)`. However, :code:`(4, 1)` and :code:`(5, 1)`
will raise a :obj:`ValueError`.


Full List of Operators
----------------------
Here, :code:`x` and :code:`y` can be :obj:`COO` arrays,
:obj:`numpy.ndarray` objects or scalars, keeping in mind :ref:`auto
densification rules <auto-densification>`. In addition, :code:`y` can also
be a :obj:`scipy.sparse.spmatrix` The following operators are supported:

* Basic algebraic operations

   * :obj:`operator.add` (:code:`x + y`)
   * :obj:`operator.neg` (:code:`-x`)
   * :obj:`operator.sub` (:code:`x - y`)
   * :obj:`operator.mul` (:code:`x * y`)
   * :obj:`operator.truediv` (:code:`x / y`)
   * :obj:`operator.floordiv` (:code:`x // y`)
   * :obj:`operator.pow` (:code:`x ** y`)

* Comparison operators

   * :obj:`operator.eq` (:code:`x == y`)
   * :obj:`operator.ne` (:code:`x != y`)
   * :obj:`operator.gt` (:code:`x > y`)
   * :obj:`operator.ge` (:code:`x >= y`)
   * :obj:`operator.lt` (:code:`x < y`)
   * :obj:`operator.le` (:code:`x <= y`)

* Bitwise operators

   * :obj:`operator.and_` (:code:`x & y`)
   * :obj:`operator.or_` (:code:`x | y`)
   * :obj:`operator.xor` (:code:`x ^ y`)

* Bit-shifting operators

   * :obj:`operator.lshift` (:code:`x << y`)
   * :obj:`operator.rshift` (:code:`x >> y`)

.. note:: In-place operators are not supported at this time.


Reductions
----------
:obj:`COO` objects support a number of reductions. However, not all important
reductions are currently implemented (help welcome!) All of the following
currently work:

.. code-block:: python

   x.sum(axis=1)
   np.max(x)
   np.min(x, axis=(0, 2))
   x.prod()

.. note::
   If you are performing multiple reductions along the same axes, it may
   be beneficial to call :obj:`COO.enable_caching`.

:obj:`COO.reduce`
~~~~~~~~~~~~~~~~~
This method can take an arbitrary :obj:`numpy.ufunc` and performs a
reduction using that method. For example, the following will perform
a sum:

.. code-block:: python

   x.reduce(np.add, axis=1)

.. note::
   :obj:`sparse` currently performs reductions by grouping together all
   coordinates along the supplied axes and reducing those. Then, if the
   number in a group is deficient, it reduces an extra time with zero.
   As a result, if reductions can change by adding multiple zeros to
   it, this method won't be accurate. However, it works in most cases.

Partial List of Supported Reductions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Although any binary :obj:`numpy.ufunc` should work for reductions, when calling
in the form :code:`x.reduction()`, the following reductions are supported:

* :obj:`COO.sum`
* :obj:`COO.max`
* :obj:`COO.min`
* :obj:`COO.prod`


Indexing
--------
:obj:`COO` arrays can be :obj:`indexed <numpy.doc.indexing>` just like regular
:obj:`numpy.ndarray` objects. They support integer, slice and boolean indexing.
However, currently, numpy advanced indexing is not properly supported. This
means that all of the following work like in Numpy, except that they will produce
:obj:`COO` arrays rather than :obj:`numpy.ndarray` objects, and will produce
scalars where expected. Assume that :code:`z.shape` is :code:`(5, 6, 7)`

.. code-block:: python

   z[0]
   z[1, 3]
   z[1, 4, 3]
   z[:3, :2, 3]
   z[::-1, 1, 3]
   z[-1]
   z[[True, False, True, False, True], 3, 4]

All of the following will raise an :obj:`IndexError`, like in Numpy 1.13 and later.

.. code-block:: python

   z[6]
   z[3, 6]
   z[1, 4, 8]
   z[-6]
   z[[True, True, False, True], 3, 4]

.. note:: Numpy advanced indexing is currently not supported.


Other Operations
----------------
:obj:`COO` arrays support a number of other common operations. Among them are
:obj:`dot`, :obj:`tensordot`, :obj:`concatenate` and :obj:`stack`,
:obj:`COO.transpose` and :obj:`COO.reshape`. You can view the full list on the
API reference page for :obj:`sparse`
