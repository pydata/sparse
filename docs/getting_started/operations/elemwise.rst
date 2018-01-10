.. currentmodule:: sparse

Element-wise Operations
=======================
:obj:`COO` arrays support a variety of element-wise operations. However, as
with operators, operations that map zero to a nonzero value are not supported.

To illustrate, the following are all possible, and will produce another
:obj:`COO` array:

.. code-block:: python

   x.abs()
   np.sin(x)
   np.sqrt(x)
   x.conj()
   x.expm1()
   np.log1p(x)

However, the following are all unsupported and will raise a :obj:`ValueError`:

.. code-block:: python

   x.exp()
   np.cos(x)
   np.log(x)

Notice that you can apply any unary or binary :obj:`numpy.ufunc` to :obj:`COO`
arrays, :obj:`scipy.sparse.spmatrix` objects and scalars and it will work so
long as the result is not dense.

:obj:`COO.elemwise`
-------------------
This function allows you to apply any arbitrary unary or binary function where
the first object is :obj:`COO`, and the second is a scalar, :obj:`COO`, or
a :obj:`scipy.sparse.spmatrix`. For example, the following will add two
:obj:`COO` objects:

.. code-block:: python

   x.elemwise(np.add, y)

Partial List of Supported :obj:`numpy.ufunc` s
----------------------------------------------
Although any unary or binary :obj:`numpy.ufunc` should work if the result is
not dense, when calling in the form :code:`x.func()`, the following operations
are supported:

* :obj:`COO.abs`
* :obj:`COO.expm1`
* :obj:`COO.log1p`
* :obj:`COO.sin`
* :obj:`COO.sinh`
* :obj:`COO.tan`
* :obj:`COO.tanh`
* :obj:`COO.sqrt`
* :obj:`COO.ceil`
* :obj:`COO.floor`
* :obj:`COO.round`
* :obj:`COO.rint`
* :obj:`COO.conj`
* :obj:`COO.conjugate`
* :obj:`COO.astype`
