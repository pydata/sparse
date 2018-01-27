.. currentmodule:: sparse

Element-wise Operations
=======================
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
-------------------
This function allows you to apply any arbitrary unary or binary function where
the first object is :obj:`COO`, and the second is a scalar, :obj:`COO`, or
a :obj:`numpy.ndarray`. For example, the following will add two
:obj:`COO` objects:

.. code-block:: python

   x.elemwise(np.add, y)

