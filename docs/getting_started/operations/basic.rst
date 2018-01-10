.. currentmodule:: sparse

Basic Operations
================
:obj:`COO` objects can have a number of operators applied to them. They support
operations with scalars, :obj:`scipy.sparse.spmatrix` objects, and other
:obj:`COO` objects. For example, to get the sum of two :obj:`COO` objects, you
would do the following:

.. code-block:: python

   z = x + y

Note that in-place operators are currently not supported. For example,

.. code-block:: python

   x += y

will not work.

.. _auto-densification:

Auto-Densification
------------------
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
:obj:`scipy.sparse.spmatrix` objects or scalars, keeping in mind :ref:`auto
densification rules <auto-densification>`. The following operators are supported:

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
