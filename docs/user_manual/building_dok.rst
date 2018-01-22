.. currentmodule:: sparse

Building :obj:`COO` Arrays from :obj:`DOK` Arrays
=================================================
It's possible to build :obj:`COO` arrays from :obj:`DOK` arrays, if it is not
easy to construct the :code:`coords` and :obj:`data` in a simple way. :obj:`DOK`
arrays provide a simple builder interface to build :obj:`COO` arrays, but at
this time, they can do little else.

You can get started by defining the shape (and optionally, datatype) of the
:obj:`DOK` array. If you do not specify a dtype, it is inferred from the value
dictionary or is set to :code:`dtype('float64')` if that is not present.

.. code-block:: python

   s = DOK((6, 5, 2))
   s2 = DOK((2, 3, 4), dtype=np.float64)

After this, you can build the array by assigning arrays or scalars to elements
or slices of the original array. Broadcasting rules are followed.

.. code-block:: python

   s[1:3, 3:1:-1] = [[6, 5]]

At the end, you can convert the :obj:`DOK` array to a :obj:`COO` array, and
perform arithmetic or other operations on it.

.. code-block:: python

   s2 = COO(s)

In addition, it is possible to access single elements of the :obj:`DOK` array
using normal Numpy indexing.

.. code-block:: python

   s[1, 2, 1]  # 5
   s[5, 1, 1]  # 0
