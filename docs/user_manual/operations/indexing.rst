.. currentmodule:: sparse

Indexing
========
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