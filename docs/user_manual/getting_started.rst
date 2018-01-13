.. currentmodule:: sparse

Getting Started
===============
:obj:`COO` arrays can be constructed from :obj:`numpy.ndarray` objects and
:obj:`scipy.sparse.spmatrix` objects. For example, to generate the identity
matrix,

.. code-block:: python

   import numpy as np
   import scipy.sparse
   import sparse

   sps_identity = scipy.sparse.eye(5)
   identity = sparse.COO.from_scipy_sparse(sps_identity)

:obj:`COO` arrays can have operations performed on them just like :obj:`numpy.ndarray`
objects. For example, to add two :obj:`COO` arrays:

.. code-block:: python

   z = x + y

You can also apply any :obj:`numpy.ufunc` to :obj:`COO` arrays.

.. code-block:: python

   sin_x = np.sin(x)

However, operations which convert the sparse array into a dense one aren't currently
supported. For example, the following raises a :obj:`ValueError`.

.. code-block:: python

   y = x + 5

However, if you're sure you want to convert a sparse array to a dense one, you can
do this (which will result in a :obj:`numpy.ndarray`):

.. code-block:: python

   y = x.todense() + 5

That's it! You can move on to the :doc:`user manual <../user_manual>` to see what
part of this library interests you, or you can jump straight in with the :doc:`API reference
<../api>`.
