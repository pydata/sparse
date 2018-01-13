.. currentmodule:: sparse

Reductions
==========
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
-----------------
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
------------------------------------
Although any binary :obj:`numpy.ufunc` should work for reductions, when calling
in the form :code:`x.reduction()`, the following reductions are supported:

* :obj:`COO.sum`
* :obj:`COO.max`
* :obj:`COO.min`
* :obj:`COO.prod`
