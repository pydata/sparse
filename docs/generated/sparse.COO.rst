COO
===

.. currentmodule:: sparse

.. autoclass:: COO

   .. note:: :obj:`COO` objects also support :ref:`operators <operations-operators>`
      and :ref:`indexing <operations-indexing>`

   .. rubric:: Attributes
   .. autosummary::
      :toctree:

      COO.T
      COO.dtype
      COO.nbytes
      COO.ndim
      COO.nnz
      COO.size
      COO.density

   .. rubric:: :doc:`Constructing COO objects <../construct>`
   .. autosummary::
      :toctree:

      COO.from_numpy
      COO.from_scipy_sparse

   .. rubric:: :ref:`Element-wise operations <operations-elemwise>`
   .. autosummary::
      :toctree:

      COO.astype
      COO.round

   .. rubric:: :ref:`Reductions <operations-reductions>`
   .. autosummary::
      :toctree:

      COO.reduce
      COO.sum
      COO.max
      COO.min
      COO.prod

      COO.nanreduce

   .. rubric:: :ref:`Converting to other formats <converting>`
   .. autosummary::
      :toctree:

      COO.todense
      COO.maybe_densify
      COO.to_scipy_sparse
      COO.tocsc
      COO.tocsr

   .. rubric:: :ref:`Other operations <operations-other>`
   .. autosummary::
      :toctree:

      COO.dot
      COO.reshape
      COO.transpose

   .. rubric:: Utility functions
   .. autosummary::
      :toctree:

      COO.broadcast_to
      COO.enable_caching
      COO.linear_loc
      COO.sort_indices
      COO.sum_duplicates
