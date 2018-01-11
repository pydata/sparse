COO
===

.. currentmodule:: sparse

.. autoclass:: COO

   .. note::
      :obj:`COO` objects also support :doc:`operators <../getting_started/operations/basic>`
      and :doc:`indexing <../getting_started/operations/indexing>`

   .. rubric:: Attributes
   .. autosummary::
      :toctree:

      COO.T
      COO.dtype
      COO.nbytes
      COO.ndim
      COO.nnz

   .. rubric:: :doc:`Constructing COO objects <../getting_started/constructing>`
   .. autosummary::
      :toctree:

      COO.__init__
      COO.from_numpy
      COO.from_scipy_sparse

   .. rubric:: :doc:`Element-wise operations <../getting_started/operations/elemwise>`
   .. autosummary::
      :toctree:

      COO.elemwise
      COO.abs
      COO.astype
      COO.ceil
      COO.conj
      COO.conjugate
      COO.exp
      COO.expm1
      COO.floor
      COO.log1p
      COO.rint
      COO.round
      COO.sin
      COO.sinh
      COO.sqrt
      COO.tan
      COO.tanh

   .. rubric:: :doc:`Reductions <../getting_started/operations/reductions>`
   .. autosummary::
      :toctree:

      COO.reduce
      COO.sum
      COO.max
      COO.min
      COO.prod

   .. rubric:: :doc:`Converting to other formats <../getting_started/converting>`
   .. autosummary::
      :toctree:

      COO.todense
      COO.maybe_densify
      COO.to_scipy_sparse
      COO.tocsc
      COO.tocsr

   .. rubric:: :doc:`Other operations <../getting_started/operations/other>`
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
