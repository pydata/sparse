COO
===

.. currentmodule:: sparse

.. autoclass:: COO

   .. note::
      :obj:`COO` objects also support :doc:`operators <../user_manual/operations/basic>`
      and :doc:`indexing <../user_manual/operations/indexing>`

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

   .. rubric:: :doc:`Constructing COO objects <../user_manual/constructing>`
   .. autosummary::
      :toctree:

      COO.from_numpy
      COO.from_scipy_sparse

   .. rubric:: :doc:`Element-wise operations <../user_manual/operations/elemwise>`
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

   .. rubric:: :doc:`Reductions <../user_manual/operations/reductions>`
   .. autosummary::
      :toctree:

      COO.reduce
      COO.sum
      COO.max
      COO.min
      COO.prod

   .. rubric:: :doc:`Converting to other formats <../user_manual/converting>`
   .. autosummary::
      :toctree:

      COO.todense
      COO.maybe_densify
      COO.to_scipy_sparse
      COO.tocsc
      COO.tocsr

   .. rubric:: :doc:`Other operations <../user_manual/operations/other>`
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
