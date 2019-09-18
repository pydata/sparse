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
      COO.imag
      COO.real

   .. rubric:: :doc:`Constructing COO objects <../construct>`
   .. autosummary::
      :toctree:

      COO.from_iter
      COO.from_numpy
      COO.from_scipy_sparse

   .. rubric:: :ref:`Element-wise operations <operations-elemwise>`
   .. autosummary::
      :toctree:

      COO.astype
      COO.conj
      COO.clip
      COO.round

   .. rubric:: :ref:`Reductions <operations-reductions>`
   .. autosummary::
      :toctree:

      COO.reduce

      COO.sum
      COO.prod
      COO.min
      COO.max
      COO.any
      COO.all
      COO.mean
      COO.std
      COO.var

   .. rubric:: :ref:`Converting to other formats <converting>`
   .. autosummary::
      :toctree:

      COO.asformat
      COO.todense
      COO.maybe_densify
      COO.to_scipy_sparse
      COO.tocsc
      COO.tocsr

   .. rubric:: :ref:`Other operations <operations-other>`
   .. autosummary::
      :toctree:

      COO.copy
      COO.dot
      COO.reshape
      COO.resize
      COO.transpose
      COO.nonzero

   .. rubric:: Utility functions
   .. autosummary::
      :toctree:

      COO.broadcast_to
      COO.enable_caching
      COO.linear_loc
