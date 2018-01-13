.. currentmodule:: sparse

Converting :obj:`COO` objects to other Formats
==============================================
:obj:`COO` arrays can be converted to :obj:`numpy.ndarray` objects,
or to some :obj:`scipy.sparse.spmatrix` subclasses via the following
methods:

* :obj:`COO.todense`: Converts to a :obj:`numpy.ndarray` unconditionally.
* :obj:`COO.maybe_densify`: Converts to a :obj:`numpy.ndarray` based on
   certain constraints.
* :obj:`COO.to_scipy_sparse`: Converts to a :obj:`scipy.sparse.coo_matrix` if
   the array is two dimensional.
* :obj:`COO.tocsr`: Converts to a :obj:`scipy.sparse.csr_matrix` if
   the array is two dimensional.
* :obj:`COO.tocsc`: Converts to a :obj:`scipy.sparse.csc_matrix` if
   the array is two dimensional.