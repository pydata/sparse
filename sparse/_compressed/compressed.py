import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from functools import reduce
from operator import mul
from collections.abc import Iterable
import scipy.sparse as ss

from .._sparse_array import SparseArray
from .._coo.common import linear_loc
from .._common import dot, matmul
from .._utils import normalize_axis, check_zero_fill_value, check_compressed_axes
from .._coo.core import COO
from .convert import uncompress_dimension, _transpose, _1d_reshape
from .indexing import getitem


def _from_coo(x, compressed_axes=None):

    if x.ndim == 0:
        if compressed_axes is not None:
            raise ValueError("no axes to compress for 0d array")
        return ((x.data, x.coords, []), x.shape, None, x.fill_value)

    if x.ndim == 1:
        if compressed_axes is not None:
            raise ValueError("no axes to compress for 1d array")
        return ((x.data, x.coords[0], ()), x.shape, None, x.fill_value)

    compressed_axes = normalize_axis(compressed_axes, x.ndim)
    if compressed_axes is None:
        # defaults to best compression ratio
        compressed_axes = (np.argmin(x.shape),)

    check_compressed_axes(x.shape, compressed_axes)

    axis_order = list(compressed_axes)
    # array location where the uncompressed dimensions start
    axisptr = len(compressed_axes)
    axis_order.extend(np.setdiff1d(np.arange(len(x.shape)), compressed_axes))
    reordered_shape = tuple(x.shape[i] for i in axis_order)
    row_size = np.prod(reordered_shape[:axisptr])
    col_size = np.prod(reordered_shape[axisptr:])
    compressed_shape = (row_size, col_size)
    shape = x.shape

    # transpose axes, linearize, reshape, and compress
    linear = linear_loc(x.coords[axis_order], reordered_shape)
    order = np.argsort(linear)
    linear = linear[order]
    coords = np.empty((2, x.nnz), dtype=np.intp)
    strides = 1
    for i, d in enumerate(compressed_shape[::-1]):
        coords[-(i + 1), :] = (linear // strides) % d
        strides *= d
    indptr = np.empty(row_size + 1, dtype=np.intp)
    indptr[0] = 0
    np.cumsum(np.bincount(coords[0], minlength=row_size), out=indptr[1:])
    indices = coords[1]
    data = x.data[order]
    return ((data, indices, indptr), shape, compressed_axes, x.fill_value)


class GCXS(SparseArray, NDArrayOperatorsMixin):
    """
    A sparse multidimensional array.

    This is stored in GCXS format, a generalization of the GCRS/GCCS formats 
    from 'Efficient storage scheme for n-dimensional sparse array: GCRS/GCCS':
    https://ieeexplore.ieee.org/document/7237032. GCXS generalizes the csr/csc
    sparse matrix formats. For arrays with ndim == 2, GCXS is the same csr/csc.
    For arrays with ndim >2, any combination of axes can be compressed, 
    significantly reducing storage. 


    Parameters
    ----------
    arg : tuple (data, indices, indptr)
        A tuple of arrays holding the data, indices, and 
        index pointers for the nonzero values of the array.
    shape : tuple[int] (COO.ndim,)
        The shape of the array.
    compressed_axes : Iterable[int]
        The axes to compress.
    fill_value: scalar, optional
        The fill value for this array.

    Attributes
    ----------
    data : numpy.ndarray (nnz,)
        An array holding the nonzero values corresponding to :obj:`GCXS.indices`.
    indices : numpy.ndarray (nnz,)
        An array holding the coordinates of every nonzero element along uncompressed dimensions.
    indptr : numpy.ndarray
        An array holding the cumulative sums of the nonzeros along the compressed dimensions. 
    shape : tuple[int] (ndim,)
        The dimensions of this array.

    See Also
    --------
    DOK : A mostly write-only sparse array.
    """

    __array_priority__ = 12

    def __init__(self, arg, shape=None, compressed_axes=None, fill_value=0):

        if isinstance(arg, np.ndarray):
            (arg, shape, compressed_axes, fill_value) = _from_coo(
                COO(arg), compressed_axes
            )

        elif isinstance(arg, COO):
            (arg, shape, compressed_axes, fill_value) = _from_coo(arg, compressed_axes)

        if shape is None:
            raise ValueError("missing `shape` argument")

        check_compressed_axes(len(shape), compressed_axes)

        if len(shape) == 1:
            compressed_axes = None

        self.data, self.indices, self.indptr = arg

        if self.data.ndim != 1:
            raise ValueError("data must be a scalar or 1-dimensional.")

        self.shape = shape
        self.compressed_axes = (
            tuple(compressed_axes) if isinstance(compressed_axes, Iterable) else None
        )
        self.fill_value = fill_value

    @classmethod
    def from_numpy(cls, x, compressed_axes=None, fill_value=0):
        coo = COO(x, fill_value=fill_value)
        return cls.from_coo(coo, compressed_axes)

    @classmethod
    def from_coo(cls, x, compressed_axes=None):
        (arg, shape, compressed_axes, fill_value) = _from_coo(x, compressed_axes)
        return cls(
            arg, shape=shape, compressed_axes=compressed_axes, fill_value=fill_value
        )

    @classmethod
    def from_scipy_sparse(cls, x):
        if x.format == "csc":
            return cls(
                (x.data, x.indices, x.indptr), shape=x.shape, compressed_axes=(1,)
            )
        else:
            x = x.asformat("csr")
            return cls(
                (x.data, x.indices, x.indptr), shape=x.shape, compressed_axes=(0,)
            )

    @classmethod
    def from_iter(cls, x, shape=None, compressed_axes=None, fill_value=None):
        return cls.from_coo(
            COO.from_iter(x, shape, fill_value), compressed_axes=compressed_axes
        )

    @property
    def dtype(self):
        """
        The datatype of this array.
        
        Returns
        -------
        numpy.dtype
            The datatype of this array.
            
        See Also
        --------
        numpy.ndarray.dtype : Numpy equivalent property.
        scipy.sparse.csr_matrix.dtype : Scipy equivalent property.
        """
        return self.data.dtype

    @property
    def nnz(self):
        """
        The number of nonzero elements in this array.
        
        Returns
        -------
        int
            The number of nonzero elements in this array.
            
        See Also
        --------
        COO.nnz : Equivalent :obj:`COO` array property.
        DOK.nnz : Equivalent :obj:`DOK` array property.
        numpy.count_nonzero : A similar Numpy function.
        scipy.sparse.csr_matrix.nnz : The Scipy equivalent property.
        """
        return self.data.shape[0]

    @property
    def nbytes(self):
        """
        The number of bytes taken up by this object. Note that for small arrays,
        this may undercount the number of bytes due to the large constant overhead.
        
        Returns
        -------
        int
            The approximate bytes of memory taken by this object.
        
        See Also
        --------
        numpy.ndarray.nbytes : The equivalent Numpy property.
        """
        nbytes = self.data.nbytes + self.indices.nbytes + self.indptr.nbytes
        return nbytes

    @property
    def _axis_order(self):
        axis_order = list(self.compressed_axes)
        axis_order.extend(
            np.setdiff1d(np.arange(len(self.shape)), self.compressed_axes)
        )
        return axis_order

    @property
    def _axisptr(self):
        # array location where the uncompressed dimensions start
        return len(self.compressed_axes)

    @property
    def _compressed_shape(self):
        row_size = np.prod(self._reordered_shape[: self._axisptr])
        col_size = np.prod(self._reordered_shape[self._axisptr :])
        return (row_size, col_size)

    @property
    def _reordered_shape(self):
        return tuple(self.shape[i] for i in self._axis_order)

    @property
    def T(self):
        return self.transpose()

    def __str__(self):
        return "<GCXS: shape={}, dtype={}, nnz={}, fill_value={}, compressed_axes={}>".format(
            self.shape, self.dtype, self.nnz, self.fill_value, self.compressed_axes
        )

    __repr__ = __str__

    __getitem__ = getitem

    def change_compressed_axes(self, new_compressed_axes):
        """
        Returns a new array with specified compressed axes. This operation is similar to converting 
        a scipy.sparse.csc_matrix to a scipy.sparse.csr_matrix.

        Returns
        -------
        GCXS
            A new instance of the input array with compression along the specified dimensions.
        """
        if self.ndim == 1:
            raise NotImplementedError("no axes to compress for 1d array")

        new_compressed_axes = tuple(
            normalize_axis(new_compressed_axes[i], self.ndim)
            for i in range(len(new_compressed_axes))
        )

        if new_compressed_axes == self.compressed_axes:
            return self

        if len(new_compressed_axes) >= len(self.shape):
            raise ValueError("cannot compress all axes")
        if len(set(new_compressed_axes)) != len(new_compressed_axes):
            raise ValueError("repeated axis in compressed_axes")

        arg = _transpose(self, self.shape, np.arange(self.ndim), new_compressed_axes)

        return GCXS(
            arg,
            shape=self.shape,
            compressed_axes=new_compressed_axes,
            fill_value=self.fill_value,
        )

    def tocoo(self):
        """
        Convert this :obj:`GCXS` array to a :obj:`COO`. 

        Returns
        -------
        sparse.COO
            The converted COO array.
        """
        if self.ndim == 0:
            return COO(
                np.array([])[None],
                self.data,
                shape=self.shape,
                fill_value=self.fill_value,
            )
        if self.ndim == 1:
            return COO(
                self.indices[None, :],
                self.data,
                shape=self.shape,
                fill_value=self.fill_value,
            )
        uncompressed = uncompress_dimension(self.indptr)
        coords = np.vstack((uncompressed, self.indices))
        order = np.argsort(self._axis_order)
        return (
            COO(
                coords,
                self.data,
                shape=self._compressed_shape,
                fill_value=self.fill_value,
            )
            .reshape(self._reordered_shape)
            .transpose(order)
        )

    def todense(self):
        """
        Convert this :obj:`GCXS` array to a dense :obj:`numpy.ndarray`. Note that
        this may take a large amount of memory if the :obj:`GCXS` object's :code:`shape`
        is large.

        Returns
        -------
        numpy.ndarray
            The converted dense array.

        See Also
        --------
        DOK.todense : Equivalent :obj:`DOK` array method.
        COO.todense : Equivalent :obj:`COO` array method.
        scipy.sparse.coo_matrix.todense : Equivalent Scipy method.
        """
        if self.compressed_axes is None:
            out = np.full(self.shape, self.fill_value, self.dtype)
            if len(self.indices) != 0:
                out[self.indices] = self.data
            else:
                if len(self.data) != 0:
                    out[()] = self.data[0]
            return out
        return self.tocoo().todense()

    def todok(self):

        from ..dok import DOK

        return DOK.from_coo(self.tocoo())  # probably a temporary solution

    def to_scipy_sparse(self):
        """
        Converts this :obj:`GCXS` object into a :obj:`scipy.sparse.csr_matrix` or `scipy.sparse.csc_matrix`.
        Returns
        -------
        :obj:`scipy.sparse.csr_matrix` or `scipy.sparse.csc_matrix`
            The converted Scipy sparse matrix.
        Raises
        ------
        ValueError
            If the array is not two-dimensional.
        ValueError
            If all the array doesn't zero fill-values.
        """

        check_zero_fill_value(self)

        if self.ndim != 2:
            raise ValueError(
                "Can only convert a 2-dimensional array to a Scipy sparse matrix."
            )

        if 0 in self.compressed_axes:
            return ss.csr_matrix(
                (self.data, self.indices, self.indptr), shape=self.shape
            )
        else:
            return ss.csc_matrix(
                (self.data, self.indices, self.indptr), shape=self.shape
            )

    def asformat(self, format, compressed_axes=None):
        """
        Convert this sparse array to a given format.
        Parameters
        ----------
        format : str
            A format string.
        Returns
        -------
        out : SparseArray
            The converted array.
        Raises
        ------
        NotImplementedError
            If the format isn't supported.
        """

        if format == "coo":
            return self.tocoo()
        elif format == "dok":
            return self.todok()
        elif format == "gcxs":
            if compressed_axes is None:
                compressed_axes = self.compressed_axes
            return self.change_compressed_axes(compressed_axes)

        raise NotImplementedError("The given format is not supported.")

    def maybe_densify(self, max_size=1000, min_density=0.25):
        """
        Converts this :obj:`GCXS` array to a :obj:`numpy.ndarray` if not too
        costly.
        Parameters
        ----------
        max_size : int
            Maximum number of elements in output
        min_density : float
            Minimum density of output
        Returns
        -------
        numpy.ndarray
            The dense array.
        See Also
        --------
        sparse.GCXS.todense: Converts to Numpy function without checking the cost.
        sparse.COO.maybe_densify: The equivalent COO function.
        Raises
        -------
        ValueError
            If the returned array would be too large.
        """

        if self.size <= max_size or self.density >= min_density:
            return self.todense()
        else:
            raise ValueError(
                "Operation would require converting " "large sparse array to dense"
            )

    def flatten(self, order="C"):
        """
        Returns a new :obj:`GCXS` array that is a flattened version of this array.

        Returns
        -------
        GCXS
            The flattened output array.

        Notes
        -----
        The :code:`order` parameter is provided just for compatibility with
        Numpy and isn't actually supported.
        """
        if order not in {"C", None}:
            raise NotImplementedError("The `order` parameter is not" "supported.")

        return self.reshape(-1)

    def reshape(self, shape, order="C", compressed_axes=None):
        """
        Returns a new :obj:`GCXS` array that is a reshaped version of this array.
        Parameters
        ----------
        shape : tuple[int]
            The desired shape of the output array.
        compressed_axes : Iterable[int], optional
            The axes to compress to store the array. Finds the most efficient storage
            by default.
        Returns
        -------
        GCXS
            The reshaped output array.
        See Also
        --------
        numpy.ndarray.reshape : The equivalent Numpy function.
        sparse.COO.reshape: The equivalent COO function.
        Notes
        -----
        The :code:`order` parameter is provided just for compatibility with
        Numpy and isn't actually supported.

        """
        if isinstance(shape, Iterable):
            shape = tuple(shape)
        else:
            shape = (shape,)
        if order not in {"C", None}:
            raise NotImplementedError("The 'order' parameter is not supported")
        if any(d == -1 for d in shape):
            extra = int(self.size / np.prod([d for d in shape if d != -1]))
            shape = tuple([d if d != -1 else extra for d in shape])

        if self.shape == shape:
            return self

        if self.size != reduce(mul, shape, 1):
            raise ValueError(
                "cannot reshape array of size {} into shape {}".format(self.size, shape)
            )
        if len(shape) == 0:
            return self.tocoo().reshape(shape).asformat("gcxs")

        if compressed_axes is None:
            if len(shape) == self.ndim:
                compressed_axes = self.compressed_axes
            elif len(shape) == 1:
                compressed_axes = None
            else:
                compressed_axes = (np.argmin(shape),)

        if self.ndim == 1:
            arg = _1d_reshape(self, shape, compressed_axes)
        else:
            arg = _transpose(self, shape, np.arange(self.ndim), compressed_axes)
        return GCXS(
            arg,
            shape=tuple(shape),
            compressed_axes=compressed_axes,
            fill_value=self.fill_value,
        )

    def resize(self, *args, refcheck=True, compressed_axes=None):
        """
        This method changes the shape and size of an array in-place.

        Parameters
        ----------
        args : tuple, or series of integers
            The desired shape of the output array.
        compressed_axes : Iterable[int], optional
            The axes to compress to store the array. Finds the most efficient storage
            by default.

        See Also
        --------
        numpy.ndarray.resize : The equivalent Numpy function.
        sparse.COO.resize : The equivalent COO function.
        """
        from .convert import _resize

        if len(args) == 1 and isinstance(args[0], tuple):
            shape = args[0]
        elif all(isinstance(arg, int) for arg in args):
            shape = tuple(args)
        else:
            raise ValueError("Invalid input")

        if any(d < 0 for d in shape):
            raise ValueError("negative dimensions not allowed")

        if self.shape == shape:
            return

        if compressed_axes is None:
            if len(shape) == self.ndim:
                compressed_axes = self.compressed_axes
            elif len(shape) == 1:
                compressed_axes = None
            else:
                compressed_axes = (np.argmin(shape),)

        arg = _resize(self, shape, compressed_axes)
        self.data, self.indices, self.indptr = arg
        self.shape = shape
        self.compressed_axes = compressed_axes

    def transpose(self, axes=None, compressed_axes=None):
        """
        Returns a new array which has the order of the axes switched.

        Parameters
        ----------
        axes : Iterable[int], optional
            The new order of the axes compared to the previous one. Reverses the axes
            by default.
        compressed_axes : Iterable[int], optional
            The axes to compress to store the array. Finds the most efficient storage
            by default.

        Returns
        -------
        GCXS
            The new array with the axes in the desired order.

        See Also
        --------
        :obj:`GCXS.T` : A quick property to reverse the order of the axes.
        numpy.ndarray.transpose : Numpy equivalent function.
        """
        if axes is None:
            axes = list(reversed(range(self.ndim)))

        # Normalize all axes indices to positive values
        axes = normalize_axis(axes, self.ndim)

        if len(np.unique(axes)) < len(axes):
            raise ValueError("repeated axis in transpose")

        if not len(axes) == self.ndim:
            raise ValueError("axes don't match array")

        axes = tuple(axes)

        if axes == tuple(range(self.ndim)):
            return self

        if self.ndim == 2:
            return self._2d_transpose()

        shape = tuple(self.shape[ax] for ax in axes)

        if compressed_axes is None:
            compressed_axes = (np.argmin(shape),)
        arg = _transpose(self, shape, axes, compressed_axes, transpose=True)
        return GCXS(
            arg,
            shape=shape,
            compressed_axes=compressed_axes,
            fill_value=self.fill_value,
        )

    def _2d_transpose(self):
        """
        A function for performing constant-time transposes on 2d GCXS arrays.
        
        Returns
        -------
        GCXS
            The new transposed array with the opposite compressed axes as the input.

        See Also
        --------
        scipy.sparse.csr_matrix.tocsc : Scipy equivalent function.
        scipy.sparse.csc_matrix.tocsr : Scipy equivalent function.
        numpy.ndarray.transpose : Numpy equivalent function.
        """
        if self.ndim != 2:
            raise ValueError(
                "cannot perform 2d transpose on array with dimension {}".format(
                    self.ndim
                )
            )

        compressed_axes = [(self.compressed_axes[0] + 1) % 2]
        shape = self.shape[::-1]
        return GCXS(
            (self.data, self.indices, self.indptr),
            shape=shape,
            compressed_axes=compressed_axes,
            fill_value=self.fill_value,
        )

    def dot(self, other):
        """
        Performs the equivalent of :code:`x.dot(y)` for :obj:`GCXS`.

        Parameters
        ----------
        other : Union[GCXS, COO, numpy.ndarray, scipy.sparse.spmatrix]
            The second operand of the dot product operation.

        Returns
        -------
        {GCXS, numpy.ndarray}
            The result of the dot product. If the result turns out to be dense,
            then a dense array is returned, otherwise, a sparse array.

        Raises
        ------
        ValueError
            If all arguments don't have zero fill-values.

        See Also
        --------
        dot : Equivalent function for two arguments.
        :obj:`numpy.dot` : Numpy equivalent function.
        scipy.sparse.csr_matrix.dot : Scipy equivalent function.
        """
        return dot(self, other)

    def __matmul__(self, other):
        try:
            return matmul(self, other)
        except NotImplementedError:
            return NotImplemented

    def __rmatmul__(self, other):
        try:
            return matmul(other, self)
        except NotImplementedError:
            return NotImplemented
