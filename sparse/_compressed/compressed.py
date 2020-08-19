import numpy as np
import numba
import operator
from numpy.lib.mixins import NDArrayOperatorsMixin
from functools import reduce
from collections.abc import Iterable
import scipy.sparse as ss

from .._sparse_array import SparseArray
from .._coo.common import linear_loc
from .._common import dot, matmul
from .._utils import (
    normalize_axis,
    equivalent,
    check_zero_fill_value,
    check_compressed_axes,
)
from .._coo.core import COO
from .convert import uncompress_dimension, _transpose, _1d_reshape
from .indexing import getitem

_reduce_super_ufunc = {np.add: np.multiply, np.multiply: np.power}


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
        nbytes = self.data.nbytes + self.indices.nbytes
        if self.indptr != ():
            nbytes += self.indptr.nbytes
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

        if self.size != reduce(operator.mul, shape, 1):
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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, GCXS) for x in out):
            return NotImplemented

        if getattr(ufunc, "signature", None) is not None:
            return self.__array_function__(ufunc, (np.ndarray, GCXS), inputs, kwargs)

        if out is not None:
            kwargs["dtype"] = out[0].dtype

        if method == "outer":
            method = "__call__"

            cum_ndim = 0
            inputs_transformed = []
            for inp in reversed(inputs):
                inputs_transformed.append(inp[(Ellipsis,) + (None,) * cum_ndim])
                cum_ndim += inp.ndim

            inputs = tuple(reversed(inputs_transformed))

        # if method == "__call__":
        #    result = elemwise(ufunc, *inputs, **kwargs)
        if method == "reduce":
            result = GCXS._reduce(ufunc, *inputs, **kwargs)
        else:
            return NotImplemented

        if out is not None:
            (out,) = out
            if out.shape != result.shape:
                raise ValueError(
                    "non-broadcastable output operand with shape %s "
                    "doesn't match the broadcast shape %s" % (out.shape, result.shape)
                )

            out._make_shallow_copy_of(result)
            return out

        return result

    @staticmethod
    def _reduce(method, *args, **kwargs):
        assert len(args) == 1

        self = args[0]
        if isinstance(self, ss.spmatrix):
            self = GCXS.from_scipy_sparse(self)

        return self.reduce(method, **kwargs)

    def reduce(self, method, axis=(0,), keepdims=False, **kwargs):
        """
        Performs a reduction operation on this array.

        Parameters
        ----------
        method : numpy.ufunc
            The method to use for performing the reduction.
        axis : Union[int, Iterable[int]], optional
            The axes along which to perform the reduction. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        kwargs : dict
            Any extra arguments to pass to the reduction operation.

        Returns
        -------
        GCXS
            The result of the reduction operation.

        Raises
        ------
        ValueError
            If reducing an all-zero axis would produce a nonzero result.

        See Also
        --------
        numpy.ufunc.reduce : A similar Numpy method.
        COO.reduce : Equivalent operation on COO arrays.
        """
        axis = normalize_axis(axis, self.ndim)
        zero_reduce_result = method.reduce([self.fill_value, self.fill_value], **kwargs)
        reduce_super_ufunc = None

        if not equivalent(zero_reduce_result, self.fill_value):
            reduce_super_ufunc = _reduce_super_ufunc.get(method, None)

            if reduce_super_ufunc is None:
                raise ValueError(
                    "Performing this reduction operation would produce "
                    "a dense result: %s" % str(method)
                )

        if axis is None:
            if reduce_super_ufunc is None:
                temp = method(self.data, self.fill_value)
                val = method.reduce(temp, **kwargs)
            else:
                temp = method(
                    self.data, reduce_super_ufunc(self.fill_value, self.size - self.nnz)
                )
                val = method.reduce(temp, **kwargs)
            if keepdims:
                out = np.array(val)
                return out.reshape(np.ones(self.ndim, dtype=np.intp))
            return val

        if not isinstance(axis, tuple):
            axis = (axis,)

        r = np.arange(self.ndim, dtype=np.intp)
        compressed_axes = [a for a in r if a not in set(axis)]
        x = self.change_compressed_axes(compressed_axes)
        idx = np.diff(x.indptr) != 0
        indptr = x.indptr[:-1][idx]
        indices = (np.arange(x._compressed_shape[0], dtype=np.intp))[idx]
        data = method.reduceat(x.data, indptr, **kwargs)
        counts = x.indptr[1:][idx] - x.indptr[:-1][idx]

        result_fill_value = self.fill_value

        if reduce_super_ufunc is None:
            missing_counts = counts != x._compressed_shape[1]
            data[missing_counts] = method(
                data[missing_counts], self.fill_value, **kwargs
            )
        else:
            data = method(
                data,
                reduce_super_ufunc(self.fill_value, x._compressed_shape[1] - counts),
            ).astype(data.dtype)
            result_fill_value = reduce_super_ufunc(
                self.fill_value, x._compressed_shape[1]
            )

        # prune data
        mask = ~equivalent(data, result_fill_value)
        data = data[mask]
        indices = indices[mask]
        out = GCXS(
            (data, indices, []),
            shape=(x._compressed_shape[0],),
            fill_value=result_fill_value,
            compressed_axes=None,
        )

        out = out.reshape(tuple(self.shape[d] for d in compressed_axes))

        if keepdims:
            shape = list(self.shape)
            for ax in axis:
                shape[ax] = 1
            out = out.reshape(shape)

        if out.ndim == 0:
            return out[()]

        return out

    def sum(self, axis=None, keepdims=False, dtype=None, out=None):
        """
        Performs a sum operation along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to sum. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        GCXS
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.sum` : Equivalent numpy function.
        scipy.sparse.coo_matrix.sum : Equivalent Scipy function.
        COO.sum : Equivalent operation on COO arrays.
        """
        return np.add.reduce(self, out=out, axis=axis, keepdims=keepdims, dtype=dtype)

    def max(self, axis=None, keepdims=False, out=None):
        """
        Maximize along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to maximize. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        GCXS
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.max` : Equivalent numpy function.
        scipy.sparse.coo_matrix.max : Equivalent Scipy function.
        COO.max : Equivalent operation on COO arrays.
        """
        return np.maximum.reduce(self, out=out, axis=axis, keepdims=keepdims)

    amax = max

    def any(self, axis=None, keepdims=False, out=None):
        """
        See if any values along array are ``True``. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to minimize. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.

        Returns
        -------
        GCXS
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.all` : Equivalent numpy function.
        COO.any : Equivalent operation on COO arrays.
        """
        return np.logical_or.reduce(self, out=out, axis=axis, keepdims=keepdims)

    def all(self, axis=None, keepdims=False, out=None):
        """
        See if all values in an array are ``True``. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to minimize. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.

        Returns
        -------
        GCXS
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.all` : Equivalent numpy function.
        COO.all : Equivalent operation on COO arrays.
        """
        return np.logical_and.reduce(self, out=out, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False, out=None):
        """
        Minimize along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to minimize. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        GCXS
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.min` : Equivalent numpy function.
        scipy.sparse.coo_matrix.min : Equivalent Scipy function.
        COO.min : Equivalent operation on COO arrays.
        """
        return np.minimum.reduce(self, out=out, axis=axis, keepdims=keepdims)

    amin = min

    def prod(self, axis=None, keepdims=False, dtype=None, out=None):
        """
        Performs a product operation along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to multiply. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        GCXS
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.prod` : Equivalent numpy function.
        COO.prod : Equivalent operation on COO arrays.
        """
        return np.multiply.reduce(
            self, out=out, axis=axis, keepdims=keepdims, dtype=dtype
        )

    def astype(self, dtype, casting="unsafe", copy=True):
        """
        Copy of the array, cast to a specified type.

        See also
        --------
        scipy.sparse.coo_matrix.astype : SciPy sparse equivalent function
        numpy.ndarray.astype : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.
        """
        if self.dtype == dtype and not copy:
            return self
        # temporary solution
        return GCXS(
            (
                np.array(self.data, copy=copy).astype(dtype),
                np.array(self.indices, copy=copy),
                np.array(self.indptr, copy=copy),
            ),
            shape=self.shape,
            compressed_axes=self.compressed_axes,
            fill_value=self.fill_value,
        )
