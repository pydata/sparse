import copy as _copy
import operator
from collections.abc import Iterable
from functools import reduce
from typing import Union

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from .._coo.common import linear_loc
from .._coo.core import COO
from .._sparse_array import SparseArray
from .._utils import (
    _zero_of_dtype,
    can_store,
    check_compressed_axes,
    check_fill_value,
    equivalent,
    normalize_axis,
)
from .convert import _1d_reshape, _transpose, uncompress_dimension
from .indexing import getitem


def _from_coo(x, compressed_axes=None, idx_dtype=None):
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

    if idx_dtype and not can_store(idx_dtype, max(max(compressed_shape), x.nnz)):
        raise ValueError(
            f"cannot store array with the compressed shape {compressed_shape} and nnz {x.nnz} with dtype {idx_dtype}."
        )

    if not idx_dtype:
        idx_dtype = x.coords.dtype
        if not can_store(idx_dtype, max(max(compressed_shape), x.nnz)):
            idx_dtype = np.min_scalar_type(max(max(compressed_shape), x.nnz))

    # transpose axes, linearize, reshape, and compress
    linear = linear_loc(x.coords[axis_order], reordered_shape)
    order = np.argsort(linear)
    linear = linear[order]
    coords = np.empty((2, x.nnz), dtype=idx_dtype)
    strides = 1
    for i, d in enumerate(compressed_shape[::-1]):
        coords[-(i + 1), :] = (linear // strides) % d
        strides *= d
    indptr = np.empty(row_size + 1, dtype=idx_dtype)
    indptr[0] = 0
    np.cumsum(np.bincount(coords[0], minlength=row_size), out=indptr[1:])
    indices = coords[1]
    data = x.data[order]
    return ((data, indices, indptr), shape, compressed_axes, x.fill_value)


class GCXS(SparseArray, NDArrayOperatorsMixin):
    r"""
    A sparse multidimensional array.

    This is stored in GCXS format, a generalization of the GCRS/GCCS formats
    from [Efficient storage scheme for n-dimensional sparse array: GCRS/GCCS](
    https://ieeexplore.ieee.org/document/7237032). GCXS generalizes the CRS/CCS
    sparse matrix formats.

    For arrays with ndim == 2, GCXS is the same CSR/CSC.
    For arrays with ndim >2, any combination of axes can be compressed,
    significantly reducing storage.

    GCXS consists of 3 arrays. Let the 3 arrays be RO, CO and VL. The first element
    of array RO is the integer 0 and later elements are the number of
    cumulative non-zero elements in each row for GCRS, column for
    GCCS. CO stores column indexes of non-zero elements at each row for GCRS, column for GCCS.
    VL stores the values of the non-zero array elements.

    The superiority of the GCRS/GCCS over traditional (CRS/CCS) is shown by both
    theoretical analysis and experimental results, outlined in the linked research paper.

    Parameters
    ----------
    arg : tuple (data, indices, indptr)
        A tuple of arrays holding the data, indices, and
        index pointers for the nonzero values of the array.
    shape : tuple[int] (COO.ndim,)
        The shape of the array.
    compressed_axes : Iterable[int]
        The axes to compress.
    prune : bool, optional
        A flag indicating whether or not we should prune any fill-values present in
        the data array.
    fill_value: scalar, optional
        The fill value for this array.

    Attributes
    ----------
    data : numpy.ndarray (nnz,)
        An array holding the nonzero values corresponding to `indices`.
    indices : numpy.ndarray (nnz,)
        An array holding the coordinates of every nonzero element along uncompressed dimensions.
    indptr : numpy.ndarray
        An array holding the cumulative sums of the nonzeros along the compressed dimensions.
    shape : tuple[int] (ndim,)
        The dimensions of this array.

    See Also
    --------
    [`sparse.DOK`][] : A mostly write-only sparse array.
    """

    __array_priority__ = 12

    def __init__(
        self,
        arg,
        shape=None,
        compressed_axes=None,
        prune=False,
        fill_value=None,
        idx_dtype=None,
    ):
        from .._common import _is_scipy_sparse_obj

        if _is_scipy_sparse_obj(arg):
            arg = self.from_scipy_sparse(arg)

        if isinstance(arg, np.ndarray):
            (arg, shape, compressed_axes, fill_value) = _from_coo(COO(arg), compressed_axes)

        elif isinstance(arg, COO):
            (arg, shape, compressed_axes, fill_value) = _from_coo(arg, compressed_axes, idx_dtype)

        elif isinstance(arg, GCXS):
            if compressed_axes is not None and arg.compressed_axes != compressed_axes:
                arg = arg.change_compressed_axes(compressed_axes)
            (arg, shape, compressed_axes, fill_value) = (
                (arg.data, arg.indices, arg.indptr),
                arg.shape,
                arg.compressed_axes,
                arg.fill_value,
            )

        if shape is None:
            raise ValueError("missing `shape` argument")

        check_compressed_axes(len(shape), compressed_axes)

        if len(shape) == 1:
            compressed_axes = None

        self.data, self.indices, self.indptr = arg

        if self.data.ndim != 1:
            raise ValueError("data must be a scalar or 1-dimensional.")

        self.shape = shape

        if fill_value is None:
            fill_value = _zero_of_dtype(self.data.dtype)

        self._compressed_axes = tuple(compressed_axes) if isinstance(compressed_axes, Iterable) else None
        self.fill_value = self.data.dtype.type(fill_value)

        if prune:
            self._prune()

    def copy(self, deep=True):
        """Return a copy of the array.

        Parameters
        ----------
        deep : boolean, optional
            If True (default), the internal coords and data arrays are also
            copied. Set to ``False`` to only make a shallow copy.
        """
        return _copy.deepcopy(self) if deep else _copy.copy(self)

    @classmethod
    def from_numpy(cls, x, compressed_axes=None, fill_value=None, idx_dtype=None):
        coo = COO.from_numpy(x, fill_value=fill_value, idx_dtype=idx_dtype)
        return cls.from_coo(coo, compressed_axes, idx_dtype)

    @classmethod
    def from_coo(cls, x, compressed_axes=None, idx_dtype=None):
        (arg, shape, compressed_axes, fill_value) = _from_coo(x, compressed_axes, idx_dtype)
        return cls(arg, shape=shape, compressed_axes=compressed_axes, fill_value=fill_value)

    @classmethod
    def from_scipy_sparse(cls, x, /, *, fill_value=None):
        if x.format == "csc":
            return cls((x.data, x.indices, x.indptr), shape=x.shape, compressed_axes=(1,), fill_value=fill_value)

        x = x.asformat("csr")
        return cls((x.data, x.indices, x.indptr), shape=x.shape, compressed_axes=(0,), fill_value=fill_value)

    @classmethod
    def from_iter(cls, x, shape=None, compressed_axes=None, fill_value=None, idx_dtype=None):
        return cls.from_coo(
            COO.from_iter(x, shape, fill_value),
            compressed_axes,
            idx_dtype,
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
        - [`numpy.ndarray.dtype`][] : Numpy equivalent property.
        - [`scipy.sparse.csr_matrix.dtype`][] : Scipy equivalent property.
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
        - [`sparse.COO.nnz`][] : Equivalent [`sparse.COO`][] array property.
        - [`sparse.DOK.nnz`][] : Equivalent [`sparse.DOK`][] array property.
        - [`numpy.count_nonzero`][] : A similar Numpy function.
        - [`scipy.sparse.coo_matrix.nnz`][] : The Scipy equivalent property.
        """
        return self.data.shape[0]

    @property
    def format(self):
        """
        The storage format of this array.

        Returns
        -------
        str
            The storage format of this array.

        See Also
        -------
        [`scipy.sparse.dok_matrix.format`][] : The Scipy equivalent property.

        Examples
        -------
        >>> import sparse
        >>> s = sparse.random((5, 5), density=0.2, format="dok")
        >>> s.format
        'dok'
        >>> t = sparse.random((5, 5), density=0.2, format="coo")
        >>> t.format
        'coo'
        """
        return "gcxs"

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
        [`numpy.ndarray.nbytes`][] : The equivalent Numpy property.
        """
        return self.data.nbytes + self.indices.nbytes + self.indptr.nbytes

    @property
    def _axis_order(self):
        axis_order = list(self.compressed_axes)
        axis_order.extend(np.setdiff1d(np.arange(len(self.shape)), self.compressed_axes))
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

    @property
    def mT(self):
        if self.ndim < 2:
            raise ValueError("Cannot compute matrix transpose if `ndim < 2`.")

        axis = list(range(self.ndim))
        axis[-1], axis[-2] = axis[-2], axis[-1]

        return self.transpose(axis)

    def __str__(self):
        summary = (
            f"<GCXS: shape={self.shape}, dtype={self.dtype}, nnz={self.nnz}, fill_value={self.fill_value}, "
            f"compressed_axes={self.compressed_axes}>"
        )
        return self._str_impl(summary)

    __repr__ = __str__

    __getitem__ = getitem

    def _reduce_calc(self, method, axis, keepdims=False, **kwargs):
        if axis[0] is None or np.array_equal(axis, np.arange(self.ndim, dtype=np.intp)):
            x = self.flatten().tocoo()
            out = x.reduce(method, axis=None, keepdims=keepdims, **kwargs)
            if keepdims:
                return (out.reshape(np.ones(self.ndim, dtype=np.intp)),)
            return (out,)

        r = np.arange(self.ndim, dtype=np.intp)
        compressed_axes = [a for a in r if a not in set(axis)]
        x = self.change_compressed_axes(compressed_axes)
        idx = np.diff(x.indptr) != 0
        indptr = x.indptr[:-1][idx]
        indices = (np.arange(x._compressed_shape[0], dtype=self.indptr.dtype))[idx]
        data = method.reduceat(x.data, indptr, **kwargs)
        counts = x.indptr[1:][idx] - x.indptr[:-1][idx]
        arr_attrs = (x, compressed_axes, indices)
        n_cols = x._compressed_shape[1]
        return (data, counts, axis, n_cols, arr_attrs)

    def _reduce_return(self, data, arr_attrs, result_fill_value):
        x, compressed_axes, indices = arr_attrs
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
        return out.reshape(tuple(self.shape[d] for d in compressed_axes))

    def change_compressed_axes(self, new_compressed_axes):
        """
        Returns a new array with specified compressed axes. This operation is similar to converting
        a scipy.sparse.csc_matrix to a scipy.sparse.csr_matrix.

        Returns
        -------
        GCXS
            A new instance of the input array with compression along the specified dimensions.
        """
        if new_compressed_axes == self.compressed_axes:
            return self

        if self.ndim == 1:
            raise NotImplementedError("no axes to compress for 1d array")

        new_compressed_axes = tuple(
            normalize_axis(new_compressed_axes[i], self.ndim) for i in range(len(new_compressed_axes))
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
        Convert this [`sparse.GCXS`][] array to a [`sparse.COO`][].

        Returns
        -------
        sparse.COO
            The converted COO array.
        """
        if self.ndim == 0:
            return COO(
                np.array([]),
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
        Convert this [`sparse.GCXS`][] array to a dense [`numpy.ndarray`][]. Note that
        this may take a large amount of memory if the [`sparse.GCXS`][] object's `shape`
        is large.

        Returns
        -------
        numpy.ndarray
            The converted dense array.

        See Also
        --------
        - [`sparse.DOK.todense`][] : Equivalent [`sparse.DOK`][] array method.
        - [`sparse.COO.todense`][] : Equivalent [`sparse.COO`][] array method.
        - [`scipy.sparse.coo_matrix.todense`][] : Equivalent Scipy method.

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
        from .. import DOK

        return DOK.from_coo(self.tocoo())  # probably a temporary solution

    def to_scipy_sparse(self, accept_fv=None):
        """
        Converts this [`sparse.GCXS`][] object into a [`scipy.sparse.csr_matrix`][] or [`scipy.sparse.csc_matrix`][].

        Parameters
        ----------
        accept_fv : scalar or list of scalar, optional
            The list of accepted fill-values. The default accepts only zero.

        Returns
        -------
        scipy.sparse.csr_matrix or scipy.sparse.csc_matrix
            The converted Scipy sparse matrix.

        Raises
        ------
        ValueError
            If the array is not two-dimensional.
        ValueError
            If all the array doesn't zero fill-values.
        """
        import scipy.sparse

        check_fill_value(self, accept_fv=accept_fv)
        if self.ndim != 2:
            raise ValueError("Can only convert a 2-dimensional array to a Scipy sparse matrix.")

        if 0 in self.compressed_axes:
            return scipy.sparse.csr_matrix((self.data, self.indices, self.indptr), shape=self.shape)

        return scipy.sparse.csc_matrix((self.data, self.indices, self.indptr), shape=self.shape)

    def asformat(self, format, **kwargs):
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
        from .._utils import convert_format

        format = convert_format(format)
        ret = None

        if format == "coo":
            ret = self.tocoo()
        elif format == "dok":
            ret = self.todok()
        elif format == "csr":
            ret = CSR(self)
        elif format == "csc":
            ret = CSC(self)
        elif format == "gcxs":
            compressed_axes = kwargs.pop("compressed_axes", self.compressed_axes)
            return self.change_compressed_axes(compressed_axes)

        if len(kwargs) != 0:
            raise TypeError(f"Invalid keyword arguments provided: {kwargs}")

        if ret is None:
            raise NotImplementedError(f"The given format is not supported: {format}")

        return ret

    def maybe_densify(self, max_size=1000, min_density=0.25):
        """
        Converts this [`sparse.GCXS`][] array to a [`numpy.ndarray`][] if not too
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
        - [sparse.GCXS.todense][]: Converts to Numpy function without checking the cost.
        - [sparse.COO.maybe_densify][]: The equivalent COO function.

        Raises
        -------
        ValueError
            If the returned array would be too large.
        """

        if self.size > max_size and self.density < min_density:
            raise ValueError("Operation would require converting large sparse array to dense")

        return self.todense()

    def flatten(self, order="C"):
        """
        Returns a new [`sparse.GCXS`][] array that is a flattened version of this array.

        Returns
        -------
        GCXS
            The flattened output array.

        Notes
        -----
        The `order` parameter is provided just for compatibility with
        Numpy and isn't actually supported.
        """
        if order not in {"C", None}:
            raise NotImplementedError("The `order` parameter is not supported.")

        return self.reshape(-1)

    def reshape(self, shape, order="C", compressed_axes=None):
        """
        Returns a new [`sparse.GCXS`][] array that is a reshaped version of this array.

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
        - [`numpy.ndarray.reshape`][] : The equivalent Numpy function.
        - [sparse.COO.reshape][] : The equivalent COO function.

        Notes
        -----
        The `order` parameter is provided just for compatibility with
        Numpy and isn't actually supported.

        """
        shape = tuple(shape) if isinstance(shape, Iterable) else (shape,)
        if order not in {"C", None}:
            raise NotImplementedError("The 'order' parameter is not supported")
        if any(d == -1 for d in shape):
            extra = int(self.size / np.prod([d for d in shape if d != -1]))
            shape = tuple([d if d != -1 else extra for d in shape])

        if self.shape == shape:
            return self

        if self.size != reduce(operator.mul, shape, 1):
            raise ValueError(f"cannot reshape array of size {self.size} into shape {shape}")
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

    @property
    def compressed_axes(self):
        return self._compressed_axes

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
        - [`sparse.GCXS.T`][] : A quick property to reverse the order of the axes.
        - [`numpy.ndarray.transpose`][] : Numpy equivalent function.
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
        scipy.sparse.csr_matrix.transpose : Scipy equivalent function.
        scipy.sparse.csc_matrix.transpose : Scipy equivalent function.
        numpy.ndarray.transpose : Numpy equivalent function.
        """
        if self.ndim != 2:
            raise ValueError(f"cannot perform 2d transpose on array with dimension {self.ndim}")

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
        Performs the equivalent of `x.dot(y)` for [`sparse.GCXS`][].

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
        - [`sparse.dot`][] : Equivalent function for two arguments.
        - [`numpy.dot`][] : Numpy equivalent function.
        - [`scipy.sparse.coo_matrix.dot`][] : Scipy equivalent function.
        """
        from .._common import dot

        return dot(self, other)

    def __matmul__(self, other):
        from .._common import matmul

        try:
            return matmul(self, other)
        except NotImplementedError:
            return NotImplemented

    def __rmatmul__(self, other):
        from .._common import matmul

        try:
            return matmul(other, self)
        except NotImplementedError:
            return NotImplemented

    def _prune(self):
        """
        Prunes data so that if any fill-values are present, they are removed
        from both indices and data.

        Examples
        --------
        >>> coords = np.array([[0, 1, 2, 3]])
        >>> data = np.array([1, 0, 1, 2])
        >>> s = COO(coords, data, shape=(4,)).asformat("gcxs")
        >>> s._prune()
        >>> s.nnz
        3
        """
        mask = ~equivalent(self.data, self.fill_value)
        self.data = self.data[mask]
        if len(self.indptr):
            coords = np.stack((uncompress_dimension(self.indptr), self.indices))
            coords = coords[:, mask]
            self.indices = coords[1]
            row_size = self._compressed_shape[0]
            indptr = np.empty(row_size + 1, dtype=self.indptr.dtype)
            indptr[0] = 0
            np.cumsum(np.bincount(coords[0], minlength=row_size), out=indptr[1:])
            self.indptr = indptr
        else:
            self.indices = self.indices[mask]

    def isinf(self):
        return self.tocoo().isinf().asformat("gcxs", compressed_axes=self.compressed_axes)

    def isnan(self):
        return self.tocoo().isnan().asformat("gcxs", compressed_axes=self.compressed_axes)


class _Compressed2d(GCXS):
    class_compressed_axes: tuple[int]

    def __init__(self, arg, shape=None, compressed_axes=None, prune=False, fill_value=0):
        if not hasattr(arg, "shape") and shape is None:
            raise ValueError("missing `shape` argument")
        if shape is not None and hasattr(arg, "shape"):
            raise NotImplementedError("Cannot change shape in constructor")
        nd = len(shape if shape is not None else arg.shape)
        if nd != 2:
            raise ValueError(f"{type(self).__name__} must be 2-d, passed {nd}-d shape.")

        super().__init__(
            arg,
            shape=shape,
            compressed_axes=compressed_axes,
            prune=prune,
            fill_value=fill_value,
        )

    def __str__(self):
        summary = (
            f"<{type(self).__name__}: shape={self.shape}, dtype={self.dtype}, nnz={self.nnz}, "
            f"fill_value={self.fill_value}>"
        )
        return self._str_impl(summary)

    __repr__ = __str__

    @property
    def ndim(self) -> int:
        return 2

    @classmethod
    def from_numpy(cls, x, fill_value=0, idx_dtype=None):
        coo = COO.from_numpy(x, fill_value=fill_value, idx_dtype=idx_dtype)
        return cls.from_coo(coo, cls.class_compressed_axes, idx_dtype)


class CSR(_Compressed2d):
    """
    The CSR or CRS scheme stores a n-dimensional array using n+1 one-dimensional arrays.
    The 3 arrays are same as GCRS. The remaining n-2 arrays are for storing the indices of
    the non-zero values of the sparse matrix. CSR is simply the transpose of CSC.

    Sparse supports 2-D CSR.
    """

    class_compressed_axes: tuple[int] = (0,)

    def __init__(self, arg, shape=None, compressed_axes=class_compressed_axes, prune=False, fill_value=0):
        if compressed_axes != self.class_compressed_axes:
            raise ValueError(f"CSR only accepts rows as compressed axis but got: {compressed_axes}")
        super().__init__(arg, shape=shape, compressed_axes=compressed_axes, fill_value=fill_value)

    @classmethod
    def from_scipy_sparse(cls, x, /, *, fill_value=None):
        x = x.asformat("csr", copy=False)
        return cls((x.data, x.indices, x.indptr), shape=x.shape, fill_value=fill_value)

    def transpose(self, axes: None = None, copy: bool = False) -> Union["CSC", "CSR"]:
        axes = normalize_axis(axes, self.ndim)
        if axes not in [(0, 1), (1, 0), None]:
            raise ValueError(f"Invalid transpose axes: {axes}")
        if copy:
            self = self.copy()
        if axes == (0, 1):
            return self
        return CSC((self.data, self.indices, self.indptr), self.shape[::-1], fill_value=self.fill_value)


class CSC(_Compressed2d):
    """
    The CSC or CCS scheme stores a n-dimensional array using n+1 one-dimensional arrays.
    The 3 arrays are same as GCCS. The remaining n-2 arrays are for storing the indices of
    the non-zero values of the sparse matrix. CSC is simply the transpose of CSR.

    Sparse supports 2-D CSC.
    """

    class_compressed_axes: tuple[int] = (1,)

    def __init__(self, arg, shape=None, compressed_axes=class_compressed_axes, prune=False, fill_value=0):
        if compressed_axes != self.class_compressed_axes:
            raise ValueError(f"CSC only accepts columns as compressed axis but got: {compressed_axes}")
        super().__init__(arg, shape=shape, compressed_axes=compressed_axes, fill_value=fill_value)

    @classmethod
    def from_scipy_sparse(cls, x, /, *, fill_value=None):
        x = x.asformat("csc", copy=False)
        return cls((x.data, x.indices, x.indptr), shape=x.shape, fill_value=fill_value)

    def transpose(self, axes: None = None, copy: bool = False) -> Union["CSC", "CSR"]:
        axes = normalize_axis(axes, self.ndim)
        if axes not in [(0, 1), (1, 0), None]:
            raise ValueError(f"Invalid transpose axes: {axes}")
        if copy:
            self = self.copy()
        if axes == (0, 1):
            return self
        return CSR((self.data, self.indices, self.indptr), self.shape[::-1], fill_value=self.fill_value)
