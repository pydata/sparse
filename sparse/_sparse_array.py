from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from numbers import Integral
from typing import Callable
import operator
import functools

import numba
import numpy as np
import scipy.sparse as ss

from ._coo.umath import elemwise
from ._utils import _zero_of_dtype, html_table, equivalent, normalize_axis

_reduce_super_ufunc = {np.add: np.multiply, np.multiply: np.power}


class SparseArray:
    """
    An abstract base class for all the sparse array classes.

    Attributes
    ----------
    dtype : numpy.dtype
        The data type of this array.
    fill_value : scalar
        The fill value of this array.
    """

    __metaclass__ = ABCMeta

    def __init__(self, shape, fill_value=None):
        if not isinstance(shape, Iterable):
            shape = (shape,)

        if not all(isinstance(l, Integral) and int(l) >= 0 for l in shape):
            raise ValueError(
                "shape must be an non-negative integer or a tuple "
                "of non-negative integers."
            )

        self.shape = tuple(int(l) for l in shape)

        if fill_value is not None:
            if not hasattr(fill_value, "dtype") or fill_value.dtype != self.dtype:
                self.fill_value = self.dtype.type(fill_value)
            else:
                self.fill_value = fill_value
        else:
            self.fill_value = _zero_of_dtype(self.dtype)

    dtype = None

    @property
    @abstractmethod
    def nnz(self):
        """
        The number of nonzero elements in this array. Note that any duplicates in
        :code:`coords` are counted multiple times. To avoid this, call :obj:`COO.sum_duplicates`.

        Returns
        -------
        int
            The number of nonzero elements in this array.

        See Also
        --------
        DOK.nnz : Equivalent :obj:`DOK` array property.
        numpy.count_nonzero : A similar Numpy function.
        scipy.sparse.coo_matrix.nnz : The Scipy equivalent property.

        Examples
        --------
        >>> import numpy as np
        >>> from sparse import COO
        >>> x = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 0])
        >>> np.count_nonzero(x)
        6
        >>> s = COO.from_numpy(x)
        >>> s.nnz
        6
        >>> np.count_nonzero(x) == s.nnz
        True
        """

    @property
    def ndim(self):
        """
        The number of dimensions of this array.

        Returns
        -------
        int
            The number of dimensions of this array.

        See Also
        --------
        DOK.ndim : Equivalent property for :obj:`DOK` arrays.
        numpy.ndarray.ndim : Numpy equivalent property.

        Examples
        --------
        >>> from sparse import COO
        >>> import numpy as np
        >>> x = np.random.rand(1, 2, 3, 1, 2)
        >>> s = COO.from_numpy(x)
        >>> s.ndim
        5
        >>> s.ndim == x.ndim
        True
        """
        return len(self.shape)

    @property
    def size(self):
        """
        The number of all elements (including zeros) in this array.

        Returns
        -------
        int
            The number of elements.

        See Also
        --------
        numpy.ndarray.size : Numpy equivalent property.

        Examples
        --------
        >>> from sparse import COO
        >>> import numpy as np
        >>> x = np.zeros((10, 10))
        >>> s = COO.from_numpy(x)
        >>> s.size
        100
        """
        # We use this instead of np.prod because np.prod
        # returns a float64 for an empty shape.
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    def density(self):
        """
        The ratio of nonzero to all elements in this array.

        Returns
        -------
        float
            The ratio of nonzero to all elements.

        See Also
        --------
        COO.size : Number of elements.
        COO.nnz : Number of nonzero elements.

        Examples
        --------
        >>> import numpy as np
        >>> from sparse import COO
        >>> x = np.zeros((8, 8))
        >>> x[0, :] = 1
        >>> s = COO.from_numpy(x)
        >>> s.density
        0.125
        """
        return self.nnz / self.size

    def _repr_html_(self):
        """
        Diagnostic report about this array.
        Renders in Jupyter.
        """
        return html_table(self)

    @abstractmethod
    def asformat(self, format):
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

    @abstractmethod
    def todense(self):
        """
        Convert this :obj:`SparseArray` array to a dense :obj:`numpy.ndarray`. Note that
        this may take a large amount of memory and time.

        Returns
        -------
        numpy.ndarray
            The converted dense array.

        See Also
        --------
        DOK.todense : Equivalent :obj:`DOK` array method.
        COO.todense : Equivalent :obj:`COO` array method.
        scipy.sparse.coo_matrix.todense : Equivalent Scipy method.

        Examples
        --------
        >>> import sparse
        >>> x = np.random.randint(100, size=(7, 3))
        >>> s = sparse.COO.from_numpy(x)
        >>> x2 = s.todense()
        >>> np.array_equal(x, x2)
        True
        """

    def __array__(self, *args, **kwargs):
        from ._settings import AUTO_DENSIFY

        if not AUTO_DENSIFY:
            raise RuntimeError(
                "Cannot convert a sparse array to dense automatically. "
                "To manually densify, use the todense method."
            )

        return np.asarray(self.todense(), *args, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        import sparse as module

        sparse_func = None
        try:
            submodules = getattr(func, "__module__", "numpy").split(".")[1:]
            for submodule in submodules:
                module = getattr(module, submodule)
            sparse_func = getattr(module, func.__name__)
        except AttributeError:
            pass
        else:
            return sparse_func(*args, **kwargs)

        try:
            sparse_func = getattr(type(self), func.__name__)
        except AttributeError:
            pass

        if (
            not isinstance(sparse_func, Callable)
            and len(args) == 1
            and len(kwargs) == 0
        ):
            try:
                return getattr(self, func.__name__)
            except AttributeError:
                pass

        if sparse_func is None:
            return NotImplemented

        return sparse_func(*args, **kwargs)

    @staticmethod
    def _reduce(method, *args, **kwargs):
        assert len(args) == 1

        self = args[0]
        if isinstance(self, ss.spmatrix):
            self = type(self).from_scipy_sparse(self)

        return self.reduce(method, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.pop("out", None)
        if out is not None and not all(isinstance(x, type(self)) for x in out):
            return NotImplemented

        if getattr(ufunc, "signature", None) is not None:
            return self.__array_function__(
                ufunc, (np.ndarray, type(self)), inputs, kwargs
            )

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

        if method == "__call__":
            result = elemwise(ufunc, *inputs, **kwargs)
        elif method == "reduce":
            result = SparseArray._reduce(ufunc, *inputs, **kwargs)
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

        Examples
        --------
        You can use the :obj:`COO.reduce` method to apply a reduction operation to
        any Numpy :code:`ufunc`.

        >>> from sparse import COO
        >>> x = np.ones((5, 5), dtype=np.int)
        >>> s = COO.from_numpy(x)
        >>> s2 = s.reduce(np.add, axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([5, 5, 5, 5, 5])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        reduction.

        >>> s3 = s.reduce(np.add, axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        You can also pass in any keyword argument that :obj:`numpy.ufunc.reduce` supports.
        For example, :code:`dtype`. Note that :code:`out` isn't supported.

        >>> s4 = s.reduce(np.add, axis=1, dtype=np.float16)
        >>> s4.dtype
        dtype('float16')

        By default, this reduces the array by only the first axis.

        >>> s.reduce(np.add)
        <COO: shape=(5,), dtype=int64, nnz=5, fill_value=0>
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
            if hasattr(self, "indptr"):  # gcxs
                if reduce_super_ufunc is None:
                    temp = method(self.data, self.fill_value)
                    val = method.reduce(temp, **kwargs)
                else:
                    temp = method(
                        self.data,
                        reduce_super_ufunc(self.fill_value, self.size - self.nnz),
                    )
                    val = method.reduce(temp, **kwargs)
                if keepdims:
                    out = np.array(val)
                    return out.reshape(np.ones(self.ndim, dtype=np.intp))
                return val
            else:  # coo
                axis = tuple(range(self.ndim))

        if not isinstance(axis, tuple):
            axis = (axis,)

        if hasattr(self, "indptr"):  # gcxs
            r = np.arange(self.ndim, dtype=np.intp)
            compressed_axes = [a for a in r if a not in set(axis)]
            x = self.change_compressed_axes(compressed_axes)
            idx = np.diff(x.indptr) != 0
            indptr = x.indptr[:-1][idx]
            indices = (np.arange(x._compressed_shape[0], dtype=np.intp))[idx]
            data = method.reduceat(x.data, indptr, **kwargs)
            counts = x.indptr[1:][idx] - x.indptr[:-1][idx]
            n_cols = x._compressed_shape[1]

        else:  # coo
            axis = tuple(a if a >= 0 else a + self.ndim for a in axis)

            neg_axis = tuple(ax for ax in range(self.ndim) if ax not in set(axis))

            a = self.transpose(neg_axis + axis)
            a = a.reshape(
                (
                    np.prod([self.shape[d] for d in neg_axis], dtype=np.intp),
                    np.prod([self.shape[d] for d in axis], dtype=np.intp),
                )
            )
            data, inv_idx, counts = _grouped_reduce(
                a.data, a.coords[0], method, **kwargs
            )
            n_cols = a.shape[1]

        result_fill_value = self.fill_value

        if reduce_super_ufunc is None:
            missing_counts = counts != n_cols
            data[missing_counts] = method(
                data[missing_counts], self.fill_value, **kwargs
            )
        else:
            data = method(
                data, reduce_super_ufunc(self.fill_value, n_cols - counts),
            ).astype(data.dtype)
            result_fill_value = reduce_super_ufunc(self.fill_value, n_cols)

        if hasattr(self, "indptr"):  # gcxs
            # prune data
            mask = ~equivalent(data, result_fill_value)
            data = data[mask]
            indices = indices[mask]
            out = type(self)(
                (data, indices, []),
                shape=(x._compressed_shape[0],),
                fill_value=result_fill_value,
                compressed_axes=None,
            )
            out = out.reshape(tuple(self.shape[d] for d in compressed_axes))
        else:  # coo
            coords = a.coords[0:1, inv_idx]
            out = type(self)(
                coords,
                data,
                shape=(a.shape[0],),
                has_duplicates=False,
                sorted=True,
                prune=True,
                fill_value=result_fill_value,
            )
            out = out.reshape(tuple(self.shape[d] for d in neg_axis))

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
        SparseArray
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.sum` : Equivalent numpy function.
        scipy.sparse.coo_matrix.sum : Equivalent Scipy function.
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
        SparseArray
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.max` : Equivalent numpy function.
        scipy.sparse.coo_matrix.max : Equivalent Scipy function.
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
        SparseArray
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.all` : Equivalent numpy function.
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
        SparseArray
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.all` : Equivalent numpy function.
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
        SparseArray
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.min` : Equivalent numpy function.
        scipy.sparse.coo_matrix.min : Equivalent Scipy function.
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
        SparseArray
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.prod` : Equivalent numpy function.
        """
        return np.multiply.reduce(
            self, out=out, axis=axis, keepdims=keepdims, dtype=dtype
        )


@numba.jit(nopython=True, nogil=True)  # pragma: no cover
def _calc_counts_invidx(groups):
    inv_idx = []
    counts = []

    if len(groups) == 0:
        return (
            np.array(inv_idx, dtype=groups.dtype),
            np.array(counts, dtype=groups.dtype),
        )

    inv_idx.append(0)

    last_group = groups[0]
    for i in range(1, len(groups)):
        if groups[i] != last_group:
            counts.append(i - inv_idx[-1])
            inv_idx.append(i)
            last_group = groups[i]

    counts.append(len(groups) - inv_idx[-1])

    return (np.array(inv_idx, dtype=groups.dtype), np.array(counts, dtype=groups.dtype))


def _grouped_reduce(x, groups, method, **kwargs):
    """
    Performs a :code:`ufunc` grouped reduce.

    Parameters
    ----------
    x : np.ndarray
        The data to reduce.
    groups : np.ndarray
        The groups the data belongs to. The groups must be
        contiguous.
    method : np.ufunc
        The :code:`ufunc` to use to perform the reduction.
    kwargs : dict
        The kwargs to pass to the :code:`ufunc`'s :code:`reduceat`
        function.

    Returns
    -------
    result : np.ndarray
        The result of the grouped reduce operation.
    inv_idx : np.ndarray
        The index of the first element where each group is found.
    counts : np.ndarray
        The number of elements in each group.
    """
    # Partial credit to @shoyer
    # Ref: https://gist.github.com/shoyer/f538ac78ae904c936844
    inv_idx, counts = _calc_counts_invidx(groups)
    result = method.reduceat(x, inv_idx, **kwargs)
    return result, inv_idx, counts
