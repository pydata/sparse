from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from numbers import Integral
from typing import Callable
import operator
from functools import reduce

import numpy as np
import scipy.sparse as ss

from ._umath import elemwise
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
        return reduce(operator.mul, self.shape, 1)

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

    def _make_shallow_copy_of(self, other):
        self.__dict__ = other.__dict__.copy()

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
        **kwargs : dict
            Any extra arguments to pass to the reduction operation.

        See Also
        --------
        numpy.ufunc.reduce : A similar Numpy method.
        COO.reduce : This method implemented on COO arrays.
        GCXS.reduce : This method implemented on GCXS arrays.
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

        if not isinstance(axis, tuple):
            axis = (axis,)
        out = self._reduce_calc(method, axis, keepdims, **kwargs)
        if len(out) == 1:
            return out[0]
        data, counts, axis, n_cols, arr_attrs = out
        result_fill_value = self.fill_value
        if reduce_super_ufunc is None:
            missing_counts = counts != n_cols
            data[missing_counts] = method(
                data[missing_counts], self.fill_value, **kwargs
            )
        else:
            data = method(
                data,
                reduce_super_ufunc(self.fill_value, n_cols - counts),
            ).astype(data.dtype)
            result_fill_value = reduce_super_ufunc(self.fill_value, n_cols)

        out = self._reduce_return(data, arr_attrs, result_fill_value)

        if keepdims:
            shape = list(self.shape)
            for ax in axis:
                shape[ax] = 1
            out = out.reshape(shape)

        if out.ndim == 0:
            return out[()]

        return out

    def _reduce_calc(self, method, axis, keepdims, **kwargs):
        raise NotImplementedError

    def _reduce_return(self, data, arr_attrs, result_fill_value):
        raise NotImplementedError

    def sum(self, axis=None, keepdims=False, dtype=None, out=None):
        """
        Performs a sum operation along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to sum. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype : numpy.dtype
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
        out : numpy.dtype
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
        :obj:`numpy.any` : Equivalent numpy function.
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
        out : numpy.dtype
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
        dtype : numpy.dtype
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

    def round(self, decimals=0, out=None):
        """
        Evenly round to the given number of decimals.

        See Also
        --------
        :obj:`numpy.round` :
            NumPy equivalent ufunc.
        :obj:`COO.elemwise` :
            Apply an arbitrary element-wise function to one or two
            arguments.
        """
        if out is not None and not isinstance(out, tuple):
            out = (out,)
        return self.__array_ufunc__(
            np.round, "__call__", self, decimals=decimals, out=out
        )

    round_ = round

    def clip(self, min=None, max=None, out=None):
        """
        Clip (limit) the values in the array.

        Return an array whose values are limited to ``[min, max]``. One of min
        or max must be given.

        See Also
        --------
        sparse.clip : For full documentation and more details.
        numpy.clip : Equivalent NumPy function.
        """
        if min is None and max is None:
            raise ValueError("One of max or min must be given.")
        if out is not None and not isinstance(out, tuple):
            out = (out,)
        return self.__array_ufunc__(
            np.clip, "__call__", self, a_min=min, a_max=max, out=out
        )

    def astype(self, dtype, casting="unsafe", copy=True):
        """
        Copy of the array, cast to a specified type.

        See Also
        --------
        scipy.sparse.coo_matrix.astype :
            SciPy sparse equivalent function
        numpy.ndarray.astype :
            NumPy equivalent ufunc.
        :obj:`COO.elemwise` :
            Apply an arbitrary element-wise function to one or two
            arguments.
        """
        # this matches numpy's behavior
        if self.dtype == dtype and not copy:
            return self
        return self.__array_ufunc__(
            np.ndarray.astype, "__call__", self, dtype=dtype, copy=copy, casting=casting
        )

    def mean(self, axis=None, keepdims=False, dtype=None, out=None):
        """
        Compute the mean along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to compute the mean. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype : numpy.dtype
            The data type of the output array.

        Returns
        -------
        SparseArray
            The reduced output sparse array.

        See Also
        --------
        numpy.ndarray.mean : Equivalent numpy method.
        scipy.sparse.coo_matrix.mean : Equivalent Scipy method.

        Notes
        -----
        * This function internally calls :obj:`COO.sum_duplicates` to bring the
          array into canonical form.
        * The :code:`out` parameter is provided just for compatibility with
          Numpy and isn't actually supported.

        Examples
        --------
        You can use :obj:`COO.mean` to compute the mean of an array across any
        dimension.

        >>> from sparse import COO
        >>> x = np.array([[1, 2, 0, 0],
        ...               [0, 1, 0, 0]], dtype='i8')
        >>> s = COO.from_numpy(x)
        >>> s2 = s.mean(axis=1)
        >>> s2.todense()  # doctest: +SKIP
        array([0.5, 1.5, 0., 0.])

        You can also use the :code:`keepdims` argument to keep the dimensions
        after the mean.

        >>> s3 = s.mean(axis=0, keepdims=True)
        >>> s3.shape
        (1, 4)

        You can pass in an output datatype, if needed.

        >>> s4 = s.mean(axis=0, dtype=np.float16)
        >>> s4.dtype
        dtype('float16')

        By default, this reduces the array down to one number, computing the
        mean along all axes.

        >>> s.mean()
        0.5
        """

        if axis is None:
            axis = tuple(range(self.ndim))
        elif not isinstance(axis, tuple):
            axis = (axis,)
        den = reduce(operator.mul, (self.shape[i] for i in axis), 1)

        if dtype is None:
            if issubclass(self.dtype.type, (np.integer, np.bool_)):
                dtype = inter_dtype = np.dtype("f8")
            else:
                dtype = self.dtype
                inter_dtype = (
                    np.dtype("f4") if issubclass(dtype.type, np.float16) else dtype
                )
        else:
            inter_dtype = dtype

        num = self.sum(axis=axis, keepdims=keepdims, dtype=inter_dtype)

        if num.ndim:
            out = np.true_divide(num, den, casting="unsafe")
            return out.astype(dtype) if out.dtype != dtype else out
        return np.divide(num, den, dtype=dtype, out=out)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """
        Compute the variance along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to compute the variance. Uses all axes by default.
        dtype : numpy.dtype, optional
            The output datatype.
        out : SparseArray, optional
            The array to write the output to.
        ddof : int
            The degrees of freedom.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.

        Returns
        -------
        SparseArray
            The reduced output sparse array.

        See Also
        --------
        numpy.ndarray.var : Equivalent numpy method.

        Notes
        -----
        * This function internally calls :obj:`COO.sum_duplicates` to bring the
          array into canonical form.

        Examples
        --------
        You can use :obj:`COO.var` to compute the variance of an array across any
        dimension.

        >>> from sparse import COO
        >>> x = np.array([[1, 2, 0, 0],
        ...               [0, 1, 0, 0]], dtype='i8')
        >>> s = COO.from_numpy(x)
        >>> s2 = s.var(axis=1)
        >>> s2.todense()  # doctest: +SKIP
        array([0.6875, 0.1875])

        You can also use the :code:`keepdims` argument to keep the dimensions
        after the variance.

        >>> s3 = s.var(axis=0, keepdims=True)
        >>> s3.shape
        (1, 4)

        You can pass in an output datatype, if needed.

        >>> s4 = s.var(axis=0, dtype=np.float16)
        >>> s4.dtype
        dtype('float16')

        By default, this reduces the array down to one number, computing the
        variance along all axes.

        >>> s.var()
        0.5
        """
        axis = normalize_axis(axis, self.ndim)

        if axis is None:
            axis = tuple(range(self.ndim))

        if not isinstance(axis, tuple):
            axis = (axis,)

        rcount = reduce(operator.mul, (self.shape[a] for a in axis), 1)
        # Make this warning show up on top.
        if ddof >= rcount:
            warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning)

        # Cast bool, unsigned int, and int to float64 by default
        if dtype is None and issubclass(self.dtype.type, (np.integer, np.bool_)):
            dtype = np.dtype("f8")

        arrmean = self.sum(axis, dtype=dtype, keepdims=True)
        np.divide(arrmean, rcount, out=arrmean)
        x = self - arrmean
        if issubclass(self.dtype.type, np.complexfloating):
            x = x.real * x.real + x.imag * x.imag
        else:
            x = np.multiply(x, x, out=x)

        ret = x.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

        # Compute degrees of freedom and make sure it is not negative.
        rcount = max([rcount - ddof, 0])

        ret = ret[...]
        np.divide(ret, rcount, out=ret, casting="unsafe")
        return ret[()]

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """
        Compute the standard deviation along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to compute the standard deviation. Uses
            all axes by default.
        dtype : numpy.dtype, optional
            The output datatype.
        out : SparseArray, optional
            The array to write the output to.
        ddof : int
            The degrees of freedom.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.

        Returns
        -------
        SparseArray
            The reduced output sparse array.

        See Also
        --------
        numpy.ndarray.std : Equivalent numpy method.

        Notes
        -----
        * This function internally calls :obj:`COO.sum_duplicates` to bring the
          array into canonical form.

        Examples
        --------
        You can use :obj:`COO.std` to compute the standard deviation of an array
        across any dimension.

        >>> from sparse import COO
        >>> x = np.array([[1, 2, 0, 0],
        ...               [0, 1, 0, 0]], dtype='i8')
        >>> s = COO.from_numpy(x)
        >>> s2 = s.std(axis=1)
        >>> s2.todense()  # doctest: +SKIP
        array([0.8291562, 0.4330127])

        You can also use the :code:`keepdims` argument to keep the dimensions
        after the standard deviation.

        >>> s3 = s.std(axis=0, keepdims=True)
        >>> s3.shape
        (1, 4)

        You can pass in an output datatype, if needed.

        >>> s4 = s.std(axis=0, dtype=np.float16)
        >>> s4.dtype
        dtype('float16')

        By default, this reduces the array down to one number, computing the
        standard deviation along all axes.

        >>> s.std()  # doctest: +SKIP
        0.7071067811865476
        """
        ret = self.var(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

        ret = np.sqrt(ret)
        return ret

    @property
    def real(self):
        """The real part of the array.

        Examples
        --------
        >>> from sparse import COO
        >>> x = COO.from_numpy([1 + 0j, 0 + 1j])
        >>> x.real.todense()  # doctest: +SKIP
        array([1., 0.])
        >>> x.real.dtype
        dtype('float64')

        Returns
        -------
        out : SparseArray
            The real component of the array elements. If the array dtype is
            real, the dtype of the array is used for the output. If the array
            is complex, the output dtype is float.

        See Also
        --------
        numpy.ndarray.real : NumPy equivalent attribute.
        numpy.real : NumPy equivalent function.
        """
        return self.__array_ufunc__(np.real, "__call__", self)

    @property
    def imag(self):
        """The imaginary part of the array.

        Examples
        --------
        >>> from sparse import COO
        >>> x = COO.from_numpy([1 + 0j, 0 + 1j])
        >>> x.imag.todense()  # doctest: +SKIP
        array([0., 1.])
        >>> x.imag.dtype
        dtype('float64')

        Returns
        -------
        out : SparseArray
            The imaginary component of the array elements. If the array dtype
            is real, the dtype of the array is used for the output. If the
            array is complex, the output dtype is float.

        See Also
        --------
        numpy.ndarray.imag : NumPy equivalent attribute.
        numpy.imag : NumPy equivalent function.
        """
        return self.__array_ufunc__(np.imag, "__call__", self)

    def conj(self):
        """Return the complex conjugate, element-wise.

        The complex conjugate of a complex number is obtained by changing the
        sign of its imaginary part.

        Examples
        --------
        >>> from sparse import COO
        >>> x = COO.from_numpy([1 + 2j, 2 - 1j])
        >>> res = x.conj()
        >>> res.todense()  # doctest: +SKIP
        array([1.-2.j, 2.+1.j])
        >>> res.dtype
        dtype('complex128')

        Returns
        -------
        out : SparseArray
            The complex conjugate, with same dtype as the input.

        See Also
        --------
        numpy.ndarray.conj : NumPy equivalent method.
        numpy.conj : NumPy equivalent function.
        """
        return np.conj(self)
