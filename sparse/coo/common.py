from functools import reduce, wraps
import operator
import warnings
from collections import Iterable

import numpy as np
import scipy.sparse
import numba

from ..sparse_array import SparseArray
from ..compatibility import range, int
from ..utils import isscalar, normalize_axis, check_zero_fill_value, check_consistent_fill_value


def asCOO(x, name='asCOO', check=True):
    """
    Convert the input to :obj:`COO`. Passes through :obj:`COO` objects as-is.

    Parameters
    ----------
    x : Union[SparseArray, scipy.sparse.spmatrix, numpy.ndarray]
        The input array to convert.
    name : str, optional
        The name of the operation to use in the exception.
    check : bool, optional
        Whether to check for a dense input.

    Returns
    -------
    COO
        The converted :obj:`COO` array.

    Raises
    ------
    ValueError
        If ``check`` is true and a dense input is supplied.
    """
    from .core import COO

    if check and not isinstance(x, (SparseArray, scipy.sparse.spmatrix)):
        raise ValueError('Performing this operation would produce a dense result: %s' % name)

    if not isinstance(x, COO):
        x = COO(x)

    return x


def linear_loc(coords, shape):
    out = np.zeros(coords.shape[1], dtype=np.intp)
    tmp = np.zeros(coords.shape[1], dtype=np.intp)
    strides = int(1)
    for i, d in enumerate(shape[::-1]):
        np.multiply(coords[-(i + 1), :], strides, out=tmp)
        np.add(tmp, out, out=out)
        strides *= d
    return out


def tensordot(a, b, axes=2):
    """
    Perform the equivalent of :obj:`numpy.tensordot`.

    Parameters
    ----------
    a, b : Union[COO, np.ndarray, scipy.sparse.spmatrix]
        The arrays to perform the :code:`tensordot` operation on.
    axes : tuple[Union[int, tuple[int], Union[int, tuple[int]], optional
        The axes to match when performing the sum.

    Returns
    -------
    Union[COO, numpy.ndarray]
        The result of the operation.

    Raises
    ------
    ValueError
        If all arguments don't have zero fill-values.

    See Also
    --------
    numpy.tensordot : NumPy equivalent function
    """
    # Much of this is stolen from numpy/core/numeric.py::tensordot
    # Please see license at https://github.com/numpy/numpy/blob/master/LICENSE.txt
    check_zero_fill_value(a, b)

    if scipy.sparse.issparse(a):
        a = asCOO(a)
    if scipy.sparse.issparse(b):
        b = asCOO(b)

    try:
        iter(axes)
    except TypeError:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    # a, b = asarray(a), asarray(b)  # <--- modified
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (-1, N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, -1)
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = _dot(at, bt)
    return res.reshape(olda + oldb)


def matmul(a, b):
    """Perform the equivalent of :obj:`numpy.matmul` on two arrays.

    Parameters
    ----------
    a, b : Union[COO, np.ndarray, scipy.sparse.spmatrix]
        The arrays to perform the :code:`matmul` operation on.

    Returns
    -------
    Union[COO, numpy.ndarray]
        The result of the operation.

    Raises
    ------
    ValueError
        If all arguments don't have zero fill-values, or the shape of the two arrays is not broadcastable.

    See Also
    --------
    numpy.matmul : NumPy equivalent function.
    COO.__matmul__ : Equivalent function for COO objects.
    """
    check_zero_fill_value(a, b)
    if not hasattr(a, 'ndim') or not hasattr(b, 'ndim'):
        raise TypeError(
            "Cannot perform dot product on types %s, %s" %
            (type(a), type(b)))

    # When b is 2-d, it is equivalent to dot
    if b.ndim <= 2:
        return dot(a, b)

    # when a is 2-d, we need to transpose result after dot
    if a.ndim <= 2:
        res = dot(a, b)
        axes = list(range(res.ndim))
        axes.insert(-1, axes.pop(0))
        return res.transpose(axes)

    # If a can be squeeze to a vector, use dot will be faster
    if a.ndim <= b.ndim and np.prod(a.shape[:-1]) == 1:
        res = dot(a.reshape(-1), b)
        shape = list(res.shape)
        shape.insert(-1, 1)
        return res.reshape(shape)

    # If b can be squeeze to a matrix, use dot will be faster
    if b.ndim <= a.ndim and np.prod(b.shape[:-2]) == 1:
        return dot(a, b.reshape(b.shape[-2:]))

    if a.ndim < b.ndim:
        a = a[(None,) * (b.ndim - a.ndim)]
    if a.ndim > b.ndim:
        b = b[(None,) * (a.ndim - b.ndim)]
    for i, j in zip(a.shape[:-2], b.shape[:-2]):
        if i != 1 and j != 1 and i != j:
            raise ValueError('shapes of a and b are not broadcastable')

    def _matmul_recurser(a, b):
        if a.ndim == 2:
            return dot(a, b)
        res = []
        for i in range(max(a.shape[0], b.shape[0])):
            a_i = a[0] if a.shape[0] == 1 else a[i]
            b_i = b[0] if b.shape[0] == 1 else b[i]
            res.append(_matmul_recurser(a_i, b_i))
        mask = [isinstance(x, SparseArray) for x in res]
        if all(mask):
            return stack(res)
        else:
            res = [x.todense() if isinstance(x, SparseArray) else x
                   for x in res]
            return np.stack(res)
    return _matmul_recurser(a, b)


def dot(a, b):
    """
    Perform the equivalent of :obj:`numpy.dot` on two arrays.

    Parameters
    ----------
    a, b : Union[COO, np.ndarray, scipy.sparse.spmatrix]
        The arrays to perform the :code:`dot` operation on.

    Returns
    -------
    Union[COO, numpy.ndarray]
        The result of the operation.

    Raises
    ------
    ValueError
        If all arguments don't have zero fill-values.

    See Also
    --------
    numpy.dot : NumPy equivalent function.
    COO.dot : Equivalent function for COO objects.
    """
    check_zero_fill_value(a, b)
    if not hasattr(a, 'ndim') or not hasattr(b, 'ndim'):
        raise TypeError(
            "Cannot perform dot product on types %s, %s" %
            (type(a), type(b)))

    if a.ndim == 1 and b.ndim == 1:
        return (a * b).sum()

    a_axis = -1
    b_axis = -2

    if b.ndim == 1:
        b_axis = -1
    return tensordot(a, b, axes=(a_axis, b_axis))


def _dot(a, b):
    from .core import COO
    out_shape = (a.shape[0], b.shape[1])
    if isinstance(a, COO) and isinstance(b, COO):
        b = b.T
        coords, data = _dot_coo_coo_type(a.dtype, b.dtype)(a.coords, a.data, b.coords, b.data)

        return COO(coords, data, shape=out_shape, has_duplicates=False, sorted=True)

    if isinstance(a, COO) and isinstance(b, np.ndarray):
        b = b.view(type=np.ndarray).T
        return _dot_coo_ndarray_type(a.dtype, b.dtype)(a.coords, a.data, b, out_shape)

    if isinstance(a, np.ndarray) and isinstance(b, COO):
        b = b.T
        a = a.view(type=np.ndarray)
        return _dot_ndarray_coo_type(a.dtype, b.dtype)(a, b.coords, b.data, out_shape)


def kron(a, b):
    """Kronecker product of 2 sparse arrays.

    Parameters
    ----------
    a, b : SparseArray, scipy.sparse.spmatrix, or np.ndarray
        The arrays over which to compute the Kronecker product.

    Returns
    -------
    res : COO
        The kronecker product

    Raises
    ------
    ValueError
        If all arguments are dense or arguments have nonzero fill-values.

    Examples
    --------
    >>> a = eye(3, dtype='i8')
    >>> b = np.array([1, 2, 3], dtype='i8')
    >>> res = kron(a, b)
    >>> res.todense()  # doctest: +SKIP
    array([[1, 2, 3, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 2, 3, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 2, 3]], dtype=int64)
    """
    from .core import COO
    from .umath import _cartesian_product
    check_zero_fill_value(a, b)

    a_sparse = isinstance(a, (SparseArray, scipy.sparse.spmatrix))
    b_sparse = isinstance(b, (SparseArray, scipy.sparse.spmatrix))
    a_ndim = np.ndim(a)
    b_ndim = np.ndim(b)

    if not (a_sparse or b_sparse):
        raise ValueError('Performing this operation would produce a dense '
                         'result: kron')

    if a_ndim == 0 or b_ndim == 0:
        return a * b

    a = asCOO(a, check=False)
    b = asCOO(b, check=False)

    # Match dimensions
    max_dim = max(a.ndim, b.ndim)
    a = a.reshape((1,) * (max_dim - a.ndim) + a.shape)
    b = b.reshape((1,) * (max_dim - b.ndim) + b.shape)

    a_idx, b_idx = _cartesian_product(np.arange(a.nnz), np.arange(b.nnz))

    a_expanded_coords = a.coords[:, a_idx]
    b_expanded_coords = b.coords[:, b_idx]
    o_coords = a_expanded_coords * np.asarray(b.shape)[:, None] + b_expanded_coords
    o_data = a.data[a_idx] * b.data[b_idx]
    o_shape = tuple(i * j for i, j in zip(a.shape, b.shape))

    return COO(o_coords, o_data, shape=o_shape, has_duplicates=False)


def concatenate(arrays, axis=0):
    """
    Concatenate the input arrays along the given dimension.

    Parameters
    ----------
    arrays : Iterable[SparseArray]
        The input arrays to concatenate.
    axis : int, optional
        The axis along which to concatenate the input arrays. The default is zero.

    Returns
    -------
    COO
        The output concatenated array.

    Raises
    ------
    ValueError
        If all elements of :code:`arrays` don't have the same fill-value.

    See Also
    --------
    numpy.concatenate : NumPy equivalent function
    """
    from .core import COO
    check_consistent_fill_value(arrays)

    arrays = [x if isinstance(x, COO) else COO(x) for x in arrays]
    axis = normalize_axis(axis, arrays[0].ndim)
    assert all(x.shape[ax] == arrays[0].shape[ax]
               for x in arrays
               for ax in set(range(arrays[0].ndim)) - {axis})
    nnz = 0
    dim = sum(x.shape[axis] for x in arrays)
    shape = list(arrays[0].shape)
    shape[axis] = dim

    data = np.concatenate([x.data for x in arrays])
    coords = np.concatenate([x.coords for x in arrays], axis=1)

    dim = 0
    for x in arrays:
        if dim:
            coords[axis, nnz:x.nnz + nnz] += dim
        dim += x.shape[axis]
        nnz += x.nnz

    return COO(coords, data, shape=shape, has_duplicates=False,
               sorted=(axis == 0), fill_value=arrays[0].fill_value)


def stack(arrays, axis=0):
    """
    Stack the input arrays along the given dimension.

    Parameters
    ----------
    arrays : Iterable[SparseArray]
        The input arrays to stack.
    axis : int, optional
        The axis along which to stack the input arrays.

    Returns
    -------
    COO
        The output stacked array.

    Raises
    ------
    ValueError
        If all elements of :code:`arrays` don't have the same fill-value.

    See Also
    --------
    numpy.stack : NumPy equivalent function
    """
    from .core import COO
    check_consistent_fill_value(arrays)

    assert len(set(x.shape for x in arrays)) == 1
    arrays = [x if isinstance(x, COO) else COO(x) for x in arrays]
    axis = normalize_axis(axis, arrays[0].ndim + 1)
    data = np.concatenate([x.data for x in arrays])
    coords = np.concatenate([x.coords for x in arrays], axis=1)
    shape = list(arrays[0].shape)
    shape.insert(axis, len(arrays))

    nnz = 0
    dim = 0
    new = np.empty(shape=(coords.shape[1],), dtype=np.intp)
    for x in arrays:
        new[nnz:x.nnz + nnz] = dim
        dim += 1
        nnz += x.nnz

    coords = [coords[i] for i in range(coords.shape[0])]
    coords.insert(axis, new)
    coords = np.stack(coords, axis=0)

    return COO(coords, data, shape=shape, has_duplicates=False,
               sorted=(axis == 0), fill_value=arrays[0].fill_value)


def triu(x, k=0):
    """
    Returns an array with all elements below the k-th diagonal set to zero.

    Parameters
    ----------
    x : COO
        The input array.
    k : int, optional
        The diagonal below which elements are set to zero. The default is
        zero, which corresponds to the main diagonal.

    Returns
    -------
    COO
        The output upper-triangular matrix.

    Raises
    ------
    ValueError
        If :code:`x` doesn't have zero fill-values.

    See Also
    --------
    numpy.triu : NumPy equivalent function
    """
    from .core import COO
    check_zero_fill_value(x)

    if not x.ndim >= 2:
        raise NotImplementedError('sparse.triu is not implemented for scalars or 1-D arrays.')

    mask = x.coords[-2] + k <= x.coords[-1]

    coords = x.coords[:, mask]
    data = x.data[mask]

    return COO(coords, data, shape=x.shape, has_duplicates=False, sorted=True)


def tril(x, k=0):
    """
    Returns an array with all elements above the k-th diagonal set to zero.

    Parameters
    ----------
    x : COO
        The input array.
    k : int, optional
        The diagonal above which elements are set to zero. The default is
        zero, which corresponds to the main diagonal.

    Returns
    -------
    COO
        The output lower-triangular matrix.

    Raises
    ------
    ValueError
        If :code:`x` doesn't have zero fill-values.

    See Also
    --------
    numpy.tril : NumPy equivalent function
    """
    from .core import COO
    check_zero_fill_value(x)

    if not x.ndim >= 2:
        raise NotImplementedError('sparse.tril is not implemented for scalars or 1-D arrays.')

    mask = x.coords[-2] + k >= x.coords[-1]

    coords = x.coords[:, mask]
    data = x.data[mask]

    return COO(coords, data, shape=x.shape, has_duplicates=False, sorted=True)


def nansum(x, axis=None, keepdims=False, dtype=None, out=None):
    """
    Performs a ``NaN`` skipping sum operation along the given axes. Uses all axes by default.

    Parameters
    ----------
    x : SparseArray
        The array to perform the reduction on.
    axis : Union[int, Iterable[int]], optional
        The axes along which to sum. Uses all axes by default.
    keepdims : bool, optional
        Whether or not to keep the dimensions of the original array.
    dtype: numpy.dtype
        The data type of the output array.

    Returns
    -------
    COO
        The reduced output sparse array.

    See Also
    --------
    :obj:`COO.sum` : Function without ``NaN`` skipping.
    numpy.nansum : Equivalent Numpy function.
    """
    assert out is None
    x = asCOO(x, name='nansum')
    return nanreduce(x, np.add, axis=axis, keepdims=keepdims, dtype=dtype)


def nanmean(x, axis=None, keepdims=False, dtype=None, out=None):
    """
    Performs a ``NaN`` skipping mean operation along the given axes. Uses all axes by default.

    Parameters
    ----------
    x : SparseArray
        The array to perform the reduction on.
    axis : Union[int, Iterable[int]], optional
        The axes along which to compute the mean. Uses all axes by default.
    keepdims : bool, optional
        Whether or not to keep the dimensions of the original array.
    dtype: numpy.dtype
        The data type of the output array.

    Returns
    -------
    COO
        The reduced output sparse array.

    See Also
    --------
    :obj:`COO.mean` : Function without ``NaN`` skipping.
    numpy.nanmean : Equivalent Numpy function.
    """
    assert out is None
    x = asCOO(x, name='nanmean')

    if not np.issubdtype(x.dtype, np.floating):
        return x.mean(axis=axis, keepdims=keepdims, dtype=dtype)

    mask = np.isnan(x)
    x2 = where(mask, 0, x)

    # Count the number non-nan elements along axis
    nancount = mask.sum(axis=axis, dtype='i8', keepdims=keepdims)
    if axis is None:
        axis = tuple(range(x.ndim))
    elif not isinstance(axis, tuple):
        axis = (axis,)
    den = reduce(operator.mul, (x.shape[i] for i in axis), 1)
    den -= nancount

    if (den == 0).any():
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)

    num = np.sum(x2, axis=axis, dtype=dtype, keepdims=keepdims)

    with np.errstate(invalid='ignore', divide='ignore'):
        if num.ndim:
            return np.true_divide(num, den, casting='unsafe')
        return (num / den).astype(dtype)


def nanmax(x, axis=None, keepdims=False, dtype=None, out=None):
    """
    Maximize along the given axes, skipping ``NaN`` values. Uses all axes by default.

    Parameters
    ----------
    x : SparseArray
        The array to perform the reduction on.
    axis : Union[int, Iterable[int]], optional
        The axes along which to maximize. Uses all axes by default.
    keepdims : bool, optional
        Whether or not to keep the dimensions of the original array.
    dtype: numpy.dtype
        The data type of the output array.

    Returns
    -------
    COO
        The reduced output sparse array.

    See Also
    --------
    :obj:`COO.max` : Function without ``NaN`` skipping.
    numpy.nanmax : Equivalent Numpy function.
    """
    assert out is None
    x = asCOO(x, name='nanmax')

    ar = x.reduce(np.fmax, axis=axis, keepdims=keepdims,
                  dtype=dtype)

    if (isscalar(ar) and np.isnan(ar)) or np.isnan(ar.data).any():
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)

    return ar


def nanmin(x, axis=None, keepdims=False, dtype=None, out=None):
    """
    Minimize along the given axes, skipping ``NaN`` values. Uses all axes by default.

    Parameters
    ----------
    x : SparseArray
        The array to perform the reduction on.
    axis : Union[int, Iterable[int]], optional
        The axes along which to minimize. Uses all axes by default.
    keepdims : bool, optional
        Whether or not to keep the dimensions of the original array.
    dtype: numpy.dtype
        The data type of the output array.

    Returns
    -------
    COO
        The reduced output sparse array.

    See Also
    --------
    :obj:`COO.min` : Function without ``NaN`` skipping.
    numpy.nanmin : Equivalent Numpy function.
    """
    assert out is None
    x = asCOO(x, name='nanmin')

    ar = x.reduce(np.fmin, axis=axis, keepdims=keepdims,
                  dtype=dtype)

    if (isscalar(ar) and np.isnan(ar)) or np.isnan(ar.data).any():
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)

    return ar


def nanprod(x, axis=None, keepdims=False, dtype=None, out=None):
    """
    Performs a product operation along the given axes, skipping ``NaN`` values.
    Uses all axes by default.

    Parameters
    ----------
    x : SparseArray
        The array to perform the reduction on.
    axis : Union[int, Iterable[int]], optional
        The axes along which to multiply. Uses all axes by default.
    keepdims : bool, optional
        Whether or not to keep the dimensions of the original array.
    dtype: numpy.dtype
        The data type of the output array.

    Returns
    -------
    COO
        The reduced output sparse array.

    See Also
    --------
    :obj:`COO.prod` : Function without ``NaN`` skipping.
    numpy.nanprod : Equivalent Numpy function.
    """
    assert out is None
    x = asCOO(x)
    return nanreduce(x, np.multiply, axis=axis, keepdims=keepdims, dtype=dtype)


def where(condition, x=None, y=None):
    """
    Select values from either ``x`` or ``y`` depending on ``condition``.
    If ``x`` and ``y`` are not given, returns indices where ``condition``
    is nonzero.

    Performs the equivalent of :obj:`numpy.where`.

    Parameters
    ----------
    condition : SparseArray
        The condition based on which to select values from
        either ``x`` or ``y``.
    x : SparseArray, optional
        The array to select values from if ``condition`` is nonzero.
    y : SparseArray, optional
        The array to select values from if ``condition`` is zero.

    Returns
    -------
    COO
        The output array with selected values if ``x`` and ``y`` are given;
        else where the array is nonzero.

    Raises
    ------
    ValueError
        If the operation would produce a dense result; or exactly one of
        ``x`` and ``y`` are given.

    See Also
    --------
    numpy.where : Equivalent Numpy function.
    """
    from .umath import elemwise

    x_given = x is not None
    y_given = y is not None

    if not (x_given or y_given):
        condition = asCOO(condition, name=str(np.where))
        return tuple(condition.coords)

    if x_given != y_given:
        raise ValueError('either both or neither of x and y should be given')

    return elemwise(np.where, condition, x, y)


def _replace_nan(array, value):
    """
    Replaces ``NaN``s in ``array`` with ``value``.

    Parameters
    ----------
    array : COO
        The input array.
    value : numpy.number
        The values to replace ``NaN`` with.

    Returns
    -------
    COO
        A copy of ``array`` with the ``NaN``s replaced.
    """
    if not np.issubdtype(array.dtype, np.floating):
        return array

    return where(np.isnan(array), value, array)


def nanreduce(x, method, identity=None, axis=None, keepdims=False, **kwargs):
    """
    Performs an ``NaN`` skipping reduction on this array. See the documentation
    on :obj:`COO.reduce` for examples.

    Parameters
    ----------
    x : COO
        The array to reduce.
    method : numpy.ufunc
        The method to use for performing the reduction.
    identity : numpy.number
        The identity value for this reduction. Inferred from ``method`` if not given.
        Note that some ``ufunc`` objects don't have this, so it may be necessary to give it.
    axis : Union[int, Iterable[int]], optional
        The axes along which to perform the reduction. Uses all axes by default.
    keepdims : bool, optional
        Whether or not to keep the dimensions of the original array.
    kwargs : dict
        Any extra arguments to pass to the reduction operation.

    Returns
    -------
    COO
        The result of the reduction operation.

    Raises
    ------
    ValueError
        If reducing an all-zero axis would produce a nonzero result.

    See Also
    --------
    COO.reduce : Similar method without ``NaN`` skipping functionality.
    """
    arr = _replace_nan(x, method.identity if identity is None else identity)
    return arr.reduce(method, axis, keepdims, **kwargs)


def roll(a, shift, axis=None):
    """
    Shifts elements of an array along specified axis. Elements that roll beyond
    the last position are circulated and re-introduced at the first.

    Parameters
    ----------
    x : COO
        Input array
    shift : int or tuple of ints
        Number of index positions that elements are shifted. If a tuple is
        provided, then axis must be a tuple of the same size, and each of the
        given axes is shifted by the corresponding number. If an int while axis
        is a tuple of ints, then broadcasting is used so the same shift is
        applied to all axes.
    axis : int or tuple of ints, optional
        Axis or tuple specifying multiple axes. By default, the
        array is flattened before shifting, after which the original shape is
        restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as a.
    """
    from .core import COO, as_coo
    a = as_coo(a)

    # roll flattened array
    if axis is None:
        return roll(a.reshape((-1,)), shift, 0).reshape(a.shape)

    # roll across specified axis
    else:
        # parse axis input, wrap in tuple
        axis = normalize_axis(axis, a.ndim)
        if not isinstance(axis, tuple):
            axis = (axis,)

        # make shift iterable
        if not isinstance(shift, Iterable):
            shift = (shift,)

        elif np.ndim(shift) > 1:
            raise ValueError(
                "'shift' and 'axis' must be integers or 1D sequences.")

        # handle broadcasting
        if len(shift) == 1:
            shift = np.full(len(axis), shift)

        # check if dimensions are consistent
        if len(axis) != len(shift):
            raise ValueError(
                "If 'shift' is a 1D sequence, "
                "'axis' must have equal length.")

        # shift elements
        coords, data = np.copy(a.coords), np.copy(a.data)
        for sh, ax in zip(shift, axis):
            coords[ax] += sh
            coords[ax] %= a.shape[ax]

        return COO(coords, data=data, shape=a.shape, has_duplicates=False, fill_value=a.fill_value)


def eye(N, M=None, k=0, dtype=float):
    """Return a 2-D COO array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the output.
    M : int, optional
        Number of columns in the output. If None, defaults to `N`.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.
    dtype : data-type, optional
        Data-type of the returned array.

    Returns
    -------
    I : COO array of shape (N, M)
        An array where all elements are equal to zero, except for the `k`-th
        diagonal, whose values are equal to one.

    Examples
    --------
    >>> eye(2, dtype=int).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 0],
           [0, 1]])
    >>> eye(3, k=1).todense()  # doctest: +SKIP
    array([[0., 1., 0.],
           [0., 0., 1.],
           [0., 0., 0.]])
    """
    from .core import COO

    if M is None:
        M = N

    N = int(N)
    M = int(M)
    k = int(k)

    data_length = min(N, M)

    if k > 0:
        data_length = max(min(data_length, M - k), 0)
        n_coords = np.arange(data_length, dtype=np.intp)
        m_coords = n_coords + k
    elif k < 0:
        data_length = max(min(data_length, N + k), 0)
        m_coords = np.arange(data_length, dtype=np.intp)
        n_coords = m_coords - k
    else:
        n_coords = m_coords = np.arange(data_length, dtype=np.intp)

    coords = np.stack([n_coords, m_coords])
    data = np.array(1, dtype=dtype)

    return COO(coords, data=data, shape=(N, M), has_duplicates=False,
               sorted=True)


def full(shape, fill_value, dtype=None):
    """Return a COO array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array. The default, `None`, means
        `np.array(fill_value).dtype`.

    Returns
    -------
    out : COO
        Array of `fill_value` with the given shape and dtype.

    Examples
    --------
    >>> full(5, 9).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([9, 9, 9, 9, 9])

    >>> full((2, 2), 9, dtype=float).todense()  # doctest: +SKIP
    array([[9., 9.],
           [9., 9.]])
    """
    from .core import COO

    if dtype is None:
        dtype = np.array(fill_value).dtype
    if not isinstance(shape, tuple):
        shape = (shape,)
    data = np.empty(0, dtype=dtype)
    coords = np.empty((len(shape), 0), dtype=np.intp)
    return COO(coords, data=data, shape=shape,
               fill_value=fill_value, has_duplicates=False,
               sorted=True)


def full_like(a, fill_value, dtype=None):
    """Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of the result will match those of `a`.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    out : COO
        Array of `fill_value` with the same shape and type as `a`.

    Examples
    --------
    >>> x = np.ones((2, 3), dtype='i8')
    >>> full_like(x, 9.0).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[9, 9, 9],
           [9, 9, 9]])
    """
    return full(a.shape, fill_value, dtype=(a.dtype if dtype is None else dtype))


def zeros(shape, dtype=float):
    """Return a COO array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.

    Returns
    -------
    out : COO
        Array of zeros with the given shape and dtype.

    Examples
    --------
    >>> zeros(5).todense()  # doctest: +SKIP
    array([0., 0., 0., 0., 0.])

    >>> zeros((2, 2), dtype=int).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0],
           [0, 0]])
    """
    return full(shape, 0, np.dtype(dtype))


def zeros_like(a, dtype=None):
    """Return a COO array of zeros with the same shape and type as ``a``.

    Parameters
    ----------
    a : array_like
        The shape and data-type of the result will match those of `a`.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    out : COO
        Array of zeros with the same shape and type as `a`.

    Examples
    --------
    >>> x = np.ones((2, 3), dtype='i8')
    >>> zeros_like(x).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0],
           [0, 0, 0]])
    """
    return zeros(a.shape, dtype=(a.dtype if dtype is None else dtype))


def ones(shape, dtype=float):
    """Return a COO array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.

    Returns
    -------
    out : COO
        Array of ones with the given shape and dtype.

    Examples
    --------
    >>> ones(5).todense()  # doctest: +SKIP
    array([1., 1., 1., 1., 1.])

    >>> ones((2, 2), dtype=int).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1],
           [1, 1]])
    """
    return full(shape, 1, np.dtype(dtype))


def ones_like(a, dtype=None):
    """Return a COO array of ones with the same shape and type as ``a``.

    Parameters
    ----------
    a : array_like
        The shape and data-type of the result will match those of `a`.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    out : COO
        Array of ones with the same shape and type as `a`.

    Examples
    --------
    >>> x = np.ones((2, 3), dtype='i8')
    >>> ones_like(x).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 1],
           [1, 1, 1]])
    """
    return ones(a.shape, dtype=(a.dtype if dtype is None else dtype))


def _memoize_dtype(f):
    """
    Memoizes a function taking in NumPy dtypes.

    Parameters
    ----------
    f : Callable

    Returns
    -------
    wrapped : Callable

    Examples
    --------
    >>> def func(dt1):
    ...     return object()
    >>> func = _memoize_dtype(func)
    >>> func(np.dtype('i8')) is func(np.dtype('int64'))
    True
    >>> func(np.dtype('i8')) is func(np.dtype('i4'))
    False
    """
    cache = {}

    @wraps(f)
    def wrapped(*args):
        key = tuple(arg.name for arg in args)
        if key in cache:
            return cache[key]

        result = f(*args)
        cache[key] = result
        return result

    return wrapped


@_memoize_dtype
def _dot_coo_coo_type(dt1, dt2):
    dtr = np.result_type(dt1, dt2)

    @numba.jit(nopython=True, nogil=True,
               locals={'data_curr': numba.numpy_support.from_dtype(dtr)})
    def _dot_coo_coo(coords1, data1, coords2, data2):  # pragma: no cover
        """
        Utility function taking in two ``COO`` objects and calculating a "sense"
        of their dot product. Acually computes ``s1 @ s2.T``.

        Parameters
        ----------
        data1, coords1 : np.ndarray
            The data and coordinates of ``s1``.

        data2, coords2 : np.ndarray
            The data and coordinates of ``s2``.
        """
        coords_out = []
        data_out = []
        didx1 = 0

        while didx1 < len(data1):
            oidx1 = coords1[0, didx1]
            didx2 = 0
            didx1_curr = didx1

            while didx2 < len(data2) and didx1 < len(data1) and coords1[0, didx1] == oidx1:
                oidx2 = coords2[0, didx2]
                data_curr = 0

                while didx2 < len(data2) and didx1 < len(data1) and \
                        coords2[0, didx2] == oidx2 and coords1[0, didx1] == oidx1:
                    if coords1[1, didx1] < coords2[1, didx2]:
                        didx1 += 1
                    elif coords1[1, didx1] > coords2[1, didx2]:
                        didx2 += 1
                    else:
                        data_curr += data1[didx1] * data2[didx2]
                        didx1 += 1
                        didx2 += 1

                while didx2 < len(data2) and coords2[0, didx2] == oidx2:
                    didx2 += 1

                if didx2 < len(data2):
                    didx1 = didx1_curr

                if data_curr != 0:
                    coords_out.append((oidx1, oidx2))
                    data_out.append(data_curr)

            while didx1 < len(data1) and coords1[0, didx1] == oidx1:
                didx1 += 1

        if len(data_out) == 0:
            return np.empty((2, 0), dtype=np.intp), np.empty((0,), dtype=dtr)

        return np.array(coords_out).T, np.array(data_out)

    return _dot_coo_coo


@_memoize_dtype
def _dot_coo_ndarray_type(dt1, dt2):
    dtr = np.result_type(dt1, dt2)

    @numba.jit(nopython=True, nogil=True)
    def _dot_coo_ndarray(coords1, data1, array2, out_shape):  # pragma: no cover
        """
        Utility function taking in one `COO` and one ``ndarray`` and
        calculating a "sense" of their dot product. Acually computes
        ``s1 @ x2.T``.

        Parameters
        ----------
        data1, coords1 : np.ndarray
            The data and coordinates of ``s1``.

        array2 : np.ndarray
            The second input array ``x2``.

        out_shape : Tuple[int]
            The output shape.
        """
        out = np.zeros(out_shape, dtype=dtr)
        didx1 = 0

        while didx1 < len(data1):
            oidx1 = coords1[0, didx1]
            didx1_curr = didx1

            for oidx2 in range(out_shape[1]):
                didx1 = didx1_curr
                while didx1 < len(data1) and coords1[0, didx1] == oidx1:
                    out[oidx1, oidx2] += data1[didx1] * array2[oidx2, coords1[1, didx1]]
                    didx1 += 1

        return out

    return _dot_coo_ndarray


@_memoize_dtype
def _dot_ndarray_coo_type(dt1, dt2):
    dtr = np.result_type(dt1, dt2)

    @numba.jit(nopython=True, nogil=True)
    def _dot_ndarray_coo(array1, coords2, data2, out_shape):  # pragma: no cover
        """
        Utility function taking in two one ``ndarray`` and one ``COO`` and
        calculating a "sense" of their dot product. Acually computes ``x1 @ s2.T``.

        Parameters
        ----------
        array1 : np.ndarray
            The input array ``x1``.

        data2, coords2 : np.ndarray
            The data and coordinates of ``s2``.

        out_shape : Tuple[int]
            The output shape.
        """
        out = np.zeros(out_shape, dtype=dtr)

        for oidx1 in range(out_shape[0]):
            for didx2 in range(len(data2)):
                oidx2 = coords2[0, didx2]
                out[oidx1, oidx2] += array1[oidx1, coords2[1, didx2]] * data2[didx2]

        return out

    return _dot_ndarray_coo
