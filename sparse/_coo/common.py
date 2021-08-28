from functools import reduce
import operator
import warnings
from collections.abc import Iterable

import numpy as np
import scipy.sparse
import numba

from .._sparse_array import SparseArray
from .._utils import (
    isscalar,
    is_unsigned_dtype,
    normalize_axis,
    check_zero_fill_value,
    check_consistent_fill_value,
    can_store,
)


def asCOO(x, name="asCOO", check=True):
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
        raise ValueError(
            "Performing this operation would produce a dense result: %s" % name
        )

    if not isinstance(x, COO):
        x = COO(x)

    return x


def linear_loc(coords, shape):
    if shape == () and len(coords) == 0:
        # `np.ravel_multi_index` is not aware of arrays, so cannot produce a
        # sensible result here (https://github.com/numpy/numpy/issues/15690).
        # Since `coords` is an array and not a sequence, we know the correct
        # dimensions.
        return np.zeros(coords.shape[1:], dtype=np.intp)
    else:
        return np.ravel_multi_index(coords, shape)


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
    >>> from sparse import eye
    >>> a = eye(3, dtype='i8')
    >>> b = np.array([1, 2, 3], dtype='i8')
    >>> res = kron(a, b)
    >>> res.todense()  # doctest: +SKIP
    array([[1, 2, 3, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 2, 3, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 2, 3]], dtype=int64)
    """
    from .core import COO
    from .._umath import _cartesian_product

    check_zero_fill_value(a, b)

    a_sparse = isinstance(a, (SparseArray, scipy.sparse.spmatrix))
    b_sparse = isinstance(b, (SparseArray, scipy.sparse.spmatrix))
    a_ndim = np.ndim(a)
    b_ndim = np.ndim(b)

    if not (a_sparse or b_sparse):
        raise ValueError(
            "Performing this operation would produce a dense " "result: kron"
        )

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
    assert all(
        x.shape[ax] == arrays[0].shape[ax]
        for x in arrays
        for ax in set(range(arrays[0].ndim)) - {axis}
    )
    nnz = 0
    dim = sum(x.shape[axis] for x in arrays)
    shape = list(arrays[0].shape)
    shape[axis] = dim

    data = np.concatenate([x.data for x in arrays])
    coords = np.concatenate([x.coords for x in arrays], axis=1)

    if not can_store(coords.dtype, max(shape)):
        coords = coords.astype(np.min_scalar_type(max(shape)))
    dim = 0
    for x in arrays:
        if dim:
            coords[axis, nnz : x.nnz + nnz] += dim
        dim += x.shape[axis]
        nnz += x.nnz

    return COO(
        coords,
        data,
        shape=shape,
        has_duplicates=False,
        sorted=(axis == 0),
        fill_value=arrays[0].fill_value,
    )


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

    assert len({x.shape for x in arrays}) == 1
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
        new[nnz : x.nnz + nnz] = dim
        dim += 1
        nnz += x.nnz

    coords = [coords[i] for i in range(coords.shape[0])]
    coords.insert(axis, new)
    coords = np.stack(coords, axis=0)

    return COO(
        coords,
        data,
        shape=shape,
        has_duplicates=False,
        sorted=(axis == 0),
        fill_value=arrays[0].fill_value,
    )


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
        raise NotImplementedError(
            "sparse.triu is not implemented for scalars or 1-D arrays."
        )

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
        raise NotImplementedError(
            "sparse.tril is not implemented for scalars or 1-D arrays."
        )

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
    dtype : numpy.dtype
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
    x = asCOO(x, name="nansum")
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
    dtype : numpy.dtype
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
    x = asCOO(x, name="nanmean")

    if not np.issubdtype(x.dtype, np.floating):
        return x.mean(axis=axis, keepdims=keepdims, dtype=dtype)

    mask = np.isnan(x)
    x2 = where(mask, 0, x)

    # Count the number non-nan elements along axis
    nancount = mask.sum(axis=axis, dtype="i8", keepdims=keepdims)
    if axis is None:
        axis = tuple(range(x.ndim))
    elif not isinstance(axis, tuple):
        axis = (axis,)
    den = reduce(operator.mul, (x.shape[i] for i in axis), 1)
    den -= nancount

    if (den == 0).any():
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)

    num = np.sum(x2, axis=axis, dtype=dtype, keepdims=keepdims)

    with np.errstate(invalid="ignore", divide="ignore"):
        if num.ndim:
            return np.true_divide(num, den, casting="unsafe")
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
    dtype : numpy.dtype
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
    x = asCOO(x, name="nanmax")

    ar = x.reduce(np.fmax, axis=axis, keepdims=keepdims, dtype=dtype)

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
    dtype : numpy.dtype
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
    x = asCOO(x, name="nanmin")

    ar = x.reduce(np.fmin, axis=axis, keepdims=keepdims, dtype=dtype)

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
    dtype : numpy.dtype
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
    from .._umath import elemwise

    x_given = x is not None
    y_given = y is not None

    if not (x_given or y_given):
        condition = asCOO(condition, name=str(np.where))
        return tuple(condition.coords)

    if x_given != y_given:
        raise ValueError("either both or neither of x and y should be given")

    return elemwise(np.where, condition, x, y)


def argwhere(a):
    """
    Find the indices of array elements that are non-zero, grouped by element.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    index_array: numpy.ndarray

    See Also
    --------
    :obj:`where`, :obj:`COO.nonzero`

    Examples
    --------
    >>> import sparse
    >>> x = sparse.COO(np.arange(6).reshape((2, 3)))
    >>> sparse.argwhere(x > 1)
    array([[0, 2],
           [1, 0],
           [1, 1],
           [1, 2]])
    """
    return np.transpose(a.nonzero())


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
    **kwargs : dict
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
    a : COO
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
    from numpy.core._exceptions import UFuncTypeError

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
            raise ValueError("'shift' and 'axis' must be integers or 1D sequences.")

        # handle broadcasting
        if len(shift) == 1:
            shift = np.full(len(axis), shift)

        # check if dimensions are consistent
        if len(axis) != len(shift):
            raise ValueError(
                "If 'shift' is a 1D sequence, " "'axis' must have equal length."
            )

        if not can_store(a.coords.dtype, max(a.shape + shift)):
            raise ValueError(
                "cannot roll with coords.dtype {} and shift {}. Try casting coords to a larger dtype.".format(
                    a.coords.dtype,
                    shift,
                )
            )

        # shift elements
        coords, data = np.copy(a.coords), np.copy(a.data)
        try:
            for sh, ax in zip(shift, axis):
                coords[ax] += sh
                coords[ax] %= a.shape[ax]
        except UFuncTypeError:
            if is_unsigned_dtype(coords.dtype):
                raise ValueError(
                    "rolling with coords.dtype as {} is not safe. Try using a signed dtype.".format(
                        coords.dtype
                    )
                )

        return COO(
            coords,
            data=data,
            shape=a.shape,
            has_duplicates=False,
            fill_value=a.fill_value,
        )


def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    Extract diagonal from a COO array. The equivalent of :obj:`numpy.diagonal`.

    Parameters
    ----------
    a : COO
        The array to perform the operation on.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Defaults to main diagonal (0).
    axis1 : int, optional
        First axis from which the diagonals should be taken.
        Defaults to first axis (0).
    axis2 : int, optional
        Second axis from which the diagonals should be taken.
        Defaults to second axis (1).

    Examples
    --------
    >>> import sparse
    >>> x = sparse.as_coo(np.arange(9).reshape(3,3))
    >>> sparse.diagonal(x).todense()
    array([0, 4, 8])
    >>> sparse.diagonal(x,offset=1).todense()
    array([1, 5])

    >>> x = sparse.as_coo(np.arange(12).reshape((2,3,2)))
    >>> x_diag = sparse.diagonal(x, axis1=0, axis2=2)
    >>> x_diag.shape
    (3, 2)
    >>> x_diag.todense()
    array([[ 0,  7],
           [ 2,  9],
           [ 4, 11]])

    Returns
    -------
    out: COO
        The result of the operation.

    Raises
    ------
    ValueError
        If a.shape[axis1] != a.shape[axis2]

    See Also
    --------
    :obj:`numpy.diagonal` : NumPy equivalent function
    """
    from .core import COO

    if a.shape[axis1] != a.shape[axis2]:
        raise ValueError("a.shape[axis1] != a.shape[axis2]")

    diag_axes = [
        axis for axis in range(len(a.shape)) if axis != axis1 and axis != axis2
    ] + [axis1]
    diag_shape = [a.shape[axis] for axis in diag_axes]
    diag_shape[-1] -= abs(offset)

    diag_idx = _diagonal_idx(a.coords, axis1, axis2, offset)

    diag_coords = [a.coords[axis][diag_idx] for axis in diag_axes]
    diag_data = a.data[diag_idx]

    return COO(diag_coords, diag_data, diag_shape)


def diagonalize(a, axis=0):
    """
    Diagonalize a COO array. The new dimension is appended at the end.

    .. WARNING:: :obj:`diagonalize` is not :obj:`numpy` compatible as there is no direct :obj:`numpy` equivalent. The API may change in the future.

    Parameters
    ----------
    a : Union[COO, np.ndarray, scipy.sparse.spmatrix]
        The array to diagonalize.
    axis : int, optional
        The axis to diagonalize. Defaults to first axis (0).

    Examples
    --------
    >>> import sparse
    >>> x = sparse.as_coo(np.arange(1,4))
    >>> sparse.diagonalize(x).todense()
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]])

    >>> x = sparse.as_coo(np.arange(24).reshape((2,3,4)))
    >>> x_diag = sparse.diagonalize(x, axis=1)
    >>> x_diag.shape
    (2, 3, 4, 3)

    :obj:`diagonalize` is the inverse of :obj:`diagonal`

    >>> a = sparse.random((3,3,3,3,3), density=0.3)
    >>> a_diag = sparse.diagonalize(a, axis=2)
    >>> (sparse.diagonal(a_diag, axis1=2, axis2=5) == a.transpose([0,1,3,4,2])).all()
    True

    Returns
    -------
    out: COO
        The result of the operation.

    See Also
    --------
    :obj:`numpy.diag` : NumPy equivalent for 1D array
    """
    from .core import COO, as_coo

    a = as_coo(a)

    diag_shape = a.shape + (a.shape[axis],)
    diag_coords = np.vstack([a.coords, a.coords[axis]])

    return COO(diag_coords, a.data, diag_shape)


def isposinf(x, out=None):
    """
    Test element-wise for positive infinity, return result as sparse ``bool`` array.

    Parameters
    ----------
    x
        Input
    out, optional
        Output array

    Examples
    --------
    >>> import sparse
    >>> x = sparse.as_coo(np.array([np.inf]))
    >>> sparse.isposinf(x).todense()
    array([ True])

    See Also
    --------
    numpy.isposinf : The NumPy equivalent
    """
    from .core import elemwise

    return elemwise(lambda x, out=None, dtype=None: np.isposinf(x, out=out), x, out=out)


def isneginf(x, out=None):
    """
    Test element-wise for negative infinity, return result as sparse ``bool`` array.

    Parameters
    ----------
    x
        Input
    out, optional
        Output array

    Examples
    --------
    >>> import sparse
    >>> x = sparse.as_coo(np.array([-np.inf]))
    >>> sparse.isneginf(x).todense()
    array([ True])

    See Also
    --------
    numpy.isneginf : The NumPy equivalent
    """
    from .core import elemwise

    return elemwise(lambda x, out=None, dtype=None: np.isneginf(x, out=out), x, out=out)


def result_type(*arrays_and_dtypes):
    """Returns the type that results from applying the NumPy type promotion rules to the
    arguments.

    See Also
    --------
    numpy.result_type : The NumPy equivalent
    """
    return np.result_type(*(_as_result_type_arg(x) for x in arrays_and_dtypes))


def _as_result_type_arg(x):
    if not isinstance(x, SparseArray):
        return x
    if x.ndim > 0:
        return x.dtype
    # 0-dimensional arrays give different result_type outputs than their dtypes
    return x.todense()


@numba.jit(nopython=True, nogil=True)
def _diagonal_idx(coordlist, axis1, axis2, offset):
    """
    Utility function that returns all indices that correspond to a diagonal element.

    Parameters
    ----------
    coordlist : list of lists
        Coordinate indices.
    axis1, axis2 : int
        The axes of the diagonal.
    offset : int
        Offset of the diagonal from the main diagonal. Defaults to main diagonal (0).
    """
    return np.array(
        [
            i
            for i in range(len(coordlist[axis1]))
            if coordlist[axis1][i] + offset == coordlist[axis2][i]
        ]
    )


def clip(a, a_min=None, a_max=None, out=None):
    """
    Clip (limit) the values in the array.

    Return an array whose values are limited to ``[min, max]``. One of min
    or max must be given.

    Parameters
    ----------
    a
    a_min : scalar or `SparseArray` or `None`
        Minimum value. If `None`, clipping is not performed on lower
        interval edge.
    a_max : scalar or `SparseArray` or `None`
        Maximum value. If `None`, clipping is not performed on upper
        interval edge.
    out : SparseArray, optional
        If provided, the results will be placed in this array. It may be
        the input array for in-place clipping. `out` must be of the right
        shape to hold the output. Its type is preserved.

    Returns
    -------
    clipped_array : SparseArray
        An array with the elements of `self`, but where values < `min` are
        replaced with `min`, and those > `max` with `max`.

    Examples
    --------
    >>> import sparse
    >>> x = sparse.COO.from_numpy([0, 0, 0, 1, 2, 3])
    >>> sparse.clip(x, a_min=1).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([1, 1, 1, 1, 2, 3])
    >>> sparse.clip(x, a_max=1).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([0, 0, 0, 1, 1, 1])
    >>> sparse.clip(x, a_min=1, a_max=2).todense() # doctest: +NORMALIZE_WHITESPACE
    array([1, 1, 1, 1, 2, 2])

    See Also
    --------
    numpy.clip : Equivalent NumPy function
    """
    a = asCOO(a, name="clip")
    return a.clip(a_min, a_max)
