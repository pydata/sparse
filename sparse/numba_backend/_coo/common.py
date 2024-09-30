import operator
import warnings
from collections.abc import Iterable
from functools import reduce
from typing import Any, NamedTuple

import numba

import numpy as np

from .._sparse_array import SparseArray
from .._utils import (
    can_store,
    check_consistent_fill_value,
    check_zero_fill_value,
    is_unsigned_dtype,
    isscalar,
    normalize_axis,
)


def asCOO(x, name="asCOO", check=True):
    """
    Convert the input to [`sparse.COO`][]. Passes through [`sparse.COO`][] objects as-is.

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
        The converted [`sparse.COO`][] array.

    Raises
    ------
    ValueError
        If `check` is true and a dense input is supplied.
    """
    from .._common import _is_sparse
    from .core import COO

    if check and not _is_sparse(x):
        raise ValueError(f"Performing this operation would produce a dense result: {name}")

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
    >>> a = eye(3, dtype="i8")
    >>> b = np.array([1, 2, 3], dtype="i8")
    >>> res = kron(a, b)
    >>> res.todense()  # doctest: +SKIP
    array([[1, 2, 3, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 2, 3, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 2, 3]], dtype=int64)
    """
    from .._common import _is_sparse
    from .._umath import _cartesian_product
    from .core import COO

    check_zero_fill_value(a, b)

    a_sparse = _is_sparse(a)
    b_sparse = _is_sparse(b)
    a_ndim = np.ndim(a)
    b_ndim = np.ndim(b)

    if not (a_sparse or b_sparse):
        raise ValueError("Performing this operation would produce a dense result: kron")

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
    o_shape = tuple(i * j for i, j in zip(a.shape, b.shape, strict=True))

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
        If all elements of `arrays` don't have the same fill-value.

    See Also
    --------
    [`numpy.concatenate`][] : NumPy equivalent function
    """
    from .core import COO

    check_consistent_fill_value(arrays)

    if axis is None:
        axis = 0
        arrays = [x.flatten() for x in arrays]

    arrays = [x if isinstance(x, COO) else COO(x) for x in arrays]
    axis = normalize_axis(axis, arrays[0].ndim)
    assert all(x.shape[ax] == arrays[0].shape[ax] for x in arrays for ax in set(range(arrays[0].ndim)) - {axis})
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
        If all elements of `arrays` don't have the same fill-value.

    See Also
    --------
    [`numpy.stack`][] : NumPy equivalent function
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
    new = np.empty(shape=(coords.shape[1],), dtype=np.intp)
    for dim, x in enumerate(arrays):
        new[nnz : x.nnz + nnz] = dim
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
        If `x` doesn't have zero fill-values.

    See Also
    --------
    - [`numpy.triu`][] : NumPy equivalent function
    """
    from .core import COO

    check_zero_fill_value(x)

    if not x.ndim >= 2:
        raise NotImplementedError("sparse.triu is not implemented for scalars or 1-D arrays.")

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
        If `x` doesn't have zero fill-values.

    See Also
    --------
    - [`numpy.tril`][] : NumPy equivalent function
    """
    from .core import COO

    check_zero_fill_value(x)

    if not x.ndim >= 2:
        raise NotImplementedError("sparse.tril is not implemented for scalars or 1-D arrays.")

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
    - [`sparse.COO.sum`][] : Function without ``NaN`` skipping.
    - [`numpy.nansum`][] : Equivalent Numpy function.
    """
    assert out is None
    x = asCOO(x, name="nansum")
    return nanreduce(x, np.add, axis=axis, keepdims=keepdims, dtype=dtype)


def nanmean(x, axis=None, keepdims=False, dtype=None, out=None):
    """
    Performs a `NaN` skipping mean operation along the given axes. Uses all axes by default.

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
    - [`sparse.COO.mean`][] : Function without `NaN` skipping.
    - [`numpy.nanmean`][] : Equivalent Numpy function.
    """
    assert out is None
    x = asCOO(x, name="nanmean")

    if not (np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating)):
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
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=1)

    num = np.sum(x2, axis=axis, dtype=dtype, keepdims=keepdims)

    with np.errstate(invalid="ignore", divide="ignore"):
        if num.ndim:
            return np.true_divide(num, den, casting="unsafe")
        return (num / den).astype(dtype if dtype is not None else x.dtype)


def nanmax(x, axis=None, keepdims=False, dtype=None, out=None):
    """
    Maximize along the given axes, skipping `NaN` values. Uses all axes by default.

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
    - [`sparse.COO.max`][] : Function without `NaN` skipping.
    - [`numpy.nanmax`][] : Equivalent Numpy function.
    """
    assert out is None
    x = asCOO(x, name="nanmax")

    ar = x.reduce(np.fmax, axis=axis, keepdims=keepdims, dtype=dtype)

    if (isscalar(ar) and np.isnan(ar)) or np.isnan(ar.data).any():
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=1)

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
    - [`sparse.COO.min`][] : Function without `NaN` skipping.
    - [`numpy.nanmin`][] : Equivalent Numpy function.
    """
    assert out is None
    x = asCOO(x, name="nanmin")

    ar = x.reduce(np.fmin, axis=axis, keepdims=keepdims, dtype=dtype)

    if (isscalar(ar) and np.isnan(ar)) or np.isnan(ar.data).any():
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=1)

    return ar


def nanprod(x, axis=None, keepdims=False, dtype=None, out=None):
    """
    Performs a product operation along the given axes, skipping `NaN` values.
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
    - [`sparse.COO.prod`][] : Function without `NaN` skipping.
    - [`numpy.nanprod`][] : Equivalent Numpy function.
    """
    assert out is None
    x = asCOO(x)
    return nanreduce(x, np.multiply, axis=axis, keepdims=keepdims, dtype=dtype)


def where(condition, x=None, y=None):
    """
    Select values from either ``x`` or ``y`` depending on ``condition``.
    If ``x`` and ``y`` are not given, returns indices where ``condition``
    is nonzero.

    Performs the equivalent of [`numpy.where`][].

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
        The output array with selected values if `x` and `y` are given;
        else where the array is nonzero.

    Raises
    ------
    ValueError
        If the operation would produce a dense result; or exactly one of
        `x` and `y` are given.

    See Also
    --------
    [`numpy.where`][] : Equivalent Numpy function.
    """
    from .._umath import elemwise

    x_given = x is not None
    y_given = y is not None

    if not (x_given or y_given):
        check_zero_fill_value(condition)
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
    index_array : numpy.ndarray

    See Also
    --------
    [`sparse.where`][], [`sparse.COO.nonzero`][]

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


def argmax(x, /, *, axis=None, keepdims=False):
    """
    Returns the indices of the maximum values along a specified axis.
    When the maximum value occurs multiple times, only the indices
    corresponding to the first occurrence are returned.

    Parameters
    ----------
    x : SparseArray
        Input array. The fill value must be ``0.0`` and all non-zero values
        must be greater than ``0.0``.
    axis : int, optional
        Axis along which to search. If ``None``, the function must return
        the index of the maximum value of the flattened array. Default: ``None``.
    keepdims : bool, optional
        If ``True``, the reduced axes (dimensions) must be included in the result
        as singleton dimensions, and, accordingly, the result must be compatible
        with the input array. Otherwise, if ``False``, the reduced axes (dimensions)
        must not be included in the result. Default: ``False``.

    Returns
    -------
    out : numpy.ndarray
        If ``axis`` is ``None``, a zero-dimensional array containing the index of
        the first occurrence of the maximum value. Otherwise, a non-zero-dimensional
        array containing the indices of the maximum values.
    """
    return _arg_minmax_common(x, axis=axis, keepdims=keepdims, mode="max")


def argmin(x, /, *, axis=None, keepdims=False):
    """
    Returns the indices of the minimum values along a specified axis.
    When the minimum value occurs multiple times, only the indices
    corresponding to the first occurrence are returned.

    Parameters
    ----------
    x : SparseArray
        Input array. The fill value must be ``0.0`` and all non-zero values
        must be less than ``0.0``.
    axis : int, optional
        Axis along which to search. If ``None``, the function must return
        the index of the minimum value of the flattened array. Default: ``None``.
    keepdims : bool, optional
        If ``True``, the reduced axes (dimensions) must be included in the result
        as singleton dimensions, and, accordingly, the result must be compatible
        with the input array. Otherwise, if ``False``, the reduced axes (dimensions)
        must not be included in the result. Default: ``False``.

    Returns
    -------
    out : numpy.ndarray
        If ``axis`` is ``None``, a zero-dimensional array containing the index of
        the first occurrence of the minimum value. Otherwise, a non-zero-dimensional
        array containing the indices of the minimum values.
    """
    return _arg_minmax_common(x, axis=axis, keepdims=keepdims, mode="min")


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
    Performs an `NaN` skipping reduction on this array. See the documentation
    on [`sparse.COO.reduce`][] for examples.

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
    [`sparse.COO.reduce`][] : Similar method without `NaN` skipping functionality.
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

    a = as_coo(a)

    # roll flattened array
    if axis is None:
        return roll(a.reshape((-1,)), shift, 0).reshape(a.shape)

    # roll across specified axis
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
        raise ValueError("If 'shift' is a 1D sequence, 'axis' must have equal length.")

    if not can_store(a.coords.dtype, max(a.shape + shift)):
        raise ValueError(
            f"cannot roll with coords.dtype {a.coords.dtype} and shift {shift}. Try casting coords to a larger dtype."
        )

    # shift elements
    coords, data = np.copy(a.coords), np.copy(a.data)
    try:
        for sh, ax in zip(shift, axis, strict=True):
            coords[ax] += sh
            coords[ax] %= a.shape[ax]
    except TypeError as e:
        if is_unsigned_dtype(coords.dtype):
            raise ValueError(
                f"rolling with coords.dtype as {coords.dtype} is not safe. Try using a signed dtype."
            ) from e

    return COO(
        coords,
        data=data,
        shape=a.shape,
        has_duplicates=False,
        fill_value=a.fill_value,
    )


def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    Extract diagonal from a COO array. The equivalent of [`numpy.diagonal`][].

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
    >>> x = sparse.as_coo(np.arange(9).reshape(3, 3))
    >>> sparse.diagonal(x).todense()
    array([0, 4, 8])
    >>> sparse.diagonal(x, offset=1).todense()
    array([1, 5])

    >>> x = sparse.as_coo(np.arange(12).reshape((2, 3, 2)))
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
    [`numpy.diagonal`][] : NumPy equivalent function
    """
    from .core import COO

    if a.shape[axis1] != a.shape[axis2]:
        raise ValueError("a.shape[axis1] != a.shape[axis2]")

    diag_axes = [axis for axis in range(len(a.shape)) if axis != axis1 and axis != axis2] + [axis1]
    diag_shape = [a.shape[axis] for axis in diag_axes]
    diag_shape[-1] -= abs(offset)

    diag_idx = _diagonal_idx(a.coords, axis1, axis2, offset)

    diag_coords = [a.coords[axis][diag_idx] for axis in diag_axes]
    diag_data = a.data[diag_idx]

    return COO(diag_coords, diag_data, diag_shape)


def diagonalize(a, axis=0):
    """
    Diagonalize a COO array. The new dimension is appended at the end.

    !!! warning

        [`sparse.diagonalize`][] is not [numpy][] compatible as there is no direct [numpy][] equivalent. The
        API may change in the future.

    Parameters
    ----------
    a : Union[COO, np.ndarray, scipy.sparse.spmatrix]
        The array to diagonalize.
    axis : int, optional
        The axis to diagonalize. Defaults to first axis (0).

    Examples
    --------
    >>> import sparse
    >>> x = sparse.as_coo(np.arange(1, 4))
    >>> sparse.diagonalize(x).todense()
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]])

    >>> x = sparse.as_coo(np.arange(24).reshape((2, 3, 4)))
    >>> x_diag = sparse.diagonalize(x, axis=1)
    >>> x_diag.shape
    (2, 3, 4, 3)

    [`sparse.diagonalize`][] is the inverse of [`sparse.diagonal`][]

    >>> a = sparse.random((3, 3, 3, 3, 3), density=0.3)
    >>> a_diag = sparse.diagonalize(a, axis=2)
    >>> (sparse.diagonal(a_diag, axis1=2, axis2=5) == a.transpose([0, 1, 3, 4, 2])).all()
    np.True_

    Returns
    -------
    out: COO
        The result of the operation.

    See Also
    --------
    [`numpy.diag`][] : NumPy equivalent for 1D array
    """
    from .core import COO, as_coo

    a = as_coo(a)

    diag_shape = a.shape + (a.shape[axis],)
    diag_coords = np.vstack([a.coords, a.coords[axis]])

    return COO(diag_coords, a.data, diag_shape)


def isposinf(x, out=None):
    """
    Test element-wise for positive infinity, return result as sparse `bool` array.

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
    [`numpy.isposinf`][] : The NumPy equivalent
    """
    from sparse import elemwise

    return elemwise(lambda x, out=None, dtype=None: np.isposinf(x, out=out), x, out=out)


def isneginf(x, out=None):
    """
    Test element-wise for negative infinity, return result as sparse `bool` array.

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
    [`numpy.isneginf`][] : The NumPy equivalent
    """
    from sparse import elemwise

    return elemwise(lambda x, out=None, dtype=None: np.isneginf(x, out=out), x, out=out)


def result_type(*arrays_and_dtypes):
    """Returns the type that results from applying the NumPy type promotion rules to the
    arguments.

    See Also
    --------
    [`numpy.result_type`][] : The NumPy equivalent
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
    return np.array([i for i in range(len(coordlist[axis1])) if coordlist[axis1][i] + offset == coordlist[axis2][i]])


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
    >>> sparse.clip(x, a_min=1, a_max=2).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([1, 1, 1, 1, 2, 2])

    See Also
    --------
    numpy.clip : Equivalent NumPy function
    """
    a = asCOO(a, name="clip")
    return a.clip(a_min, a_max)


def expand_dims(x, /, *, axis=0):
    """
    Expands the shape of an array by inserting a new axis (dimension) of size
    one at the position specified by ``axis``.

    Parameters
    ----------
    a : COO
        Input COO array.
    axis : int
        Position in the expanded axes where the new axis is placed.

    Returns
    -------
    result : COO
        An expanded output COO array having the same data type as ``x``.

    Examples
    --------
    >>> import sparse
    >>> x = sparse.COO.from_numpy([[1, 0, 0, 0, 2, -3]])
    >>> x.shape
    (1, 6)
    >>> y1 = sparse.expand_dims(x, axis=1)
    >>> y1.shape
    (1, 1, 6)
    >>> y2 = sparse.expand_dims(x, axis=2)
    >>> y2.shape
    (1, 6, 1)

    """

    x = _validate_coo_input(x)

    if not isinstance(axis, int):
        raise IndexError(f"Invalid axis position: {axis}")

    axis = normalize_axis(axis, x.ndim + 1)

    new_coords = np.insert(x.coords, obj=axis, values=np.zeros(x.nnz, dtype=np.intp), axis=0)
    new_shape = list(x.shape)
    new_shape.insert(axis, 1)
    new_shape = tuple(new_shape)

    from .core import COO

    return COO(
        new_coords,
        x.data,
        shape=new_shape,
        fill_value=x.fill_value,
    )


def flip(x, /, *, axis=None):
    """
    Reverses the order of elements in an array along the given axis.

    The shape of the array is preserved.

    Parameters
    ----------
    a : COO
        Input COO array.
    axis : int or tuple of ints, optional
        Axis (or axes) along which to flip. If ``axis`` is ``None``, the function must
        flip all input array axes. If ``axis`` is negative, the function must count from
        the last dimension. If provided more than one axis, the function must flip only
        the specified axes. Default: ``None``.

    Returns
    -------
    result : COO
        An output array having the same data type and shape as ``x`` and whose elements,
        relative to ``x``, are reordered.

    """

    x = _validate_coo_input(x)

    if axis is None:
        axis = range(x.ndim)
    if not isinstance(axis, Iterable):
        axis = (axis,)

    new_coords = x.coords.copy()
    for ax in axis:
        new_coords[ax, :] = x.shape[ax] - 1 - x.coords[ax, :]

    from .core import COO

    return COO(
        new_coords,
        x.data,
        shape=x.shape,
        fill_value=x.fill_value,
    )


# Array API set functions


class UniqueCountsResult(NamedTuple):
    values: np.ndarray
    counts: np.ndarray


def unique_counts(x, /):
    """
    Returns the unique elements of an input array `x`, and the corresponding
    counts for each unique element in `x`.

    Parameters
    ----------
    x : COO
        Input COO array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : namedtuple
        The result containing:
        * values - The unique elements of an input array.
        * counts - The corresponding counts for each unique element.

    Raises
    ------
    ValueError
        If the input array is in a different format than COO.

    Examples
    --------
    >>> import sparse
    >>> x = sparse.COO.from_numpy([1, 0, 2, 1, 2, -3])
    >>> sparse.unique_counts(x)
    UniqueCountsResult(values=array([-3,  0,  1,  2]), counts=array([1, 1, 2, 2]))
    """

    x = _validate_coo_input(x)

    x = x.flatten()
    values, counts = np.unique(x.data, return_counts=True)
    if x.nnz < x.size:
        values = np.concatenate([[x.fill_value], values])
        counts = np.concatenate([[x.size - x.nnz], counts])
        sorted_indices = np.argsort(values)
        values[sorted_indices] = values.copy()
        counts[sorted_indices] = counts.copy()

    return UniqueCountsResult(values, counts)


def unique_values(x, /):
    """
    Returns the unique elements of an input array `x`.

    Parameters
    ----------
    x : COO
        Input COO array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : ndarray
        The unique elements of an input array.

    Raises
    ------
    ValueError
        If the input array is in a different format than COO.

    Examples
    --------
    >>> import sparse
    >>> x = sparse.COO.from_numpy([1, 0, 2, 1, 2, -3])
    >>> sparse.unique_values(x)
    array([-3,  0,  1,  2])
    """

    x = _validate_coo_input(x)

    x = x.flatten()
    values = np.unique(x.data)
    if x.nnz < x.size:
        values = np.sort(np.concatenate([[x.fill_value], values]))
    return values


def sort(x, /, *, axis=-1, descending=False, stable=False):
    """
    Returns a sorted copy of an input array ``x``.

    Parameters
    ----------
    x : SparseArray
        Input array. Should have a real-valued data type.
    axis : int
        Axis along which to sort. If set to ``-1``, the function must sort along
        the last axis. Default: ``-1``.
    descending : bool
        Sort order. If ``True``, the array must be sorted in descending order (by value).
        If ``False``, the array must be sorted in ascending order (by value).
        Default: ``False``.
    stable : bool
        Whether the sort is stable. Only ``False`` is supported currently.

    Returns
    -------
    out : COO
        A sorted array.

    Raises
    ------
    ValueError
        If the input array isn't and can't be converted to COO format.

    Examples
    --------
    >>> import sparse
    >>> x = sparse.COO.from_numpy([1, 0, 2, 0, 2, -3])
    >>> sparse.sort(x).todense()
    array([-3,  0,  0,  1,  2,  2])
    >>> sparse.sort(x, descending=True).todense()
    array([ 2,  2,  1,  0,  0, -3])

    """
    from .._common import moveaxis
    from .core import COO

    x = _validate_coo_input(x)

    if stable:
        raise ValueError("`stable=True` isn't currently supported.")

    original_ndim = x.ndim
    if x.ndim == 1:
        x = x[None, :]
        axis = -1

    x = moveaxis(x, source=axis, destination=-1)
    x_shape = x.shape
    x = x.reshape((-1, x_shape[-1]))

    new_coords, new_data = _sort_coo(x.coords, x.data, x.fill_value, sort_axis_len=x_shape[-1], descending=descending)

    x = COO(new_coords, new_data, x.shape, has_duplicates=False, sorted=True, fill_value=x.fill_value)

    x = x.reshape(x_shape[:-1] + (x_shape[-1],))
    x = moveaxis(x, source=-1, destination=axis)

    return x if original_ndim == x.ndim else x.squeeze()


def take(x, indices, /, *, axis=None):
    """
    Returns elements of an array along an axis.

    Parameters
    ----------
    x : SparseArray
        Input array.
    indices : ndarray
        Array indices. The array must be one-dimensional and have an integer data type.
    axis : int
        Axis over which to select values. If ``axis`` is negative, the function must
        determine the axis along which to select values by counting from the last dimension.
        For ``None``, the flattened input array is used. Default: ``None``.

    Returns
    -------
    out : COO
        A COO array with requested indices.

    Raises
    ------
    ValueError
        If the input array isn't and can't be converted to COO format.
    """

    x = _validate_coo_input(x)

    if axis is None:
        x = x.flatten()
        return x[indices]

    axis = normalize_axis(axis, x.ndim)
    full_index = (slice(None),) * axis + (indices, ...)
    return x[full_index]


def _validate_coo_input(x: Any):
    from .._common import _is_scipy_sparse_obj
    from .core import COO

    if _is_scipy_sparse_obj(x):
        x = COO.from_scipy_sparse(x)
    elif not isinstance(x, SparseArray):
        raise ValueError(f"Input must be an instance of SparseArray, but it's {type(x)}.")
    elif not isinstance(x, COO):
        x = x.asformat(COO)

    return x


@numba.jit(nopython=True, nogil=True)
def _sort_coo(
    coords: np.ndarray, data: np.ndarray, fill_value: float, sort_axis_len: int, descending: bool
) -> tuple[np.ndarray, np.ndarray]:
    assert coords.shape[0] == 2
    group_coords = coords[0, :]
    sort_coords = coords[1, :]

    data = data.copy()
    result_indices = np.empty_like(sort_coords)

    # We iterate through all groups and sort each one of them.
    # first and last index of a group is tracked.
    prev_group = -1
    group_first_idx = -1
    group_last_idx = -1
    # We add `-1` sentinel to know when the last group ends
    for idx, group in enumerate(np.append(group_coords, -1)):
        if group == prev_group:
            continue

        if prev_group != -1:
            group_last_idx = idx

            group_slice = slice(group_first_idx, group_last_idx)
            group_size = group_last_idx - group_first_idx

            # SORT VALUES
            if group_size > 1:
                # np.sort in numba doesn't support `np.sort`'s arguments so `stable`
                # keyword can't be supported.
                # https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html#other-methods
                data[group_slice] = np.sort(data[group_slice])
                if descending:
                    data[group_slice] = data[group_slice][::-1]

            # SORT INDICES
            fill_value_count = sort_axis_len - group_size
            indices = np.arange(group_size)
            # find a place where fill_value would be
            for pos in range(group_size):
                if (not descending and fill_value < data[group_slice][pos]) or (
                    descending and fill_value > data[group_slice][pos]
                ):
                    indices[pos:] += fill_value_count
                    break
            result_indices[group_first_idx:group_last_idx] = indices

        prev_group = group
        group_first_idx = idx

    return np.vstack((group_coords, result_indices)), data


@numba.jit(nopython=True, nogil=True)
def _compute_minmax_args(
    coords: np.ndarray,
    data: np.ndarray,
    reduce_size: int,
    fill_value: float,
    max_mode_flag: bool,
) -> tuple[np.ndarray, np.ndarray]:
    assert coords.shape[0] == 2
    reduce_coords = coords[0, :]
    index_coords = coords[1, :]

    result_indices = np.unique(index_coords)
    result_data = []

    # we iterate through each trace
    for result_index in np.nditer(result_indices):
        mask = index_coords == result_index
        masked_reduce_coords = reduce_coords[mask]
        masked_data = data[mask]

        compared_data = operator.gt(masked_data, fill_value) if max_mode_flag else operator.lt(masked_data, fill_value)

        if np.any(compared_data) or len(masked_data) == reduce_size:
            # best value is a non-fill value
            best_arg = np.argmax(masked_data) if max_mode_flag else np.argmin(masked_data)
            result_data.append(masked_reduce_coords[best_arg])
        else:
            # best value is a fill value, find the first occurrence of it
            current_coord = np.array(-1, dtype=coords.dtype)
            found = False
            for idx, new_coord in enumerate(np.nditer(np.sort(masked_reduce_coords))):
                # there is at least one fill value between consecutive non-fill values
                if new_coord - current_coord > 1:
                    result_data.append(idx)
                    found = True
                    break
                current_coord = new_coord
            # get the first fill value after all non-fill values
            if not found:
                result_data.append(current_coord + 1)

    return (result_indices, np.array(result_data, dtype=np.intp))


def _arg_minmax_common(
    x: SparseArray,
    axis: int | None,
    keepdims: bool,
    mode: str,
):
    """
    Internal implementation for argmax and argmin functions.
    """
    assert mode in ("max", "min")
    max_mode_flag = mode == "max"

    x = _validate_coo_input(x)

    if not isinstance(axis, int | type(None)):
        raise ValueError(f"`axis` must be `int` or `None`, but it's: {type(axis)}.")
    if isinstance(axis, int) and axis >= x.ndim:
        raise ValueError(f"`axis={axis}` is out of bounds for array of dimension {x.ndim}.")
    if x.ndim == 0:
        raise ValueError("Input array must be at least 1-D, but it's 0-D.")

    # If `axis` is None then we need to flatten the input array and memorize
    # the original dimensionality for the final reshape operation.
    axis_none_original_ndim: int | None = None
    if axis is None:
        axis_none_original_ndim = x.ndim
        x = x.reshape(-1)[:, None]
        axis = 0

    # A 1-D array must have one more singleton dimension.
    if axis == 0 and x.ndim == 1:
        x = x[:, None]

    # We need to move `axis` to the front.
    new_transpose = list(range(x.ndim))
    new_transpose.insert(0, new_transpose.pop(axis))
    new_transpose = tuple(new_transpose)

    # And reshape it to 2-D (reduce axis, the rest of axes flattened)
    new_shape = list(x.shape)
    new_shape.insert(0, new_shape.pop(axis))
    new_shape = tuple(new_shape)

    x = x.transpose(new_transpose)
    x = x.reshape((new_shape[0], np.prod(new_shape[1:])))

    # Compute max/min arguments
    result_indices, result_data = _compute_minmax_args(
        x.coords.copy(),
        x.data.copy(),
        reduce_size=x.shape[0],
        fill_value=x.fill_value,
        max_mode_flag=max_mode_flag,
    )

    from .core import COO

    result = COO(result_indices, result_data, shape=(x.shape[1],), fill_value=0, prune=True)

    # Let's reshape the result to the original shape.
    result = result.reshape((1, *new_shape[1:]))
    new_transpose = list(range(result.ndim))
    new_transpose.insert(axis, new_transpose.pop(0))
    result = result.transpose(new_transpose)

    # If `axis=None` we need to reshape flattened array into original dimensionality.
    if axis_none_original_ndim is not None:
        result = result.reshape([1 for _ in range(axis_none_original_ndim)])

    return result if keepdims else result.squeeze()


def matrix_transpose(x, /):
    """
    Transposes a matrix or a stack of matrices.

    Parameters
    ----------
    x : SparseArray
        Input array.

    Returns
    -------
    out : COO
        Transposed COO array.

    Raises
    ------
    ValueError
        If the input array isn't and can't be converted to COO format, or if ``x.ndim < 2``.
    """
    if hasattr(x, "ndim") and x.ndim < 2:
        raise ValueError("`x.ndim >= 2` must hold.")
    x = _validate_coo_input(x)
    transpose_axes = list(range(x.ndim))
    transpose_axes[-2:] = transpose_axes[-2:][::-1]

    return x.transpose(transpose_axes)
