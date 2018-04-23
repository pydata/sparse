from functools import reduce
import operator
import warnings

import numpy as np
import scipy.sparse

from ..sparse_array import SparseArray
from ..compatibility import range
from ..utils import isscalar, normalize_axis


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


def linear_loc(coords, shape, signed=False):
    n = reduce(operator.mul, shape, 1)
    if signed:
        n = -n
    dtype = np.min_scalar_type(n)
    out = np.zeros(coords.shape[1], dtype=dtype)
    tmp = np.zeros(coords.shape[1], dtype=dtype)
    strides = 1
    for i, d in enumerate(shape[::-1]):
        np.multiply(coords[-(i + 1), :], strides, out=tmp, dtype=dtype)
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

    See Also
    --------
    numpy.tensordot : NumPy equivalent function
    """
    # Much of this is stolen from numpy/core/numeric.py::tensordot
    # Please see license at https://github.com/numpy/numpy/blob/master/LICENSE.txt
    from .core import COO

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
    if isinstance(res, scipy.sparse.spmatrix):
        if res.nnz > reduce(operator.mul, res.shape) / 2:
            res = res.todense()
        else:
            res = COO.from_scipy_sparse(res)  # <--- modified
            res.has_duplicates = False
    if isinstance(res, np.matrix):
        res = np.asarray(res)
    return res.reshape(olda + oldb)


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

    See Also
    --------
    numpy.dot : NumPy equivalent function.
    COO.dot : Equivalent function for COO objects.
    """
    if not hasattr(a, 'ndim') or not hasattr(b, 'ndim'):
        raise NotImplementedError(
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

    if isinstance(b, COO) and not isinstance(a, COO):
        return _dot(b.T, a.T).T
    aa = a.tocsr()

    if isinstance(b, (COO, scipy.sparse.spmatrix)):
        b = b.tocsc()
    return aa.dot(b)


def concatenate(arrays, axis=0):
    """
    Concatenate the input arrays along the given dimension.

    Parameters
    ----------
    arrays : Iterable[Union[COO, numpy.ndarray, scipy.sparse.spmatrix]]
        The input arrays to concatenate.
    axis : int, optional
        The axis along which to concatenate the input arrays. The default is zero.

    Returns
    -------
    COO
        The output concatenated array.

    See Also
    --------
    numpy.concatenate : NumPy equivalent function
    """
    from .core import COO

    arrays = [x if isinstance(x, COO) else COO(x) for x in arrays]
    axis = normalize_axis(axis, arrays[0].ndim)
    assert all(x.shape[ax] == arrays[0].shape[ax]
               for x in arrays
               for ax in set(range(arrays[0].ndim)) - {axis})
    nnz = 0
    dim = sum(x.shape[axis] for x in arrays)
    shape = list(arrays[0].shape)
    shape[axis] = dim

    coords_dtype = np.min_scalar_type(max(shape) - 1) if len(shape) != 0 else np.uint8
    data = np.concatenate([x.data for x in arrays])
    coords = np.concatenate([x.coords for x in arrays], axis=1).astype(coords_dtype)

    dim = 0
    for x in arrays:
        if dim:
            coords[axis, nnz:x.nnz + nnz] += dim
        dim += x.shape[axis]
        nnz += x.nnz

    return COO(coords, data, shape=shape, has_duplicates=False,
               sorted=(axis == 0))


def stack(arrays, axis=0):
    """
    Stack the input arrays along the given dimension.

    Parameters
    ----------
    arrays : Iterable[Union[COO, numpy.ndarray, scipy.sparse.spmatrix]]
        The input arrays to stack.
    axis : int, optional
        The axis along which to stack the input arrays.

    Returns
    -------
    COO
        The output stacked array.

    See Also
    --------
    numpy.stack : NumPy equivalent function
    """
    from .core import COO

    assert len(set(x.shape for x in arrays)) == 1
    arrays = [x if isinstance(x, COO) else COO(x) for x in arrays]
    axis = normalize_axis(axis, arrays[0].ndim + 1)
    data = np.concatenate([x.data for x in arrays])
    coords = np.concatenate([x.coords for x in arrays], axis=1)
    shape = list(arrays[0].shape)
    shape.insert(axis, len(arrays))

    coords_dtype = np.min_scalar_type(max(shape) - 1) if len(shape) != 0 else np.uint8

    nnz = 0
    dim = 0
    new = np.empty(shape=(coords.shape[1],), dtype=coords_dtype)
    for x in arrays:
        new[nnz:x.nnz + nnz] = dim
        dim += 1
        nnz += x.nnz

    coords = [coords[i].astype(coords_dtype) for i in range(coords.shape[0])]
    coords.insert(axis, new)
    coords = np.stack(coords, axis=0)

    return COO(coords, data, shape=shape, has_duplicates=False,
               sorted=(axis == 0))


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

    See Also
    --------
    numpy.triu : NumPy equivalent function
    """
    from .core import COO

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

    See Also
    --------
    numpy.tril : NumPy equivalent function
    """
    from .core import COO

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
    from .core import COO

    if not np.issubdtype(array.dtype, np.floating):
        return array

    data = np.copy(array.data)
    data[np.isnan(data)] = value

    return COO(array.coords, data, shape=array.shape,
               has_duplicates=False,
               sorted=True)


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
