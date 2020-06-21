import numpy as np
from collections.abc import Iterable
from sparse import COO

from ._utils import check_compressed_axes, normalize_axis
from ._coo import (
    clip,
    tensordot,
    dot,
    matmul,
    triu,
    tril,
    where,
    nansum,
    nanmean,
    nanprod,
    nanmin,
    nanmax,
    nanreduce,
    roll,
    kron,
    argwhere,
    isposinf,
    isneginf,
    result_type,
    diagonal,
    diagonalize,
    elemwise,
    as_coo,
)


def stack(arrays, axis=0, compressed_axes=None):
    """
    Stack the input arrays along the given dimension.

    Parameters
    ----------
    arrays : Iterable[SparseArray]
        The input arrays to stack.
    axis : int, optional
        The axis along which to stack the input arrays.
    compressed_axes : iterable, optional
        The axes to compress if returning a GCXS array.

    Returns
    -------
    SparseArray
        The output stacked array.

    Raises
    ------
    ValueError
        If all elements of :code:`arrays` don't have the same fill-value.

    See Also
    --------
    numpy.stack : NumPy equivalent function
    """
    from ._coo import COO

    if any(isinstance(arr, COO) for arr in arrays):
        from ._coo import stack as coo_stack

        return coo_stack(arrays, axis)
    else:
        from ._compressed import stack as gcxs_stack

        return gcxs_stack(arrays, axis, compressed_axes)


def concatenate(arrays, axis=0, compressed_axes=None):
    """
    Concatenate the input arrays along the given dimension.

    Parameters
    ----------
    arrays : Iterable[SparseArray]
        The input arrays to concatenate.
    axis : int, optional
        The axis along which to concatenate the input arrays. The default is zero.
    compressed_axes : iterable, optional
        The axes to compress if returning a GCXS array.

    Returns
    -------
    SparseArray
        The output concatenated array.

    Raises
    ------
    ValueError
        If all elements of :code:`arrays` don't have the same fill-value.

    See Also
    --------
    numpy.concatenate : NumPy equivalent function
    """
    from ._coo import COO

    if any(isinstance(arr, COO) for arr in arrays):
        from ._coo import concatenate as coo_concat

        return coo_concat(arrays, axis)
    else:
        from ._compressed import concatenate as gcxs_concat

        return gcxs_concat(arrays, axis, compressed_axes)


def eye(N, M=None, k=0, dtype=float, format="coo", compressed_axes=None):
    """Return a 2-D array in the specified format with ones on the diagonal and zeros elsewhere.

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
    format : str, optional
        A format string.
    compressed_axes : iterable, optional
        The axes to compress if returning a GCXS array.

    Returns
    -------
    I : SparseArray of shape (N, M)
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
    from sparse import COO

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

    return COO(
        coords, data=data, shape=(N, M), has_duplicates=False, sorted=True
    ).asformat(format, compressed_axes=compressed_axes)


def full(shape, fill_value, dtype=None, format="coo", compressed_axes=None):
    """Return a SparseArray of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array. The default, `None`, means
        `np.array(fill_value).dtype`.
    format : str, optional
        A format string.
    compressed_axes : iterable, optional
        The axes to compress if returning a GCXS array.
    Returns
    -------
    out : SparseArray
        Array of `fill_value` with the given shape and dtype.

    Examples
    --------
    >>> full(5, 9).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([9, 9, 9, 9, 9])

    >>> full((2, 2), 9, dtype=float).todense()  # doctest: +SKIP
    array([[9., 9.],
           [9., 9.]])
    """
    from sparse import COO

    if dtype is None:
        dtype = np.array(fill_value).dtype
    if not isinstance(shape, tuple):
        shape = (shape,)
    if compressed_axes is not None:
        check_compressed_axes(shape, compressed_axes)
    data = np.empty(0, dtype=dtype)
    coords = np.empty((len(shape), 0), dtype=np.intp)
    return COO(
        coords,
        data=data,
        shape=shape,
        fill_value=fill_value,
        has_duplicates=False,
        sorted=True,
    ).asformat(format, compressed_axes=compressed_axes)


def full_like(a, fill_value, dtype=None, format=None, compressed_axes=None):
    """Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of the result will match those of `a`.
    dtype : data-type, optional
        Overrides the data type of the result.
    format : str, optional
        A format string.
    compressed_axes : iterable, optional
        The axes to compress if returning a GCXS array.

    Returns
    -------
    out : SparseArray
        Array of `fill_value` with the same shape and type as `a`.

    Examples
    --------
    >>> x = np.ones((2, 3), dtype='i8')
    >>> full_like(x, 9.0).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[9, 9, 9],
           [9, 9, 9]])
    """
    if format is None and not isinstance(a, np.ndarray):
        format = type(a).__name__.lower()
    else:
        format = "coo"
    if hasattr(a, "compressed_axes") and compressed_axes is None:
        compressed_axes = a.compressed_axes
    return full(
        a.shape,
        fill_value,
        dtype=(a.dtype if dtype is None else dtype),
        format=format,
        compressed_axes=compressed_axes,
    )


def zeros(shape, dtype=float, format="coo", compressed_axes=None):
    """Return a SparseArray of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    format : str, optional
        A format string.
    compressed_axes : iterable, optional
        The axes to compress if returning a GCXS array.

    Returns
    -------
    out : SparseArray
        Array of zeros with the given shape and dtype.

    Examples
    --------
    >>> zeros(5).todense()  # doctest: +SKIP
    array([0., 0., 0., 0., 0.])

    >>> zeros((2, 2), dtype=int).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0],
           [0, 0]])
    """
    if compressed_axes is not None:
        check_compressed_axes(shape, compressed_axes)
    return full(shape, 0, np.dtype(dtype)).asformat(
        format, compressed_axes=compressed_axes
    )


def zeros_like(a, dtype=None, format=None, compressed_axes=None):
    """Return a SparseArray of zeros with the same shape and type as ``a``.

    Parameters
    ----------
    a : array_like
        The shape and data-type of the result will match those of `a`.
    dtype : data-type, optional
        Overrides the data type of the result.
    format : str, optional
        A format string.
    compressed_axes : iterable, optional
        The axes to compress if returning a GCXS array.

    Returns
    -------
    out : SparseArray
        Array of zeros with the same shape and type as `a`.

    Examples
    --------
    >>> x = np.ones((2, 3), dtype='i8')
    >>> zeros_like(x).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0],
           [0, 0, 0]])
    """
    if format is None and not isinstance(a, np.ndarray):
        format = type(a).__name__.lower()
    else:
        format = "coo"
    if hasattr(a, "compressed_axes") and compressed_axes is None:
        compressed_axes = a.compressed_axes
    return zeros(
        a.shape,
        dtype=(a.dtype if dtype is None else dtype),
        format=format,
        compressed_axes=compressed_axes,
    )


def ones(shape, dtype=float, format="coo", compressed_axes=None):
    """Return a SparseArray of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    format : str, optional
        A format string.
    compressed_axes : iterable, optional
        The axes to compress if returning a GCXS array.

    Returns
    -------
    out : SparseArray
        Array of ones with the given shape and dtype.

    Examples
    --------
    >>> ones(5).todense()  # doctest: +SKIP
    array([1., 1., 1., 1., 1.])

    >>> ones((2, 2), dtype=int).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1],
           [1, 1]])
    """
    if compressed_axes is not None:
        check_compressed_axes(shape, compressed_axes)
    return full(shape, 1, np.dtype(dtype)).asformat(
        format, compressed_axes=compressed_axes
    )


def ones_like(a, dtype=None, format=None, compressed_axes=None):
    """Return a SparseArray of ones with the same shape and type as ``a``.

    Parameters
    ----------
    a : array_like
        The shape and data-type of the result will match those of `a`.
    dtype : data-type, optional
        Overrides the data type of the result.
    format : str, optional
        A format string.
    compressed_axes : iterable, optional
        The axes to compress if returning a GCXS array.

    Returns
    -------
    out : SparseArray
        Array of ones with the same shape and type as `a`.

    Examples
    --------
    >>> x = np.ones((2, 3), dtype='i8')
    >>> ones_like(x).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 1],
           [1, 1, 1]])
    """
    if format is None and not isinstance(a, np.ndarray):
        format = type(a).__name__.lower()
    else:
        format = "coo"
    if hasattr(a, "compressed_axes") and compressed_axes is None:
        compressed_axes = a.compressed_axes
    return ones(
        a.shape,
        dtype=(a.dtype if dtype is None else dtype),
        format=format,
        compressed_axes=compressed_axes,
    )


def outer(a, b, out=None):
    """
    Return outer product of two sparse arrays.

    Parameters
    ----------
    a, b : sparse.SparseArray
        The input arrays.
    out : sparse.SparseArray
        The output array.
    
    Examples
    --------
    >>> import numpy as np
    >>> import sparse
    >>> a = sparse.COO(np.arange(4))
    >>> o = sparse.outer(a, a)
    >>> o.todense()
    array([[0, 0, 0, 0],
           [0, 1, 2, 3],
           [0, 2, 4, 6],
           [0, 3, 6, 9]])
    """
    from sparse import SparseArray, COO

    if isinstance(a, SparseArray):
        a = COO(a)
    if isinstance(b, SparseArray):
        b = COO(b)
    return np.multiply.outer(a.flatten(), b.flatten(), out=out)


def asnumpy(a, dtype=None, order=None):
    """Returns a dense numpy array from an arbitrary source array.

    Args:
        a: Arbitrary object that can be converted to :class:`numpy.ndarray`.
        order ({'C', 'F', 'A'}): The desired memory layout of the output
            array. When ``order`` is 'A', it uses 'F' if ``a`` is
            fortran-contiguous and 'C' otherwise.
    Returns:
        numpy.ndarray: Converted array on the host memory.
    """
    from ._sparse_array import SparseArray

    if isinstance(a, SparseArray):
        a = a.todense()
    return np.array(a, dtype=dtype, copy=False, order=order)


# this code was taken from numpy.moveaxis
# (cf. numpy/core/numeric.py, lines 1340-1409, v1.18.4)
# https://github.com/numpy/numpy/blob/v1.18.4/numpy/core/numeric.py#L1340-L1409
def moveaxis(a, source, destination):
    """
    Move axes of an array to new positions.

    Other axes remain in their original order.

    Parameters
    ----------
    a : COO
        The array whose axes should be reordered.
    source : int or List[int]
        Original positions of the axes to move. These must be unique.
    destination : int or List[int]
        Destination positions for each of the original axes. These must also be unique.

    Returns
    -------
    COO
        Array with moved axes.

    Examples
    --------
    >>> import numpy as np
    >>> import sparse
    >>> x = sparse.COO.from_numpy(np.ones((2, 3, 4, 5)))
    >>> sparse.moveaxis(x, (0, 1), (2, 3))
    <COO: shape=(4, 5, 2, 3), dtype=float64, nnz=120, fill_value=0.0>
    """

    if not isinstance(source, Iterable):
        source = (source,)
    if not isinstance(destination, Iterable):
        destination = (destination,)

    source = normalize_axis(source, a.ndim)
    destination = normalize_axis(destination, a.ndim)

    if len(source) != len(destination):
        raise ValueError(
            "`source` and `destination` arguments must have "
            "the same number of elements"
        )

    order = [n for n in range(a.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    result = a.transpose(order)
    return result
