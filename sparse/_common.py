import numpy as np

from ._coo import (
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
    COO or GCXS
        The output stacked array.

    Raises
    ------
    ValueError
        If all elements of :code:`arrays` don't have the same fill-value.

    See Also
    --------
    numpy.stack : NumPy equivalent function
    """
    from ._compressed import GCXS

    if any(isinstance(arr, GCXS) for arr in arrays):
        from ._compressed import stack as gcxs_stack

        return gcxs_stack(arrays, axis, compressed_axes)
    else:
        from ._coo import stack as coo_stack

        return coo_stack(arrays, axis)


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
    COO or GCXS
        The output concatenated array.

    Raises
    ------
    ValueError
        If all elements of :code:`arrays` don't have the same fill-value.

    See Also
    --------
    numpy.concatenate : NumPy equivalent function
    """
    from ._compressed import GCXS

    if any(isinstance(arr, GCXS) for arr in arrays):
        from ._compressed import concatenate as gcxs_concat

        return gcxs_concat(arrays, axis, compressed_axes)
    else:
        from ._coo import concatenate as coo_concat

        return coo_concat(arrays, axis)


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
    I : [COO, GCXS, DOK] array of shape (N, M)
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
    format : str, optional
        A format string.
    compressed_axes : iterable, optional
        The axes to compress if returning a GCXS array.
    Returns
    -------
    out : [COO, GCXS, DOK]
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


def full_like(a, fill_value, dtype=None, format="coo", compressed_axes=None):
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
    out : array with the specified format [COO, GCXS, DOK]
        Array of `fill_value` with the same shape and type as `a`.

    Examples
    --------
    >>> x = np.ones((2, 3), dtype='i8')
    >>> full_like(x, 9.0).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[9, 9, 9],
           [9, 9, 9]])
    """
    return full(
        a.shape, fill_value, dtype=(a.dtype if dtype is None else dtype)
    ).asformat(format, compressed_axes=compressed_axes)


def zeros(shape, dtype=float, format="coo", compressed_axes=None):
    """Return a COO array of given shape and type, filled with zeros.

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
    out : array with the specified format [COO, GCXS, DOK]
        Array of zeros with the given shape and dtype.

    Examples
    --------
    >>> zeros(5).todense()  # doctest: +SKIP
    array([0., 0., 0., 0., 0.])

    >>> zeros((2, 2), dtype=int).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0],
           [0, 0]])
    """
    return full(shape, 0, np.dtype(dtype)).asformat(
        format, compressed_axes=compressed_axes
    )


def zeros_like(a, dtype=None, format="coo", compressed_axes=None):
    """Return a COO array of zeros with the same shape and type as ``a``.

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
    out : [COO, GCXS, DOK]
        Array of zeros with the same shape and type as `a`.

    Examples
    --------
    >>> x = np.ones((2, 3), dtype='i8')
    >>> zeros_like(x).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0],
           [0, 0, 0]])
    """
    return zeros(a.shape, dtype=(a.dtype if dtype is None else dtype)).asformat(
        format, compressed_axes=compressed_axes
    )


def ones(shape, dtype=float, format="coo", compressed_axes=None):
    """Return a COO array of given shape and type, filled with ones.

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
    out : [COO, GCXS, DOK]
        Array of ones with the given shape and dtype.

    Examples
    --------
    >>> ones(5).todense()  # doctest: +SKIP
    array([1., 1., 1., 1., 1.])

    >>> ones((2, 2), dtype=int).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1],
           [1, 1]])
    """
    return full(shape, 1, np.dtype(dtype)).asformat(
        format, compressed_axes=compressed_axes
    )


def ones_like(a, dtype=None, format="coo", compressed_axes=None):
    """Return a COO array of ones with the same shape and type as ``a``.

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
    out : [COO, GCXS, DOK]
        Array of ones with the same shape and type as `a`.

    Examples
    --------
    >>> x = np.ones((2, 3), dtype='i8')
    >>> ones_like(x).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 1],
           [1, 1, 1]])
    """
    return ones(a.shape, dtype=(a.dtype if dtype is None else dtype)).asformat(
        format, compressed_axes=compressed_axes
    )
