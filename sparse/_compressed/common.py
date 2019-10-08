from itertools import zip_longest
import numpy as np 
import numba





def diagonal(a, offset=0, axis1=0, axis2=1):
    
    if a.shape[axis1] != a.shape[axis2]:
        raise ValueError("a.shape[axis1] != a.shape[axis2]")

    diag_axes = [
        axis for axis in range(len(a.shape)) if axis != axis1 and axis != axis2
    ] + [axis1]
    diag_shape = [a.shape[axis] for axis in diag_axes]
    diag_shape[-1] -= abs(offset)
    
    # convert to linearized coordinates
    rows, cols = [], []
    operations = np.prod(diag_shape)
    current_idx = np.zeros(a.shape)
    current_idx[axis1] = offset
    a1 = offset
    axes = list(reversed(diag_axes[:-1]))
    first_axis = axes[0]
    for _ in range(operations):
        if a1 == a.shape[axis1]:
            a1 = offset
            current_idx[axis1] = offset
            current_idx[axis2] = 0
            current_idx[first_axis] +=1
            for i in range(len(axes-1)):
                if current_idx[axes[i]] == a.shape[axes[i]]:
                    current_idx[axes[i]] = 0
                    current_idx[axes[i+1]] += 1

        ind = np.ravel_multi_index(current_idx, a.reordered_shape)
        row, col = np.unravel_index(ind, a.compressed_shape)
        rows.append(row)
        cols.append(col)
        a1 += 1
        current_idx[axis1] = a1
        current_idx[axis2] += 1
    
    # search the diagonals
    coords = []
    mask = []
    count = 0
    for r in rows:
        current_row = a.indices[a.indptr[r:r+1]]
        for c in cols:
            s = np.searchsorted(current_row, c)
            if not (s >= current_row.size or current_row[s] != col[c]):
                s += a.indptr[r]
                mask.append(s)
                coords.append(count)
            count += 1
    coords = np.array(coords)
    return GCXS.from_coo(COO(coords[None,:],a.data[mask], fill_value=a.fill_value).reshape(diag_shape))

        

@numba.jit(nopython=True,nogil=True)
def _diagonal_idx(indices, indptr, axis1, axis2, offset):

     # convert from nd
     linearized = np.ravel_multi_index()

def matmul(a, b):
    pass

def dot(a, b):
    pass

def tensordot(a, b, axes=2):
    pass

def kron(a, b):
    from .._coo.umath import _cartesian_product
    
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

def concatenate(arrays, axis=0, compressed_axes=(0,)):
    
    check_consistent_fill_value(arrays)
    arrays = [arr if isinstance(arr, GCXS) else GCXS(arr) for arr in arrays]
    axis = normalize_axis(axis, arrays[0].ndim)
    nnz = 0
    dim = sum(x.shape[axis] for x in arrays)
    shape = list(arrays[0].shape)
    shape[axis] = dim
    assert all(
        x.shape[ax] == arrays[0].shape[ax]
        for x in arrays
        for ax in set(range(arrays[0].ndim)) - {axis}
    )
    # arrays may have different compressed_axes
    # flatten to have a better coordinate system
    arrays = [arr.flatten() for arr in arrays]
    indices = np.concatenate([arr.indices for arr in arrays])
    data = np.concatenate([arr.data for arr in arrays])

    dim = 0
    for x in arrays:
        if dim:
            indices[nnz : x.nnz + nnz] += dim
        dim += x.shape[axis]
        nnz += x.nnz
    
    if axis != 0:
        order = np.argsort(indices, kind='mergesort')
        indices = indices[order]
        data = data[order]

    return GCXS((data, indices, ()),
         fill_value=arrays[0].fill_value).reshape(shape,
          compressed_axes=compressed_axes)

def stack(arrays, axis=0):

    from .compressed import GCXS
    check_consistent_fill_value(arrays)
    arrays = [arr if isinstance(arr, GCXS) else GCXS(arr) for arr in arrays]
    axis = normalize_axis(axis, arrays[0].ndim)
    nnz = 0
    shape = list(arrays[0].shape)
    shape.insert(len(arrays), axis)
    assert all(
        x.shape[ax] == arrays[0].shape[ax]
        for x in arrays
        for ax in set(range(arrays[0].ndim)) - {axis}
    )
    # arrays may have different compressed_axes
    # flatten to have a better coordinate system
    arrays = [arr.flatten() for arr in arrays]
    indices = np.concatenate([arr.indices for arr in arrays])
    data = np.concatenate([arr.data for arr in arrays])

    dim = 0
    for x in arrays:
        if dim:
            indices[nnz : x.nnz + nnz] += dim
        dim += x.shape[axis]
        nnz += x.nnz
    
    if axis != 0:
        order = np.argsort(indices, kind='mergesort')
        indices = indices[order]
        data = data[order]

    return GCXS((data, indices, ()),
         fill_value=arrays[0].fill_value).reshape(shape,
          compressed_axes=compressed_axes) 

def where(condition, x=None, y=None):
    pass 

def eye(N, M=None, k=0, dtype=float, compressed_axis=0):
    
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
    
    if compressed_axis==0:
        indptr = np.empty(N, dtype=np.intp)
        indptr[0] = 0
        np.cumsum(np.bincount(n_coords, minlength=N), out=indptr[1:])
        indices = m_coords
    else:
        indptr = np.empty(M, dtype=np.intp)
        indptr[0] = 0
        np.cumsum(np.bincount(m_coords, minlength=M), out=indptr[1:])
        indices = n_coords

    data = np.array(1, dtype=dtype)
    return GCXS((data,indices,indptr),
                compressed_axes=(compressed_axis,),
                dtype=dtype,
                fill_value=0)

def full(shape, fill_value, dtype=None):
    """Return a GCXS array of given shape and type, filled with `fill_value`.
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

    if dtype is None:
        dtype = np.array(fill_value).dtype
    if not isinstance(shape, tuple):
        shape = (shape,)
    data = np.empty(0, dtype=dtype)
    indices = np.empty((0, 0), dtype=np.intp)
    indptr = np.empty((0, 0), dtype=np.intp)
    return GCXS(
        (data,
        indices,
        indptr),
        shape=shape,
        fill_value=fill_value,
    )

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

def zeros_like(a, dtype=float):
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
 