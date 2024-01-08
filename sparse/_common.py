import builtins
import warnings
from collections.abc import Iterable
from functools import reduce, wraps
from itertools import chain
from operator import index, mul

import numba
from numba import literal_unroll

import numpy as np
import scipy.sparse
from scipy.sparse import spmatrix

from ._coo.common import asCOO
from ._sparse_array import SparseArray
from ._utils import (
    _zero_of_dtype,
    check_zero_fill_value,
    equivalent,
    normalize_axis,
)


@numba.njit
def nan_check(*args):
    """
    Check for the NaN values in Numpy Arrays

    Parameters
    ----------
    Union[Numpy Array, Integer, Float]

    Returns
    -------
    Boolean Whether Numpy Array Contains NaN

    """
    for i in literal_unroll(args):
        ia = np.asarray(i)
        if ia.size != 0 and np.isnan(np.min(ia)):
            return True
    return False


def check_class_nan(test):
    """
    Check NaN for Sparse Arrays

    Parameters
    ----------
    test : Union[sparse.COO, sparse.GCXS, scipy.sparse.spmatrix, Numpy Ndarrays]

    Returns
    -------
    Boolean Whether Sparse Array Contains NaN

    """
    from ._compressed import GCXS
    from ._coo import COO

    if isinstance(test, (GCXS, COO)):
        return nan_check(test.fill_value, test.data)
    if isinstance(test, spmatrix):
        return nan_check(test.data)
    return nan_check(test)


def tensordot(a, b, axes=2, *, return_type=None):
    """
    Perform the equivalent of :obj:`numpy.tensordot`.

    Parameters
    ----------
    a, b : Union[SparseArray, np.ndarray, scipy.sparse.spmatrix]
        The arrays to perform the :code:`tensordot` operation on.
    axes : tuple[Union[int, tuple[int], Union[int, tuple[int]], optional
        The axes to match when performing the sum.
    return_type : {None, COO, np.ndarray}, optional
        Type of returned array.

    Returns
    -------
    Union[SparseArray, numpy.ndarray]
        The result of the operation.

    Raises
    ------
    ValueError
        If all arguments don't have zero fill-values.

    See Also
    --------
    numpy.tensordot : NumPy equivalent function
    """
    from ._compressed import GCXS

    # Much of this is stolen from numpy/core/numeric.py::tensordot
    # Please see license at https://github.com/numpy/numpy/blob/main/LICENSE.txt
    check_zero_fill_value(a, b)

    if scipy.sparse.issparse(a):
        a = GCXS.from_scipy_sparse(a)
    if scipy.sparse.issparse(b):
        b = GCXS.from_scipy_sparse(b)

    try:
        iter(axes)
    except TypeError:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(axes))
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
    if nda == 0 or ndb == 0:
        pos = int(nda != 0)
        raise ValueError(f"Input {pos} operand does not have enough dimensions")
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

    if builtins.any(dim == 0 for dim in chain(newshape_a, newshape_b)):
        res = asCOO(np.empty(olda + oldb), check=False)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            res = res.todense()

        return res

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = _dot(at, bt, return_type)
    return res.reshape(olda + oldb)


def matmul(a, b):
    """Perform the equivalent of :obj:`numpy.matmul` on two arrays.

    Parameters
    ----------
    a, b : Union[SparseArray, np.ndarray, scipy.sparse.spmatrix]
        The arrays to perform the :code:`matmul` operation on.

    Returns
    -------
    Union[SparseArray, numpy.ndarray]
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
    if not hasattr(a, "ndim") or not hasattr(b, "ndim"):
        raise TypeError(f"Cannot perform dot product on types {type(a)}, {type(b)}")

    if check_class_nan(a) or check_class_nan(b):
        warnings.warn("Nan will not be propagated in matrix multiplication", RuntimeWarning, stacklevel=1)

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
            raise ValueError("shapes of a and b are not broadcastable")

    def _matmul_recurser(a, b):
        if a.ndim == 2:
            return dot(a, b)
        res = []
        for i in range(builtins.max(a.shape[0], b.shape[0])):
            a_i = a[0] if a.shape[0] == 1 else a[i]
            b_i = b[0] if b.shape[0] == 1 else b[i]
            res.append(_matmul_recurser(a_i, b_i))
        mask = [isinstance(x, SparseArray) for x in res]
        if builtins.all(mask):
            return stack(res)

        res = [x.todense() if isinstance(x, SparseArray) else x for x in res]
        return np.stack(res)

    return _matmul_recurser(a, b)


def dot(a, b):
    """
    Perform the equivalent of :obj:`numpy.dot` on two arrays.

    Parameters
    ----------
    a, b : Union[SparseArray, np.ndarray, scipy.sparse.spmatrix]
        The arrays to perform the :code:`dot` operation on.

    Returns
    -------
    Union[SparseArray, numpy.ndarray]
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
    if not hasattr(a, "ndim") or not hasattr(b, "ndim"):
        raise TypeError(f"Cannot perform dot product on types {type(a)}, {type(b)}")

    if a.ndim == 1 and b.ndim == 1:
        if isinstance(a, SparseArray):
            a = asCOO(a)
        if isinstance(b, SparseArray):
            b = asCOO(b)
        return (a * b).sum()

    a_axis = -1
    b_axis = -2

    if b.ndim == 1:
        b_axis = -1
    return tensordot(a, b, axes=(a_axis, b_axis))


def _dot(a, b, return_type=None):
    from ._compressed import GCXS
    from ._coo import COO
    from ._sparse_array import SparseArray

    out_shape = (a.shape[0], b.shape[1])
    if builtins.all(isinstance(arr, SparseArray) for arr in [a, b]) and builtins.any(
        isinstance(arr, GCXS) for arr in [a, b]
    ):
        a = a.asformat("gcxs")
        b = b.asformat("gcxs", compressed_axes=a.compressed_axes)

    if isinstance(a, GCXS) and isinstance(b, GCXS):
        if a.nbytes > b.nbytes:
            b = b.change_compressed_axes(a.compressed_axes)
        else:
            a = a.change_compressed_axes(b.compressed_axes)

        if a.compressed_axes == (0,):  # csr @ csr
            compressed_axes = (0,)
            data, indices, indptr = _dot_csr_csr_type(a.dtype, b.dtype)(
                out_shape, a.data, b.data, a.indices, b.indices, a.indptr, b.indptr
            )
        elif a.compressed_axes == (1,):  # csc @ csc
            # a @ b = (b.T @ a.T).T
            compressed_axes = (1,)
            data, indices, indptr = _dot_csr_csr_type(b.dtype, a.dtype)(
                out_shape[::-1],
                b.data,
                a.data,
                b.indices,
                a.indices,
                b.indptr,
                a.indptr,
            )
        out = GCXS(
            (data, indices, indptr),
            shape=out_shape,
            compressed_axes=compressed_axes,
            prune=True,
        )
        if return_type == np.ndarray:
            return out.todense()
        if return_type == COO:
            return out.tocoo()
        return out

    if isinstance(a, GCXS) and isinstance(b, np.ndarray):
        if a.compressed_axes == (0,):  # csr @ ndarray
            if return_type is None or return_type == np.ndarray:
                return _dot_csr_ndarray_type(a.dtype, b.dtype)(out_shape, a.data, a.indices, a.indptr, b)
            data, indices, indptr = _dot_csr_ndarray_type_sparse(a.dtype, b.dtype)(
                out_shape, a.data, a.indices, a.indptr, b
            )
            out = GCXS(
                (data, indices, indptr),
                shape=out_shape,
                compressed_axes=(0,),
                prune=True,
            )
            if return_type == COO:
                return out.tocoo()
            return out
        if return_type is None or return_type == np.ndarray:  # csc @ ndarray
            return _dot_csc_ndarray_type(a.dtype, b.dtype)(a.shape, b.shape, a.data, a.indices, a.indptr, b)
        data, indices, indptr = _dot_csc_ndarray_type_sparse(a.dtype, b.dtype)(
            a.shape, b.shape, a.data, a.indices, a.indptr, b
        )
        compressed_axes = (1,)
        out = GCXS(
            (data, indices, indptr),
            shape=out_shape,
            compressed_axes=compressed_axes,
            prune=True,
        )
        if return_type == COO:
            return out.tocoo()
        return out

    if isinstance(a, np.ndarray) and isinstance(b, GCXS):
        at = a.view(type=np.ndarray).T
        bt = b.T  # constant-time transpose
        if b.compressed_axes == (0,):
            if return_type is None or return_type == np.ndarray:
                out = _dot_csc_ndarray_type(bt.dtype, at.dtype)(bt.shape, at.shape, bt.data, bt.indices, bt.indptr, at)
                return out.T
            data, indices, indptr = _dot_csc_ndarray_type_sparse(bt.dtype, at.dtype)(
                bt.shape, at.shape, bt.data, b.indices, b.indptr, at
            )
            out = GCXS(
                (data, indices, indptr),
                shape=out_shape,
                compressed_axes=(0,),
                prune=True,
            )
            if return_type == COO:
                return out.tocoo()
            return out

        # compressed_axes == (1,)
        if return_type is None or return_type == np.ndarray:
            return _dot_ndarray_csc_type(a.dtype, b.dtype)(out_shape, b.data, b.indices, b.indptr, a)
        data, indices, indptr = _dot_csr_ndarray_type_sparse(bt.dtype, at.dtype)(
            out_shape[::-1], bt.data, bt.indices, bt.indptr, at
        )
        out = GCXS((data, indices, indptr), shape=out_shape, compressed_axes=(1,), prune=True)
        if return_type == COO:
            return out.tocoo()
        return out

    if isinstance(a, COO) and isinstance(b, COO):
        # convert to csr
        a_indptr = np.empty(a.shape[0] + 1, dtype=np.intp)
        a_indptr[0] = 0
        np.cumsum(np.bincount(a.coords[0], minlength=a.shape[0]), out=a_indptr[1:])

        b_indptr = np.empty(b.shape[0] + 1, dtype=np.intp)
        b_indptr[0] = 0
        np.cumsum(np.bincount(b.coords[0], minlength=b.shape[0]), out=b_indptr[1:])
        coords, data = _dot_coo_coo_type(a.dtype, b.dtype)(
            out_shape, a.coords, b.coords, a.data, b.data, a_indptr, b_indptr
        )
        out = COO(
            coords,
            data,
            shape=out_shape,
            has_duplicates=False,
            sorted=False,
            prune=True,
        )

        if return_type == np.ndarray:
            return out.todense()
        if return_type == GCXS:
            return out.asformat("gcxs")
        return out

    if isinstance(a, COO) and isinstance(b, np.ndarray):
        b = b.view(type=np.ndarray).T

        if return_type is None or return_type == np.ndarray:
            return _dot_coo_ndarray_type(a.dtype, b.dtype)(a.coords, a.data, b, out_shape)
        coords, data = _dot_coo_ndarray_type_sparse(a.dtype, b.dtype)(a.coords, a.data, b, out_shape)
        out = COO(coords, data, shape=out_shape, has_duplicates=False, sorted=True)
        if return_type == GCXS:
            return out.asformat("gcxs")
        return out

    if isinstance(a, np.ndarray) and isinstance(b, COO):
        a = a.view(type=np.ndarray)

        if return_type is None or return_type == np.ndarray:
            return _dot_ndarray_coo_type(a.dtype, b.dtype)(a, b.coords, b.data, out_shape)
        b = b.T
        coords, data = _dot_ndarray_coo_type_sparse(a.dtype, b.dtype)(a, b.coords, b.data, out_shape)
        out = COO(coords, data, shape=out_shape, has_duplicates=False, sorted=True, prune=True)
        if return_type == GCXS:
            return out.asformat("gcxs")
        return out

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.dot(a, b)

    raise TypeError("Unsupported types.")


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
    >>> func(np.dtype("i8")) is func(np.dtype("int64"))
    True
    >>> func(np.dtype("i8")) is func(np.dtype("i4"))
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


@numba.jit(nopython=True, nogil=True)
def _csr_csr_count_nnz(out_shape, a_indices, b_indices, a_indptr, b_indptr):  # pragma: no cover
    """
    A function for computing the number of nonzero values in the resulting
    array from multiplying an array with compressed rows with an array
    with compressed rows: (a @ b).nnz.

    Parameters
    ----------
    out_shape : tuple
        The shape of the output array.
    a_indices, a_indptr : np.ndarray
        The indices and index pointer array of ``a``.
    b_data, b_indices, b_indptr : np.ndarray
        The indices and index pointer array of ``b``.
    """
    n_row, n_col = out_shape
    nnz = 0
    mask = np.full(n_col, -1)
    for i in range(n_row):
        row_nnz = 0
        for j in a_indices[a_indptr[i] : a_indptr[i + 1]]:
            for k in b_indices[b_indptr[j] : b_indptr[j + 1]]:
                if mask[k] != i:
                    mask[k] = i
                    row_nnz += 1
        nnz += row_nnz
    return nnz


@numba.jit(nopython=True, nogil=True)
def _csr_ndarray_count_nnz(out_shape, indptr, a_indices, a_indptr, b):  # pragma: no cover
    """
    A function for computing the number of nonzero values in the resulting
    array from multiplying an array with compressed rows with a dense
    numpy array: (a @ b).nnz.

    Parameters
    ----------
    out_shape : tuple
        The shape of the output array.
    indptr : ndarray
        The empty index pointer array for the output.
    a_indices, a_indptr : np.ndarray
        The indices and index pointer array of ``a``.
    b : np.ndarray
        The second input array ``b``.
    """
    nnz = 0
    for i in range(out_shape[0]):
        cur_row = a_indices[a_indptr[i] : a_indptr[i + 1]]
        for j in range(out_shape[1]):
            for k in cur_row:
                if b[k, j] != 0:
                    nnz += 1
                    break
        indptr[i + 1] = nnz
    return nnz


@numba.jit(nopython=True, nogil=True)
def _csc_ndarray_count_nnz(a_shape, b_shape, indptr, a_indices, a_indptr, b):  # pragma: no cover
    """
    A function for computing the number of nonzero values in the resulting
    array from multiplying an array with compressed columns with a dense
    numpy array: (a @ b).nnz.

    Parameters
    ----------
    a_shape, b_shape : tuple
        The shapes of the input arrays.
    indptr : ndarray
        The empty index pointer array for the output.
    a_indices, a_indptr : np.ndarray
        The indices and index pointer array of ``a``.
    b : np.ndarray
        The second input array ``b``.
    """
    nnz = 0
    mask = np.full(a_shape[0], -1)
    for i in range(b_shape[1]):
        col_nnz = 0
        for j in range(b_shape[0]):
            for k in a_indices[a_indptr[j] : a_indptr[j + 1]]:
                if b[j, i] != 0 and mask[k] != i:
                    mask[k] = i
                    col_nnz += 1
        nnz += col_nnz
        indptr[i + 1] = nnz
    return nnz


def _dot_dtype(dt1, dt2):
    return (np.zeros((), dtype=dt1) * np.zeros((), dtype=dt2)).dtype


@_memoize_dtype
def _dot_csr_csr_type(dt1, dt2):
    dtr = _dot_dtype(dt1, dt2)

    @numba.jit(
        nopython=True,
        nogil=True,
        locals={"data_curr": numba.np.numpy_support.from_dtype(dtr)},
    )
    def _dot_csr_csr(out_shape, a_data, b_data, a_indices, b_indices, a_indptr, b_indptr):  # pragma: no cover
        """
        Utility function taking in two ``GCXS`` objects and calculating
        their dot product: a @ b for a and b with compressed rows.

        Parameters
        ----------
        out_shape : tuple
            The shape of the output array.
        a_data, a_indices, a_indptr : np.ndarray
            The data, indices, and index pointer arrays of ``a``.
        b_data, b_indices, b_indptr : np.ndarray
            The data, indices, and index pointer arrays of ``b``.
        """

        # much of this is borrowed from:
        # https://github.com/scipy/scipy/blob/main/scipy/sparse/sparsetools/csr.h

        # calculate nnz before multiplying so we can use static arrays
        nnz = _csr_csr_count_nnz(out_shape, a_indices, b_indices, a_indptr, b_indptr)
        n_row, n_col = out_shape
        indptr = np.empty(n_row + 1, dtype=np.intp)
        indptr[0] = 0
        indices = np.empty(nnz, dtype=np.intp)
        data = np.empty(nnz, dtype=dtr)
        next_ = np.full(n_col, -1)
        sums = np.zeros(n_col, dtype=dtr)
        nnz = 0

        for i in range(n_row):
            head = -2
            length = 0
            next_[:] = -1
            for j, av in zip(
                a_indices[a_indptr[i] : a_indptr[i + 1]],
                a_data[a_indptr[i] : a_indptr[i + 1]],
            ):
                for k, bv in zip(
                    b_indices[b_indptr[j] : b_indptr[j + 1]],
                    b_data[b_indptr[j] : b_indptr[j + 1]],
                ):
                    sums[k] += av * bv
                    if next_[k] == -1:
                        next_[k] = head
                        head = k
                        length += 1

            for _ in range(length):
                if next_[head] != -1:
                    indices[nnz] = head
                    data[nnz] = sums[head]
                    nnz += 1

                temp = head
                head = next_[head]

                next_[temp] = -1
                sums[temp] = 0

            indptr[i + 1] = nnz

        if len(indices) == (n_col * n_row):
            for i in range(len(indices) // n_col):
                j = n_col * i
                k = n_col * (1 + i)
                data[j:k] = data[j:k][::-1]
                indices[j:k] = indices[j:k][::-1]
        return data, indices, indptr

    return _dot_csr_csr


@_memoize_dtype
def _dot_csr_ndarray_type(dt1, dt2):
    dtr = _dot_dtype(dt1, dt2)

    @numba.jit(
        nopython=True,
        nogil=True,
        locals={"data_curr": numba.np.numpy_support.from_dtype(dtr)},
    )
    def _dot_csr_ndarray(out_shape, a_data, a_indices, a_indptr, b):  # pragma: no cover
        """
        Utility function taking in one `GCXS` and one ``ndarray`` and
        calculating their dot product: a @ b for a with compressed rows.
        Returns a dense result.

        Parameters
        ----------
        a_data, a_indices, a_indptr : np.ndarray
            The data, indices, and index pointers of ``a``.
        b : np.ndarray
            The second input array ``b``.
        out_shape : Tuple[int]
            The shape of the output array.
        """
        out = np.empty(out_shape, dtype=dtr)
        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
                val = 0
                for k in range(a_indptr[i], a_indptr[i + 1]):
                    ind = a_indices[k]
                    v = a_data[k]
                    val += v * b[ind, j]
                out[i, j] = val
        return out

    return _dot_csr_ndarray


@_memoize_dtype
def _dot_csr_ndarray_type_sparse(dt1, dt2):
    dtr = _dot_dtype(dt1, dt2)

    @numba.jit(
        nopython=True,
        nogil=True,
        locals={"data_curr": numba.np.numpy_support.from_dtype(dtr)},
    )
    def _dot_csr_ndarray_sparse(out_shape, a_data, a_indices, a_indptr, b):  # pragma: no cover
        """
        Utility function taking in one `GCXS` and one ``ndarray`` and
        calculating their dot product: a @ b for a with compressed rows.
        Returns a sparse result.

        Parameters
        ----------
        a_data, a_indices, a_indptr : np.ndarray
            The data, indices, and index pointers of ``a``.
        b : np.ndarray
            The second input array ``b``.
        out_shape : Tuple[int]
            The shape of the output array.
        """
        indptr = np.empty(out_shape[0] + 1, dtype=np.intp)
        indptr[0] = 0
        nnz = _csr_ndarray_count_nnz(out_shape, indptr, a_indices, a_indptr, b)
        indices = np.empty(nnz, dtype=np.intp)
        data = np.empty(nnz, dtype=dtr)
        current = 0
        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
                val = 0
                nonzero = False
                for k in range(a_indptr[i], a_indptr[i + 1]):
                    ind = a_indices[k]
                    v = a_data[k]
                    val += v * b[ind, j]
                    if b[ind, j] != 0:
                        nonzero = True
                if nonzero:
                    data[current] = val
                    indices[current] = j
                    current += 1
        return data, indices, indptr

    return _dot_csr_ndarray_sparse


@_memoize_dtype
def _dot_csc_ndarray_type_sparse(dt1, dt2):
    dtr = _dot_dtype(dt1, dt2)

    @numba.jit(
        nopython=True,
        nogil=True,
        locals={"data_curr": numba.np.numpy_support.from_dtype(dtr)},
    )
    def _dot_csc_ndarray_sparse(a_shape, b_shape, a_data, a_indices, a_indptr, b):  # pragma: no cover
        """
        Utility function taking in one `GCXS` and one ``ndarray`` and
        calculating their dot product: a @ b for a with compressed columns.
        Returns a sparse result.

        Parameters
        ----------
        a_data, a_indices, a_indptr : np.ndarray
            The data, indices, and index pointers of ``a``.
        b : np.ndarray
            The second input array ``b``.
        a_shape, b_shape : Tuple[int]
            The shapes of the input arrays.
        """
        indptr = np.empty(b_shape[1] + 1, dtype=np.intp)
        nnz = _csc_ndarray_count_nnz(a_shape, b_shape, indptr, a_indices, a_indptr, b)
        indices = np.empty(nnz, dtype=np.intp)
        data = np.empty(nnz, dtype=dtr)
        sums = np.zeros(a_shape[0])
        mask = np.full(a_shape[0], -1)
        nnz = 0
        indptr[0] = 0
        for i in range(b_shape[1]):
            head = -2
            length = 0
            for j in range(b_shape[0]):
                u = b[j, i]
                if u != 0:
                    for k in range(a_indptr[j], a_indptr[j + 1]):
                        ind = a_indices[k]
                        v = a_data[k]
                        sums[ind] += u * v
                        if mask[ind] == -1:
                            mask[ind] = head
                            head = ind
                            length += 1
            for _ in range(length):
                if sums[head] != 0:
                    indices[nnz] = head
                    data[nnz] = sums[head]
                    nnz += 1

                temp = head
                head = mask[head]

                mask[temp] = -1
                sums[temp] = 0
        return data, indices, indptr

    return _dot_csc_ndarray_sparse


@_memoize_dtype
def _dot_csc_ndarray_type(dt1, dt2):
    dtr = _dot_dtype(dt1, dt2)

    @numba.jit(
        nopython=True,
        nogil=True,
        locals={"data_curr": numba.np.numpy_support.from_dtype(dtr)},
    )
    def _dot_csc_ndarray(a_shape, b_shape, a_data, a_indices, a_indptr, b):  # pragma: no cover
        """
        Utility function taking in one `GCXS` and one ``ndarray`` and
        calculating their dot product: a @ b for a with compressed columns.
        Returns a dense result.

        Parameters
        ----------
        a_data, a_indices, a_indptr : np.ndarray
            The data, indices, and index pointers of ``a``.
        b : np.ndarray
            The second input array ``b``.
        a_shape, b_shape : Tuple[int]
            The shapes of the input arrays.
        """
        out = np.zeros((a_shape[0], b_shape[1]), dtype=dtr)
        for j in range(b_shape[1]):
            for i in range(b_shape[0]):
                for k in range(a_indptr[i], a_indptr[i + 1]):
                    out[a_indices[k], j] += a_data[k] * b[i, j]
        return out

    return _dot_csc_ndarray


@_memoize_dtype
def _dot_ndarray_csc_type(dt1, dt2):
    dtr = _dot_dtype(dt1, dt2)

    @numba.jit(
        nopython=True,
        nogil=True,
        locals={"data_curr": numba.np.numpy_support.from_dtype(dtr)},
    )
    def _dot_ndarray_csc(out_shape, b_data, b_indices, b_indptr, a):  # pragma: no cover
        """
        Utility function taking in one `ndarray` and one ``GCXS`` and
        calculating their dot product: a @ b for b with compressed columns.

        Parameters
        ----------
        a : np.ndarray
            The input array ``a``.
        b_data, b_indices, b_indptr : np.ndarray
            The data, indices, and index pointers of ``b``.
        out_shape : Tuple[int]
            The shape of the output array.
        """
        out = np.empty(out_shape, dtype=dtr)
        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
                total = 0
                for k in range(b_indptr[j], b_indptr[j + 1]):
                    total += a[i, b_indices[k]] * b_data[k]
                out[i, j] = total
        return out

    return _dot_ndarray_csc


@_memoize_dtype
def _dot_coo_coo_type(dt1, dt2):
    dtr = _dot_dtype(dt1, dt2)

    @numba.jit(
        nopython=True,
        nogil=True,
        locals={"data_curr": numba.np.numpy_support.from_dtype(dtr)},
    )
    def _dot_coo_coo(out_shape, a_coords, b_coords, a_data, b_data, a_indptr, b_indptr):  # pragma: no cover
        """
        Utility function taking in two ``COO`` objects and calculating
        their dot product: a @ b.

        Parameters
        ----------
        a_shape, b_shape : tuple
            The shapes of the input arrays.
        a_data, a_coords : np.ndarray
            The data and coordinates of ``a``.
        b_data, b_coords : np.ndarray
            The data and coordinates of ``b``.
        """

        # much of this is borrowed from:
        # https://github.com/scipy/scipy/blob/main/scipy/sparse/sparsetools/csr.h

        n_row, n_col = out_shape
        # calculate nnz before multiplying so we can use static arrays
        nnz = _csr_csr_count_nnz(out_shape, a_coords[1], b_coords[1], a_indptr, b_indptr)
        coords = np.empty((2, nnz), dtype=np.intp)
        data = np.empty(nnz, dtype=dtr)
        next_ = np.full(n_col, -1)
        sums = np.zeros(n_col, dtype=dtr)
        nnz = 0

        for i in range(n_row):
            head = -2
            length = 0
            next_[:] = -1
            for j, av in zip(
                a_coords[1, a_indptr[i] : a_indptr[i + 1]],
                a_data[a_indptr[i] : a_indptr[i + 1]],
            ):
                for k, bv in zip(
                    b_coords[1, b_indptr[j] : b_indptr[j + 1]],
                    b_data[b_indptr[j] : b_indptr[j + 1]],
                ):
                    sums[k] += av * bv
                    if next_[k] == -1:
                        next_[k] = head
                        head = k
                        length += 1

            for _ in range(length):
                if next_[head] != -1:
                    coords[0, nnz] = i
                    coords[1, nnz] = head
                    data[nnz] = sums[head]
                    nnz += 1

                temp = head
                head = next_[head]

                next_[temp] = -1
                sums[temp] = 0

        return coords, data

    return _dot_coo_coo


@_memoize_dtype
def _dot_coo_ndarray_type(dt1, dt2):
    dtr = _dot_dtype(dt1, dt2)

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
def _dot_coo_ndarray_type_sparse(dt1, dt2):
    dtr = _dot_dtype(dt1, dt2)

    @numba.jit(
        nopython=True,
        nogil=True,
        locals={"data_curr": numba.np.numpy_support.from_dtype(dtr)},
    )
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

        out_data = []
        out_coords = []

        # coords1.shape = (2, len(data1))
        # coords1[0, :] = rows, sorted
        # coords1[1, :] = columns

        didx1 = 0
        while didx1 < len(data1):
            current_row = coords1[0, didx1]

            cur_didx1 = didx1
            oidx2 = 0
            while oidx2 < out_shape[1]:
                cur_didx1 = didx1
                data_curr = 0
                while cur_didx1 < len(data1) and coords1[0, cur_didx1] == current_row:
                    data_curr += data1[cur_didx1] * array2[oidx2, coords1[1, cur_didx1]]
                    cur_didx1 += 1
                if data_curr != 0:
                    out_data.append(data_curr)
                    out_coords.append((current_row, oidx2))
                oidx2 += 1
            didx1 = cur_didx1

        if len(out_data) == 0:
            return np.empty((2, 0), dtype=np.intp), np.empty((0,), dtype=dtr)

        return np.array(out_coords).T, np.array(out_data)

    return _dot_coo_ndarray


@_memoize_dtype
def _dot_ndarray_coo_type(dt1, dt2):
    dtr = _dot_dtype(dt1, dt2)

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
                oidx2 = coords2[1, didx2]
                out[oidx1, oidx2] += array1[oidx1, coords2[0, didx2]] * data2[didx2]

        return out

    return _dot_ndarray_coo


@_memoize_dtype
def _dot_ndarray_coo_type_sparse(dt1, dt2):
    dtr = _dot_dtype(dt1, dt2)

    @numba.jit(
        nopython=True,
        nogil=True,
        locals={"data_curr": numba.np.numpy_support.from_dtype(dtr)},
    )
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
        out_data = []
        out_coords = []

        # coords2.shape = (2, len(data2))
        # coords2[0, :] = columns, sorted
        # coords2[1, :] = rows

        for oidx1 in range(out_shape[0]):
            data_curr = 0
            current_col = 0
            for didx2 in range(len(data2)):
                if coords2[0, didx2] != current_col:
                    if data_curr != 0:
                        out_data.append(data_curr)
                        out_coords.append([oidx1, current_col])
                        data_curr = 0
                    current_col = coords2[0, didx2]

                data_curr += array1[oidx1, coords2[1, didx2]] * data2[didx2]

            if data_curr != 0:
                out_data.append(data_curr)
                out_coords.append([oidx1, current_col])

        if len(out_data) == 0:
            return np.empty((2, 0), dtype=np.intp), np.empty((0,), dtype=dtr)

        return np.array(out_coords).T, np.array(out_data)

    return _dot_ndarray_coo


# Copied from : https://github.com/numpy/numpy/blob/59fec4619403762a5d785ad83fcbde5a230416fc/numpy/core/einsumfunc.py#L523
# under BSD-3-Clause license : https://github.com/numpy/numpy/blob/v1.24.0/LICENSE.txt
def _parse_einsum_input(operands):
    """
    A copy of the numpy parse_einsum_input that
    does not cast the operands to numpy array.

    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the numpy contraction
    Examples
    --------
    The operand list is simplified to reduce printing:
    >>> np.random.seed(123)
    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> _parse_einsum_input(("...a,...a->...", a, b))
    ('za,xza', 'xz', [a, b]) # may vary
    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('za,xza', 'xz', [a, b]) # may vary
    """

    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = operands[1:]

        # Ensure all characters are valid
        for s in subscripts:
            if s in ".,->":
                continue
            if s not in np.core.einsumfunc.einsum_symbols:
                raise ValueError(f"Character {s} is not a valid symbol.")

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for _ in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = operand_list
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = index(s)
                    except TypeError as e:
                        raise TypeError("For this input type lists must contain either int or Ellipsis") from e
                    subscripts += np.core.einsumfunc.einsum_symbols[s]
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = index(s)
                    except TypeError as e:
                        raise TypeError("For this input type lists must contain either int or Ellipsis") from e
                    subscripts += np.core.einsumfunc.einsum_symbols[s]
    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(np.core.einsumfunc.einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(",")
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = builtins.max(operands[num].ndim, 1)
                    ellipse_count -= len(sub) - 3

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                if ellipse_count == 0:
                    split_subscripts[num] = sub.replace("...", "")
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace("...", rep_inds)

        subscripts = ",".join(split_subscripts)
        out_ellipse = "" if longest == 0 else ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (np.core.einsumfunc.einsum_symbols):
                    raise ValueError(f"Character {s} is not a valid symbol.")
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = "".join(sorted(set(output_subscript) - set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in np.core.einsumfunc.einsum_symbols:
                raise ValueError(f"Character {s} is not a valid symbol.")
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError(f"Output character {char} did not appear in the input")

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(",")) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the number of operands.")

    return (input_subscripts, output_subscript, operands)


def _einsum_single(lhs, rhs, operand):
    """Perform a single term einsum, i.e. any combination of transposes, sums
    and traces of dimensions.

    Parameters
    ----------
    lhs : str
        The indices of the input array.
    rhs : str
        The indices of the output array.
    operand : SparseArray
        The array to perform the einsum on.

    Returns
    -------
    output : SparseArray
    """
    from ._coo import COO

    if lhs == rhs:
        if not rhs:
            # ensure scalar output
            return operand.sum()
        return operand

    if not isinstance(operand, SparseArray):
        # just use numpy for dense input
        return np.einsum(f"{lhs}->{rhs}", operand)

    # else require COO for operations, but check if should convert back
    to_output_format = getattr(operand, "from_coo", lambda x: x)
    operand = asCOO(operand)

    # check if repeated / 'trace' indices mean we are only taking a subset
    where = {}
    for i, ix in enumerate(lhs):
        where.setdefault(ix, []).append(i)

    selector = None
    for locs in where.values():
        loc0, *rlocs = locs
        if rlocs:
            # repeated index
            if len({operand.shape[loc] for loc in locs}) > 1:
                raise ValueError("Repeated indices must have the same dimension.")

            # only select data where all indices match
            subselector = (operand.coords[loc0] == operand.coords[rlocs]).all(axis=0)
            if selector is None:
                selector = subselector
            else:
                selector &= subselector

    # indices that are removed (i.e. not in the output / `perm`)
    # are handled by `has_duplicates=True` below
    perm = [lhs.index(ix) for ix in rhs]
    new_shape = tuple(operand.shape[i] for i in perm)

    # select the new COO data
    if selector is not None:
        new_coords = operand.coords[:, selector][perm]
        new_data = operand.data[selector]
    else:
        new_coords = operand.coords[perm]
        new_data = operand.data

    if not rhs:
        # scalar output - match numpy behaviour by not wrapping as array
        return new_data.sum()

    return to_output_format(COO(new_coords, new_data, shape=new_shape, has_duplicates=True))


def einsum(*operands, **kwargs):
    """
    Perform the equivalent of :obj:`numpy.einsum`.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    operands : sequence of SparseArray
        These are the arrays for the operation.
    dtype : data-type, optional
        If provided, forces the calculation to use the data type specified.
        Default is ``None``.
    **kwargs : dict, optional
        Any additional arguments to pass to the function.

    Returns
    -------
    output : SparseArray
        The calculation based on the Einstein summation convention.
    """

    lhs, rhs, operands = _parse_einsum_input(operands)  # Parse input

    check_zero_fill_value(*operands)

    if "dtype" in kwargs and kwargs["dtype"] is not None:
        operands = [o.astype(kwargs["dtype"]) for o in operands]

    if len(operands) == 1:
        return _einsum_single(lhs, rhs, operands[0])

    # if multiple arrays: align, broadcast multiply and then use single einsum
    # for example:
    #     "aab,cbd->dac"
    # we first perform single term reductions and align:
    #     aab -> ab..
    #     cbd -> .bcd
    # (where dots represent broadcastable size 1 dimensions), then multiply all
    # to form the 'minimal outer product' and do a final single term einsum:
    #     abcd -> dac

    # get ordered union of indices from all terms, indicies that only appear
    # on a single term will be removed in the 'preparation' step below
    terms = lhs.split(",")
    total = {}
    sizes = {}
    for t, term in enumerate(terms):
        shape = operands[t].shape
        for ix, d in zip(term, shape):
            if d != sizes.setdefault(ix, d):
                raise ValueError(f"Inconsistent shape for index '{ix}'.")
            total.setdefault(ix, set()).add(t)
    for ix in rhs:
        total[ix].add(-1)
    aligned_term = "".join(ix for ix, apps in total.items() if len(apps) > 1)

    # NB: if every index appears exactly twice,
    # we could identify and dispatch to tensordot here?

    parrays = []
    for term, array in zip(terms, operands):
        # calc the target indices for this term
        pterm = "".join(ix for ix in aligned_term if ix in term)
        if pterm != term:
            # perform necessary transpose and reductions
            array = _einsum_single(term, pterm, array)
        # calc broadcastable shape
        shape = tuple(array.shape[pterm.index(ix)] if ix in pterm else 1 for ix in aligned_term)
        parrays.append(array.reshape(shape) if array.shape != shape else array)

    aligned_array = reduce(mul, parrays)

    return _einsum_single(aligned_term, rhs, aligned_array)


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
    from ._compressed import GCXS

    if not builtins.all(isinstance(arr, GCXS) for arr in arrays):
        from ._coo import stack as coo_stack

        return coo_stack(arrays, axis)

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
    from ._compressed import GCXS

    if not builtins.all(isinstance(arr, GCXS) for arr in arrays):
        from ._coo import concatenate as coo_concat

        return coo_concat(arrays, axis)

    from ._compressed import concatenate as gcxs_concat

    return gcxs_concat(arrays, axis, compressed_axes)


concat = concatenate


def eye(N, M=None, k=0, dtype=float, format="coo", **kwargs):
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
    from ._coo import COO

    if M is None:
        M = N

    N = int(N)
    M = int(M)
    k = int(k)

    data_length = builtins.min(N, M)

    if k > 0:
        data_length = builtins.max(builtins.min(data_length, M - k), 0)
        n_coords = np.arange(data_length, dtype=np.intp)
        m_coords = n_coords + k
    elif k < 0:
        data_length = builtins.max(builtins.min(data_length, N + k), 0)
        m_coords = np.arange(data_length, dtype=np.intp)
        n_coords = m_coords - k
    else:
        n_coords = m_coords = np.arange(data_length, dtype=np.intp)

    coords = np.stack([n_coords, m_coords])
    data = np.array(1, dtype=dtype)

    return COO(coords, data=data, shape=(N, M), has_duplicates=False, sorted=True).asformat(format, **kwargs)


def full(shape, fill_value, dtype=None, format="coo", order="C", **kwargs):
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
    order : {'C', None}
        Values except these are not currently supported and raise a
        NotImplementedError.

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
    if order not in {"C", None}:
        raise NotImplementedError("Currently, only 'C' and None are supported.")
    data = np.empty(0, dtype=dtype)
    coords = np.empty((len(shape), 0), dtype=np.intp)
    return COO(
        coords,
        data=data,
        shape=shape,
        fill_value=fill_value,
        has_duplicates=False,
        sorted=True,
    ).asformat(format, **kwargs)


def full_like(a, fill_value, dtype=None, shape=None, format=None, **kwargs):
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
    >>> x = np.ones((2, 3), dtype="i8")
    >>> full_like(x, 9.0).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[9, 9, 9],
           [9, 9, 9]])
    """
    if format is None and not isinstance(a, np.ndarray):
        format = type(a).__name__.lower()
    elif format is None:
        format = "coo"

    compressed_axes = kwargs.pop("compressed_axes", None)
    if hasattr(a, "compressed_axes") and compressed_axes is None:
        compressed_axes = a.compressed_axes
    return full(
        a.shape if shape is None else shape,
        fill_value,
        dtype=(a.dtype if dtype is None else dtype),
        format=format,
        **kwargs,
    )


def zeros(shape, dtype=float, format="coo", **kwargs):
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
    return full(shape, fill_value=0, dtype=np.dtype(dtype), format=format, **kwargs)


def zeros_like(a, dtype=None, shape=None, format=None, **kwargs):
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
    >>> x = np.ones((2, 3), dtype="i8")
    >>> zeros_like(x).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0],
           [0, 0, 0]])
    """
    return full_like(a, fill_value=0, dtype=dtype, shape=shape, format=format, **kwargs)


def ones(shape, dtype=float, format="coo", **kwargs):
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
    return full(shape, fill_value=1, dtype=np.dtype(dtype), format=format, **kwargs)


def ones_like(a, dtype=None, shape=None, format=None, **kwargs):
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
    >>> x = np.ones((2, 3), dtype="i8")
    >>> ones_like(x).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 1],
           [1, 1, 1]])
    """
    return full_like(a, fill_value=1, dtype=dtype, shape=shape, format=format, **kwargs)


def empty(shape, dtype=float, format="coo", **kwargs):
    return full(shape, fill_value=0, dtype=np.dtype(dtype), format=format, **kwargs)


empty.__doc__ = zeros.__doc__


def empty_like(a, dtype=None, shape=None, format=None, **kwargs):
    return full_like(a, fill_value=0, dtype=dtype, shape=shape, format=format, **kwargs)


empty_like.__doc__ = zeros_like.__doc__


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
    from ._coo import COO
    from ._sparse_array import SparseArray

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
    a : SparseArray
        The array whose axes should be reordered.
    source : int or List[int]
        Original positions of the axes to move. These must be unique.
    destination : int or List[int]
        Destination positions for each of the original axes. These must also be unique.

    Returns
    -------
    SparseArray
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
        raise ValueError("`source` and `destination` arguments must have the same number of elements")

    order = [n for n in range(a.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    return a.transpose(order)


def pad(array, pad_width, mode="constant", **kwargs):
    """
    Performs the equivalent of :obj:`numpy.pad` for :obj:`SparseArray`. Note that
    this function returns a new array instead of a view.

    Parameters
    ----------
    array : SparseArray
        Sparse array which is to be padded.

    pad_width : {sequence, array_like, int}
        Number of values padded to the edges of each axis. ((before_1, after_1), … (before_N, after_N)) unique pad
        widths for each axis. ((before, after),) yields same before and after pad for each axis. (pad,) or int is a
        shortcut for before = after = pad width for all axes.

    mode : str
        Pads to a constant value which is fill value. Currently only constant mode is implemented

    constant_values : int
        The values to set the padded values for each axis. Default is 0. This must be same as fill value.

    Returns
    -------
    SparseArray
        The padded sparse array.

    Raises
    ------
    NotImplementedError
        If mode != 'constant' or there are unknown arguments.

    ValueError
        If constant_values != self.fill_value

    See Also
    --------
    :obj:`numpy.pad` : NumPy equivalent function

    """
    if not isinstance(array, SparseArray):
        raise NotImplementedError("Input array is not compatible.")

    if mode.lower() != "constant":
        raise NotImplementedError(f"Mode '{mode}' is not yet supported.")

    if not equivalent(kwargs.pop("constant_values", _zero_of_dtype(array.dtype)), array.fill_value):
        raise ValueError("constant_values can only be equal to fill value.")

    if kwargs:
        raise NotImplementedError("Additional Unknown arguments present.")

    from ._coo import COO

    array = array.asformat("coo")

    pad_width = np.broadcast_to(pad_width, (len(array.shape), 2))
    new_coords = array.coords + pad_width[:, 0:1]
    new_shape = tuple([array.shape[i] + pad_width[i, 0] + pad_width[i, 1] for i in range(len(array.shape))])
    new_data = array.data
    return COO(new_coords, new_data, new_shape, fill_value=array.fill_value)


def format_to_string(format):
    if isinstance(format, type):
        if not issubclass(format, SparseArray):
            raise ValueError(f"invalid format: {format}")
        format = format.__name__.lower()

    if isinstance(format, str):
        return format

    raise ValueError(f"invalid format: {format}")


def asarray(obj, /, *, dtype=None, format="coo", backend="pydata", device=None, copy=False):
    """
    Convert the input to a sparse array.
    Parameters
    ----------
    obj : array_like
        Object to be converted to an array.
    dtype : dtype, optional
        Output array data type.
    format : str, optional
        Output array sparse format.
    backend : str, optional
        Backend for the output array.
    device : str, optional
        Device on which to place the created array.
    copy : bool, optional
        Boolean indicating whether or not to copy the input.
    Returns
    -------
    out : Union[SparseArray, numpy.ndarray]
        Sparse or 0-D array containing the data from `obj`.
    Examples
    --------
    >>> x = np.eye(8, dtype="i8")
    >>> sparse.asarray(x, format="COO")
    <COO: shape=(8, 8), dtype=int64, nnz=8, fill_value=0>
    """
    if format not in {"coo", "dok", "gcxs"}:
        raise ValueError(f"{format} format not supported.")

    if backend not in {"pydata", "taco"}:
        raise ValueError(f"{backend} backend not supported.")

    from ._compressed import GCXS
    from ._coo import COO
    from ._dok import DOK

    format_dict = {"coo": COO, "dok": DOK, "gcxs": GCXS}

    if backend == "taco":
        raise ValueError("Taco not yet supported.")

    if isinstance(obj, (COO, DOK, GCXS)):
        return obj.asformat(format)

    if isinstance(obj, spmatrix):
        sparse_obj = format_dict[format].from_scipy_sparse(obj)
        if dtype is None:
            dtype = sparse_obj.dtype
        return sparse_obj.astype(dtype=dtype, copy=copy)

    if np.isscalar(obj) or isinstance(obj, (np.ndarray, Iterable)):
        sparse_obj = format_dict[format].from_numpy(np.asarray(obj))
        if dtype is None:
            dtype = sparse_obj.dtype
        return sparse_obj.astype(dtype=dtype, copy=copy)

    raise ValueError(f"{type(obj)} not supported.")


def _support_numpy(func):
    """
    In case a NumPy array is passed to `sparse` namespace function
    we want to flag it and dispatch to NumPy.
    """

    def wrapper_func(*args, **kwargs):
        x = args[0]
        if isinstance(x, (np.ndarray, np.number)):
            warnings.warn(
                f"Sparse {func.__name__} received dense NumPy array instead "
                "of sparse array. Dispatching to NumPy function.",
                RuntimeWarning,
                stacklevel=2,
            )
            return getattr(np, func.__name__)(*args, **kwargs)

        return func(*args, **kwargs)

    return wrapper_func


def all(x, /, *, axis=None, keepdims=False):
    return x.all(axis=axis, keepdims=keepdims)


def any(x, /, *, axis=None, keepdims=False):
    return x.any(axis=axis, keepdims=keepdims)


def permute_dims(x, /, axes=None):
    return x.transpose(axes=axes)


def max(x, /, *, axis=None, keepdims=False):
    return x.max(axis=axis, keepdims=keepdims)


def mean(x, /, *, axis=None, keepdims=False, dtype=None):
    return x.mean(axis=axis, keepdims=keepdims, dtype=dtype)


def min(x, /, *, axis=None, keepdims=False):
    return x.min(axis=axis, keepdims=keepdims)


def prod(x, /, *, axis=None, dtype=None, keepdims=False):
    return x.prod(axis=axis, keepdims=keepdims, dtype=dtype)


def std(x, /, *, axis=None, correction=0.0, keepdims=False):
    return x.std(axis=axis, ddof=correction, keepdims=keepdims)


def sum(x, /, *, axis=None, dtype=None, keepdims=False):
    return x.sum(axis=axis, keepdims=keepdims, dtype=dtype)


def var(x, /, *, axis=None, correction=0.0, keepdims=False):
    return x.var(axis=axis, ddof=correction, keepdims=keepdims)


def abs(x, /):
    return x.__abs__()


def reshape(x, /, shape, *, copy=None):
    return x.reshape(shape=shape)


def astype(x, dtype, /, *, copy=True):
    return x.astype(dtype, copy=copy)


@_support_numpy
def squeeze(x, /, axis=None):
    return x.squeeze(axis=axis)


@_support_numpy
def broadcast_to(x, /, shape):
    return x.broadcast_to(shape)


def broadcast_arrays(*arrays):
    shape = np.broadcast_shapes(*[a.shape for a in arrays])
    return [a.broadcast_to(shape) for a in arrays]


def equal(x1, x2, /):
    return x1 == x2


@_support_numpy
def round(x, /, decimals=0, out=None):
    return x.round(decimals=decimals, out=out)


@_support_numpy
def isinf(x, /):
    return x.isinf()


@_support_numpy
def isnan(x, /):
    return x.isnan()


def isfinite(x, /):
    return ~isinf(x)


def nonzero(x, /):
    return x.nonzero()
