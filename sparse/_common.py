import numpy as np
import numba
import scipy.sparse
from functools import wraps
from itertools import chain
from collections.abc import Iterable
from scipy.sparse import spmatrix
from numba import literal_unroll
import warnings

from ._sparse_array import SparseArray
from ._utils import (
    check_compressed_axes,
    normalize_axis,
    check_zero_fill_value,
    equivalent,
    _zero_of_dtype,
)

from ._umath import elemwise
from ._coo.common import (
    clip,
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
    asCOO,
    linear_loc,
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
    elif isinstance(test, spmatrix):
        return nan_check(test.data)
    else:
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
    # Please see license at https://github.com/numpy/numpy/blob/master/LICENSE.txt
    check_zero_fill_value(a, b)

    if scipy.sparse.issparse(a):
        a = GCXS.from_scipy_sparse(a)
    if scipy.sparse.issparse(b):
        b = GCXS.from_scipy_sparse(b)

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
    if nda == 0 or ndb == 0:
        pos = int(nda != 0)
        raise ValueError("Input {} operand does not have enough dimensions".format(pos))
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

    if any(dim == 0 for dim in chain(newshape_a, newshape_b)):
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
        raise TypeError(
            "Cannot perform dot product on types %s, %s" % (type(a), type(b))
        )

    if check_class_nan(a) or check_class_nan(b):
        warnings.warn(
            "Nan will not be propagated in matrix multiplication", RuntimeWarning
        )

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
        for i in range(max(a.shape[0], b.shape[0])):
            a_i = a[0] if a.shape[0] == 1 else a[i]
            b_i = b[0] if b.shape[0] == 1 else b[i]
            res.append(_matmul_recurser(a_i, b_i))
        mask = [isinstance(x, SparseArray) for x in res]
        if all(mask):
            return stack(res)
        else:
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
        raise TypeError(
            "Cannot perform dot product on types %s, %s" % (type(a), type(b))
        )

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
    from ._coo import COO
    from ._compressed import GCXS
    from ._compressed.convert import uncompress_dimension
    from ._sparse_array import SparseArray

    out_shape = (a.shape[0], b.shape[1])
    if all(isinstance(arr, SparseArray) for arr in [a, b]) and any(
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
        elif return_type == COO:
            return out.tocoo()
        return out

    if isinstance(a, GCXS) and isinstance(b, np.ndarray):
        if a.compressed_axes == (0,):  # csr @ ndarray
            if return_type is None or return_type == np.ndarray:
                return _dot_csr_ndarray_type(a.dtype, b.dtype)(
                    out_shape, a.data, a.indices, a.indptr, b
                )
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
            return _dot_csc_ndarray_type(a.dtype, b.dtype)(
                a.shape, b.shape, a.data, a.indices, a.indptr, b
            )
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
                out = _dot_csc_ndarray_type(bt.dtype, at.dtype)(
                    bt.shape, at.shape, bt.data, bt.indices, bt.indptr, at
                )
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
            return _dot_ndarray_csc_type(a.dtype, b.dtype)(
                out_shape, b.data, b.indices, b.indptr, a
            )
        data, indices, indptr = _dot_csr_ndarray_type_sparse(bt.dtype, at.dtype)(
            out_shape[::-1], bt.data, bt.indices, bt.indptr, at
        )
        out = GCXS(
            (data, indices, indptr), shape=out_shape, compressed_axes=(1,), prune=True
        )
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
        elif return_type == GCXS:
            return out.asformat("gcxs")
        return out

    if isinstance(a, COO) and isinstance(b, np.ndarray):
        b = b.view(type=np.ndarray).T

        if return_type is None or return_type == np.ndarray:
            return _dot_coo_ndarray_type(a.dtype, b.dtype)(
                a.coords, a.data, b, out_shape
            )
        coords, data = _dot_coo_ndarray_type_sparse(a.dtype, b.dtype)(
            a.coords, a.data, b, out_shape
        )
        out = COO(coords, data, shape=out_shape, has_duplicates=False, sorted=True)
        if return_type == GCXS:
            return out.asformat("gcxs")
        return out

    if isinstance(a, np.ndarray) and isinstance(b, COO):
        b = b.T
        a = a.view(type=np.ndarray)

        if return_type is None or return_type == np.ndarray:
            return _dot_ndarray_coo_type(a.dtype, b.dtype)(
                a, b.coords, b.data, out_shape
            )
        coords, data = _dot_ndarray_coo_type_sparse(a.dtype, b.dtype)(
            a, b.coords, b.data, out_shape
        )
        out = COO(
            coords, data, shape=out_shape, has_duplicates=False, sorted=True, prune=True
        )
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


@numba.jit(nopython=True, nogil=True)
def _csr_csr_count_nnz(
    out_shape, a_indices, b_indices, a_indptr, b_indptr
):  # pragma: no cover
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
def _csr_ndarray_count_nnz(
    out_shape, indptr, a_indices, a_indptr, b
):  # pragma: no cover
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
def _csc_ndarray_count_nnz(
    a_shape, b_shape, indptr, a_indices, a_indptr, b
):  # pragma: no cover
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
    def _dot_csr_csr(
        out_shape, a_data, b_data, a_indices, b_indices, a_indptr, b_indptr
    ):  # pragma: no cover
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
        # https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h

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
    def _dot_csr_ndarray_sparse(
        out_shape, a_data, a_indices, a_indptr, b
    ):  # pragma: no cover
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
    def _dot_csc_ndarray_sparse(
        a_shape, b_shape, a_data, a_indices, a_indptr, b
    ):  # pragma: no cover
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
            start = nnz
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
    def _dot_csc_ndarray(
        a_shape, b_shape, a_data, a_indices, a_indptr, b
    ):  # pragma: no cover
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
    def _dot_coo_coo(
        out_shape, a_coords, b_coords, a_data, b_data, a_indptr, b_indptr
    ):  # pragma: no cover
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
        # https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h

        n_row, n_col = out_shape
        # calculate nnz before multiplying so we can use static arrays
        nnz = _csr_csr_count_nnz(
            out_shape, a_coords[1], b_coords[1], a_indptr, b_indptr
        )
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

            start = nnz
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
                oidx2 = coords2[0, didx2]
                out[oidx1, oidx2] += array1[oidx1, coords2[1, didx2]] * data2[didx2]

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

    if not all(isinstance(arr, GCXS) for arr in arrays):
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
    from ._compressed import GCXS

    if not all(isinstance(arr, GCXS) for arr in arrays):
        from ._coo import concatenate as coo_concat

        return coo_concat(arrays, axis)
    else:
        from ._compressed import concatenate as gcxs_concat

        return gcxs_concat(arrays, axis, compressed_axes)


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
    ).asformat(format, **kwargs)


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
    >>> x = np.ones((2, 3), dtype='i8')
    >>> full_like(x, 9.0).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[9, 9, 9],
           [9, 9, 9]])
    """
    if format is None and not isinstance(a, np.ndarray):
        format = type(a).__name__.lower()
    elif format is None:
        format = "coo"
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
    return full(shape, 0, np.dtype(dtype)).asformat(format, **kwargs)


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
    >>> x = np.ones((2, 3), dtype='i8')
    >>> zeros_like(x).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0],
           [0, 0, 0]])
    """
    return full_like(a, 0, dtype=dtype, shape=shape, format=format, **kwargs)


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
    return full(shape, 1, np.dtype(dtype)).asformat(format, **kwargs)


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
    >>> x = np.ones((2, 3), dtype='i8')
    >>> ones_like(x).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 1, 1],
           [1, 1, 1]])
    """
    return full_like(a, 1, dtype=dtype, shape=shape, format=format, **kwargs)


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
        raise ValueError(
            "`source` and `destination` arguments must have "
            "the same number of elements"
        )

    order = [n for n in range(a.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    result = a.transpose(order)
    return result


def pad(array, pad_width, mode="constant", **kwargs):
    """
    Performs the equivalent of :obj:`numpy.pad` for :obj:`SparseArray`. Note that
    this function returns a new array instead of a view.

    Parameters
    ----------
    array : SparseArray
        Sparse array which is to be padded.

    pad_width : {sequence, array_like, int}
        Number of values padded to the edges of each axis. ((before_1, after_1), … (before_N, after_N)) unique pad widths for each axis. ((before, after),) yields same before and after pad for each axis. (pad,) or int is a shortcut for before = after = pad width for all axes.

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

    if not equivalent(
        kwargs.pop("constant_values", _zero_of_dtype(array.dtype)), array.fill_value
    ):
        raise ValueError("constant_values can only be equal to fill value.")

    if kwargs:
        raise NotImplementedError("Additional Unknown arguments present.")

    from ._coo import COO

    array = array.asformat("coo")

    pad_width = np.broadcast_to(pad_width, (len(array.shape), 2))
    new_coords = array.coords + pad_width[:, 0:1]
    new_shape = tuple(
        [
            array.shape[i] + pad_width[i, 0] + pad_width[i, 1]
            for i in range(len(array.shape))
        ]
    )
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
