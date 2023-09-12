from functools import lru_cache
from typing import Callable

import numpy as np
import scipy.sparse
from numba import njit

from .compressed import _Compressed2d


def op_unary(func, a):
    res = a.copy()
    res.data = func(a.data)
    return res


@lru_cache(maxsize=None)
def _numba_d(func):
    return njit(lambda *x: func(*x))


def binary_op(func, a, b):
    func = _numba_d(func)
    if isinstance(a, _Compressed2d) and isinstance(b, _Compressed2d):
        return op_union_indices(func, a, b)
    else:
        raise NotImplementedError()

# From scipy._util
def _prune_array(array):
    """Return an array equivalent to the input array. If the input
    array is a view of a much larger array, copy its contents to a
    newly allocated array. Otherwise, return the input unchanged.
    """
    if array.base is not None and array.size < array.base.size // 2:
        return array.copy()
    return array



def op_union_indices(
    op: Callable, a: scipy.sparse.csr_matrix, b: scipy.sparse.csr_matrix, *, default_value=0
):
    assert a.shape == b.shape

    if type(a) != type(b):
        b = type(a)(b)
    # a.sort_indices()
    # b.sort_indices()

    # TODO: numpy is weird with bools here
    out_dtype = np.array(op(a.data[0], b.data[0])).dtype
    default_value = out_dtype.type(default_value)
    out_indptr = np.zeros_like(a.indptr)
    out_indices = np.zeros(len(a.indices) + len(b.indices), dtype=np.promote_types(a.indices.dtype, b.indices.dtype))
    out_data = np.zeros(len(out_indices), dtype=out_dtype)

    nnz = op_union_indices_csr_csr(
            op,
            a.indptr,
            a.indices,
            a.data,
            b.indptr,
            b.indices,
            b.data,
            out_indptr,
            out_indices,
            out_data,
            out_dtype=out_dtype,
            default_value=default_value,
        )
    out_data = _prune_array(out_data[:nnz])
    out_indices = _prune_array(out_indices[:nnz])
    return type(a)((out_data, out_indices, out_indptr), shape=a.shape)


@njit
def op_union_indices_csr_csr(
    op: Callable,
    a_indptr: np.ndarray,
    a_indices: np.ndarray,
    a_data: np.ndarray,
    b_indptr: np.ndarray,
    b_indices: np.ndarray,
    b_data: np.ndarray,
    out_indptr: np.ndarray,
    out_indices: np.ndarray,
    out_data: np.ndarray,
    out_dtype,
    default_value,
):
    # out_indptr = np.zeros_like(a_indptr)
    # out_indices = np.zeros(len(a_indices) + len(b_indices), dtype=a_indices.dtype)
    # out_data = np.zeros(len(out_indices), dtype=out_dtype)

    out_idx = 0

    for i in range(len(a_indptr) - 1):

        a_idx = a_indptr[i]
        a_end = a_indptr[i + 1]
        b_idx = b_indptr[i]
        b_end = b_indptr[i + 1]

        while (a_idx < a_end) and (b_idx < b_end):
            a_j = a_indices[a_idx]
            b_j = b_indices[b_idx]
            if a_j < b_j:
                val = op(a_data[a_idx], default_value)
                if val != default_value:
                    out_indices[out_idx] = a_j
                    out_data[out_idx] = val
                    out_idx += 1
                a_idx += 1
            elif b_j < a_j:
                val = op(default_value, b_data[b_idx])
                if val != default_value:
                    out_indices[out_idx] = b_j
                    out_data[out_idx] = val
                    out_idx += 1
                b_idx += 1
            else:
                val = op(a_data[a_idx], b_data[b_idx])
                if val != default_value:
                    out_indices[out_idx] = a_j
                    out_data[out_idx] = val
                    out_idx += 1
                a_idx += 1
                b_idx += 1

        # Catch up the other set
        while a_idx < a_end:
            val = op(a_data[a_idx], default_value)
            if val != default_value:
                out_indices[out_idx] = a_indices[a_idx]
                out_data[out_idx] = val
                out_idx += 1
            a_idx += 1

        while b_idx < b_end:
            val = op(default_value, b_data[b_idx])
            if val != default_value:
                out_indices[out_idx] = b_indices[b_idx]
                out_data[out_idx] = val
                out_idx += 1
            b_idx += 1

        out_indptr[i + 1] = out_idx

    # This may need to change to be "resize" to allow memory reallocation
    # resize is currently not implemented in numba
    out_indices = out_indices[: out_idx]
    out_data = out_data[: out_idx]

    return out_idx