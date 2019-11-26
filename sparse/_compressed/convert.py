import numpy as np
import numba
from numba.typed import List


def convert_to_flat(inds, shape):
    inds = [np.array(ind) for ind in inds]
    if any(ind.ndim > 1 for ind in inds):
        raise IndexError("Only one-dimensional iterable indices supported.")
    cols = np.empty(np.prod([ind.size for ind in inds]), dtype=np.intp)
    shape_bins = transform_shape(shape)
    increments = List()
    for i in range(len(inds)):
        increments.append((inds[i] * shape_bins[i]).astype(np.int32))
    operations = np.prod([ind.shape[0] for ind in increments[:-1]])
    return compute_flat(increments, cols, operations)


@numba.jit(nopython=True, nogil=True)
def compute_flat(increments, cols, operations):  # pragma: no cover
    start = 0
    end = increments[-1].shape[0]
    positions = np.zeros(len(increments) - 1, dtype=np.intp)
    pos = len(increments) - 2
    for i in range(operations):
        if i != 0 and positions[pos] == increments[pos].shape[0]:
            positions[pos] = 0
            pos -= 1
            positions[pos] += 1
            pos += 1
        to_add = np.array(
            [increments[i][positions[i]] for i in range(len(increments) - 1)]
        ).sum()
        cols[start:end] = increments[-1] + to_add
        positions[pos] += 1
        start += increments[-1].shape[0]
        end += increments[-1].shape[0]
    return cols


def transform_shape(shape):
    """
    turns a shape into the linearized increments that
    it represents. For example, given (5,5,5), it returns
    np.array([25,5,1]).
    """
    shape_bins = np.empty(len(shape), dtype=np.intp)
    shape_bins[-1] = 1
    for i in range(len(shape) - 2, -1, -1):
        shape_bins[i] = np.prod(shape[i + 1 :])
    return shape_bins


@numba.jit(nopython=True, nogil=True)
def uncompress_dimension(indptr):  # pragma: no cover
    """converts an index pointer array into an array of coordinates"""
    uncompressed = np.empty(indptr[-1], dtype=np.intp)
    for i in range(len(indptr) - 1):
        uncompressed[indptr[i] : indptr[i + 1]] = i
    return uncompressed
