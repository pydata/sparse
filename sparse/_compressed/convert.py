import operator
from functools import reduce

import numba
from numba.typed import List

import numpy as np

from .._coo.common import linear_loc
from .._utils import check_compressed_axes, get_out_dtype


@numba.jit(nopython=True, nogil=True)
def convert_to_flat(inds, shape, dtype):
    """
    Converts the indices of either the compressed or uncompressed axes
    into a linearized form. Prepares the inputs for compute_flat.
    """
    shape_bins = transform_shape(np.asarray(shape))
    increments = List()
    for i in range(len(inds)):
        increments.append((inds[i] * shape_bins[i]).astype(dtype))

    operations = 1
    for inc in increments[:-1]:
        operations *= inc.shape[0]

    if operations == 0:
        return np.empty(0, dtype=dtype)

    cols = increments[-1].repeat(operations).reshape((-1, operations)).T.flatten()
    if len(increments) == 1:
        return cols

    return compute_flat(increments, cols, operations)


@numba.jit(nopython=True, nogil=True)
def compute_flat(increments, cols, operations):  # pragma: no cover
    """
    Iterates through indices and calculates the linearized
    indices.
    """
    start = 0
    end = increments[-1].shape[0]
    positions = np.zeros(len(increments) - 1, dtype=np.intp)
    pos = len(increments) - 2
    for _ in range(operations):
        to_add = 0
        for j in range(len(increments) - 1):
            to_add += increments[j][positions[j]]

        cols[start:end] += to_add
        start += increments[-1].shape[0]
        end += increments[-1].shape[0]

        for j in range(pos, -1, -1):
            positions[j] += 1
            if positions[j] == increments[j].shape[0]:
                positions[j] = 0
            else:
                break

    return cols


@numba.jit(nopython=True, nogil=True)
def transform_shape(shape):  # pragma: no cover
    """
    turns a shape into the linearized increments that
    it represents. For example, given (5,5,5), it returns
    np.array([25,5,1]).
    """
    shape_bins = np.empty(len(shape), dtype=np.intp)
    shape_bins[-1] = 1
    for i in range(len(shape) - 1):
        shape_bins[i] = np.prod(shape[i + 1 :])
    return shape_bins


@numba.jit(nopython=True, nogil=True)
def uncompress_dimension(indptr):  # pragma: no cover
    """converts an index pointer array into an array of coordinates"""
    uncompressed = np.empty(indptr[-1], dtype=indptr.dtype)
    for i in range(len(indptr) - 1):
        uncompressed[indptr[i] : indptr[i + 1]] = i
    return uncompressed


@numba.jit(nopython=True, nogil=True)
def is_sorted(arr):  # pragma: no cover
    """
    function to check if an indexing array is sorted without repeats. If it is,
    we can use the faster slicing algorithm.
    """
    # numba doesn't recognize the new all(...) format
    for i in range(len(arr) - 1):  # noqa: SIM110
        if arr[i + 1] <= arr[i]:
            return False

    return True


@numba.jit(nopython=True, nogil=True)
def _linearize(
    x_indices,
    shape,
    new_axis_order,
    new_reordered_shape,
    new_compressed_shape,
    new_linear,
    new_coords,
):  # pragma: no cover
    for i, n in enumerate(x_indices):
        current = unravel_index(n, shape)
        current_t = current[new_axis_order]
        new_linear[i] = ravel_multi_index(current_t, new_reordered_shape)
        new_coords[:, i] = unravel_index(new_linear[i], new_compressed_shape)


def _1d_reshape(x, shape, compressed_axes):
    check_compressed_axes(shape, compressed_axes)

    new_size = np.prod(shape)
    end_idx = np.searchsorted(x.indices, new_size, side="left")

    # for resizeing in one dimension
    if len(shape) == 1:
        return (x.data[:end_idx], x.indices[:end_idx], [])

    new_axis_order = list(compressed_axes)
    new_axis_order.extend(np.setdiff1d(np.arange(len(shape)), compressed_axes))
    new_axis_order = np.asarray(new_axis_order)
    new_reordered_shape = np.array(shape)[new_axis_order]
    axisptr = len(compressed_axes)
    row_size = np.prod(new_reordered_shape[:axisptr])
    col_size = np.prod(new_reordered_shape[axisptr:])
    new_compressed_shape = np.array((row_size, col_size))
    x_indices = x.indices[:end_idx]
    new_nnz = x_indices.size
    new_linear = np.empty(new_nnz, dtype=np.intp)
    coords_dtype = get_out_dtype(x.indices, max(max(new_compressed_shape), x.nnz))
    new_coords = np.empty((2, new_nnz), dtype=coords_dtype)

    _linearize(
        x_indices,
        np.array(shape),
        new_axis_order,
        new_reordered_shape,
        new_compressed_shape,
        new_linear,
        new_coords,
    )

    order = np.argsort(new_linear)
    new_coords = new_coords[:, order]
    indptr = np.empty(row_size + 1, dtype=coords_dtype)
    indptr[0] = 0
    np.cumsum(np.bincount(new_coords[0], minlength=row_size), out=indptr[1:])
    indices = new_coords[1]
    data = x.data[:end_idx][order]
    return (data, indices, indptr)


def _resize(x, shape, compressed_axes):
    from .compressed import GCXS

    check_compressed_axes(shape, compressed_axes)

    size = reduce(operator.mul, shape, 1)
    if x.ndim == 1:
        end_idx = np.searchsorted(x.indices, size, side="left")
        indices = x.indices[:end_idx]
        data = x.data[:end_idx]
        out = GCXS((data, indices, []), shape=(size,), fill_value=x.fill_value)
        return _1d_reshape(out, shape, compressed_axes)
    uncompressed = uncompress_dimension(x.indptr)
    coords = np.stack((uncompressed, x.indices))
    linear = linear_loc(coords, x._compressed_shape)
    sorted_axis_order = np.argsort(x._axis_order)
    linear_dtype = get_out_dtype(x.indices, np.prod(shape))
    c_linear = np.empty(x.nnz, dtype=linear_dtype)

    _c_ordering(
        linear,
        c_linear,
        np.asarray(x._reordered_shape),
        np.asarray(sorted_axis_order),
        np.asarray(x.shape),
    )

    order = np.argsort(c_linear, kind="mergesort")
    data = x.data[order]
    indices = c_linear[order]
    end_idx = np.searchsorted(indices, size, side="left")
    indices = indices[:end_idx]
    data = data[:end_idx]
    out = GCXS((data, indices, []), shape=(size,), fill_value=x.fill_value)
    return _1d_reshape(out, shape, compressed_axes)


@numba.jit(nopython=True, nogil=True)
def _c_ordering(linear, c_linear, reordered_shape, sorted_axis_order, shape):  # pragma: no cover
    for i, n in enumerate(linear):
        # c ordering
        current_coords = unravel_index(n, reordered_shape)[sorted_axis_order]
        c_linear[i] = ravel_multi_index(current_coords, shape)


def _transpose(x, shape, axes, compressed_axes, transpose=False):
    """
    An algorithm for reshaping, resizing, changing compressed axes, and transposing.
    """

    check_compressed_axes(shape, compressed_axes)
    uncompressed = uncompress_dimension(x.indptr)
    coords = np.stack((uncompressed, x.indices))
    linear = linear_loc(coords, x._compressed_shape)
    sorted_axis_order = np.argsort(x._axis_order)
    if len(shape) == 1:
        dtype = get_out_dtype(x.indices, shape[0])
        c_linear = np.empty(x.nnz, dtype=dtype)
        _c_ordering(
            linear,
            c_linear,
            np.asarray(x._reordered_shape),
            np.asarray(sorted_axis_order),
            np.asarray(x.shape),
        )
        order = np.argsort(c_linear, kind="mergesort")
        data = x.data[order]
        indices = c_linear[order]
        return (data, indices, [])

    new_axis_order = list(compressed_axes)
    new_axis_order.extend(np.setdiff1d(np.arange(len(shape)), compressed_axes))
    new_linear = np.empty(x.nnz, dtype=np.intp)
    new_reordered_shape = np.array(shape)[new_axis_order]
    axisptr = len(compressed_axes)
    row_size = np.prod(new_reordered_shape[:axisptr])
    col_size = np.prod(new_reordered_shape[axisptr:])
    new_compressed_shape = np.array((row_size, col_size))
    coords_dtype = get_out_dtype(x.indices, max(max(new_compressed_shape), x.nnz))
    new_coords = np.empty((2, x.nnz), dtype=coords_dtype)

    _convert_coords(
        linear,
        np.asarray(x.shape),
        np.asarray(x._reordered_shape),
        sorted_axis_order,
        np.asarray(axes),
        np.asarray(shape),
        np.asarray(new_axis_order),
        new_reordered_shape,
        new_linear,
        new_coords,
        new_compressed_shape,
        transpose,
    )

    order = np.argsort(new_linear, kind="mergesort")
    new_coords = new_coords[:, order]
    if len(shape) == 1:
        indptr = []
        indices = coords[0, :]
    else:
        indptr = np.empty(row_size + 1, dtype=coords_dtype)
        indptr[0] = 0
        np.cumsum(np.bincount(new_coords[0], minlength=row_size), out=indptr[1:])
        indices = new_coords[1]

    data = x.data[order]
    return (data, indices, indptr)


@numba.jit(nopython=True, nogil=True)
def unravel_index(n, shape):  # pragma: no cover
    """
    implements a subset of the functionality of np.unravel_index.
    """
    out = np.zeros(len(shape), dtype=np.intp)
    i = 1
    while i < len(shape) and n > 0:
        cur = np.prod(shape[i:])
        out[i - 1] = n // cur
        n -= out[i - 1] * cur
        i += 1
    out[-1] = n
    return out


@numba.jit(nopython=True, nogil=True)
def ravel_multi_index(arr, shape):  # pragma: no cover
    """
    implements a subset of the functionality of np.ravel_multi_index.
    """
    total = 0
    for i, a in enumerate(arr[:-1], 1):
        total += a * np.prod(shape[i:])
    total += arr[-1]
    return total


@numba.jit(nopython=True, nogil=True)
def _convert_coords(
    linear,
    old_shape,
    reordered_shape,
    sorted_axis_order,
    axes,
    shape,
    new_axis_order,
    new_reordered_shape,
    new_linear,
    new_coords,
    new_compressed_shape,
    transpose,
):  # pragma: no cover
    if transpose:
        for i, n in enumerate(linear):
            # c ordering
            current_coords = unravel_index(n, reordered_shape)[sorted_axis_order]
            # transpose
            current_coords_t = current_coords[axes][new_axis_order]
            new_linear[i] = ravel_multi_index(current_coords_t, new_reordered_shape)
            # reshape
            new_coords[:, i] = unravel_index(new_linear[i], new_compressed_shape)
    else:
        for i, n in enumerate(linear):
            # c ordering
            current_coords = unravel_index(n, reordered_shape)[sorted_axis_order]
            # linearize
            c_current = ravel_multi_index(current_coords, old_shape)
            # compress
            c_compressed = unravel_index(c_current, shape)
            c_compressed = c_compressed[new_axis_order]
            new_linear[i] = ravel_multi_index(c_compressed, new_reordered_shape)
            # reshape
            new_coords[:, i] = unravel_index(new_linear[i], new_compressed_shape)
