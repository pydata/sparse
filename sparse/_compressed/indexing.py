import numpy as np
import numba
from numbers import Integral
from itertools import zip_longest
from collections.abc import Iterable
from .._slicing import normalize_index
from .convert import convert_to_flat, uncompress_dimension, is_sorted


def getitem(x, key):
    """
    GCXS arrays are stored by transposing and reshaping them into csr matrices.
    For indexing, we first convert the n-dimensional key to its corresponding
    2-dimensional key and then iterate through each of the relevent rows and
    columns.
    """
    from .compressed import GCXS

    if x.ndim == 1:
        coo = x.tocoo()[key]
        return GCXS.from_coo(coo)

    key = list(normalize_index(key, x.shape))

    # zip_longest so things like x[..., None] are picked up.
    if len(key) != 0 and all(
        isinstance(k, slice) and k == slice(0, dim, 1)
        for k, dim in zip_longest(key, x.shape)
    ):
        return x

    # return a single element
    if all(isinstance(k, int) for k in key):
        return get_single_element(x, key)

    shape = []
    compressed_inds = np.zeros(len(x.shape), dtype=np.bool_)
    uncompressed_inds = np.zeros(len(x.shape), dtype=np.bool_)

    # which axes will be compressed in the resulting array
    shape_key = np.zeros(len(x.shape), dtype=np.intp)

    # remove Nones from key, evaluate them at the end
    Nones_removed = [k for k in key if k is not None]
    count = 0
    for i, ind in enumerate(Nones_removed):
        if isinstance(ind, Integral):
            continue
        elif isinstance(ind, slice):
            shape_key[i] = count
            shape.append(len(range(ind.start, ind.stop, ind.step)))
            if i in x.compressed_axes:
                compressed_inds[i] = True
            else:
                uncompressed_inds[i] = True
        elif isinstance(ind, Iterable):
            shape_key[i] = count
            shape.append(len(ind))
            if i in x.compressed_axes:
                compressed_inds[i] = True
            else:
                uncompressed_inds[i] = True
        count += 1

    # reorder the key according to the axis_order of the array
    reordered_key = [Nones_removed[i] for i in x._axis_order]

    # if all slices have a positive step and all
    # iterables are sorted without repeats, we can
    # use the quicker slicing algorithm
    pos_slice = True
    for ind in reordered_key[x._axisptr :]:
        if isinstance(ind, slice):
            if ind.step < 0:
                pos_slice = False
        elif isinstance(ind, Iterable):
            if not is_sorted(ind):
                pos_slice = False

    # convert all ints and slices to iterables before flattening
    for i, ind in enumerate(reordered_key):
        if isinstance(ind, Integral):
            reordered_key[i] = [ind]
        elif isinstance(ind, slice):
            reordered_key[i] = np.arange(ind.start, ind.stop, ind.step)

    shape = np.array(shape)

    # convert all indices of compressed axes to a single array index
    # this tells us which 'rows' of the underlying csr matrix to iterate through
    rows = convert_to_flat(
        reordered_key[: x._axisptr],
        x._reordered_shape[: x._axisptr],
        x.indices.dtype,
    )

    # convert all indices of uncompressed axes to a single array index
    # this tells us which 'columns' of the underlying csr matrix to iterate through
    cols = convert_to_flat(
        reordered_key[x._axisptr :],
        x._reordered_shape[x._axisptr :],
        x.indices.dtype,
    )

    starts = x.indptr[:-1][rows]  # find the start and end of each of the rows
    ends = x.indptr[1:][rows]
    if np.any(compressed_inds):
        compressed_axes = shape_key[compressed_inds]

        if len(compressed_axes) == 1:
            row_size = shape[compressed_axes]
        else:
            row_size = np.prod(shape[compressed_axes])

    # if only indexing through uncompressed axes
    else:
        compressed_axes = (0,)  # defaults to 0
        row_size = 1  # this doesn't matter

    if not np.any(uncompressed_inds):  # only indexing compressed axes
        compressed_axes = (0,)  # defaults to 0
        row_size = starts.size

    indptr = np.empty(row_size + 1, dtype=x.indptr.dtype)
    indptr[0] = 0
    if pos_slice:
        arg = get_slicing_selection(x.data, x.indices, indptr, starts, ends, cols)
    else:
        arg = get_array_selection(x.data, x.indices, indptr, starts, ends, cols)

    data, indices, indptr = arg
    size = np.prod(shape[1:])

    if not np.any(uncompressed_inds):  # only indexing compressed axes
        uncompressed = uncompress_dimension(indptr)
        if len(shape) == 1:
            indices = uncompressed
            indptr = None
        else:
            indices = uncompressed % size
            indptr = np.empty(shape[0] + 1, dtype=x.indptr.dtype)
            indptr[0] = 0
            np.cumsum(
                np.bincount(uncompressed // size, minlength=shape[0]), out=indptr[1:]
            )
    if not np.any(compressed_inds):
        if len(shape) == 1:
            indptr = None
        else:
            uncompressed = indices // size
            indptr = np.empty(shape[0] + 1, dtype=x.indptr.dtype)
            indptr[0] = 0
            np.cumsum(np.bincount(uncompressed, minlength=shape[0]), out=indptr[1:])
            indices = indices % size

    arg = (data, indices, indptr)

    # if there were Nones in the key, we insert them back here
    compressed_axes = np.array(compressed_axes)
    shape = shape.tolist()
    for i in range(len(key)):
        if key[i] is None:
            shape.insert(i, 1)
            compressed_axes[compressed_axes >= i] += 1

    compressed_axes = tuple(compressed_axes)
    shape = tuple(shape)

    if len(shape) == 1:
        compressed_axes = None

    return GCXS(
        arg, shape=shape, compressed_axes=compressed_axes, fill_value=x.fill_value
    )


@numba.jit(nopython=True, nogil=True)
def get_slicing_selection(
    arr_data, arr_indices, indptr, starts, ends, col
):  # pragma: no cover
    """
    When the requested elements come in a strictly ascending order, as is the
    case with acsending slices, we can iteratively reduce the search space,
    leading to better performance. We loop through the starts and ends, each time
    evaluating whether to use a linear filtering procedure or a binary-search-based
    method.
    """
    indices = []
    ind_list = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        inds = []
        current_row = arr_indices[start:end]
        if current_row.size < col.size:  # linear filtering
            count = 0
            col_count = 0
            nnz = 0
            while col_count < col.size and count < current_row.size:
                if current_row[-1] < col[col_count] or current_row[count] > col[-1]:
                    break
                if current_row[count] == col[col_count]:
                    nnz += 1
                    ind_list.append(count + start)
                    indices.append(col_count)
                    count += 1
                    col_count += 1
                elif current_row[count] < col[col_count]:
                    count += 1
                else:
                    col_count += 1
            indptr[i + 1] = indptr[i] + nnz
        else:  # binary searches
            prev = 0
            size = 0
            col_count = 0
            while col_count < len(col):
                while (
                    col_count < len(col)
                    and size < len(current_row)
                    and col[col_count] < current_row[size]
                ):  # skip needless searches
                    col_count += 1
                if col_count >= len(col):  # check again because of previous loop
                    break
                if current_row[-1] < col[col_count] or current_row[size] > col[-1]:
                    break
                s = np.searchsorted(current_row[size:], col[col_count])
                size += s
                s += prev
                if not (s >= current_row.size or current_row[s] != col[col_count]):
                    s += start
                    inds.append(s)
                    indices.append(col_count)
                    size += 1
                prev = size
                col_count += 1
            ind_list.extend(inds)
            indptr[i + 1] = indptr[i] + len(inds)
    ind_list = np.array(ind_list, dtype=np.int64)
    indices = np.array(indices, dtype=indptr.dtype)
    data = arr_data[ind_list]
    return (data, indices, indptr)


@numba.jit(nopython=True, nogil=True)
def get_array_selection(
    arr_data, arr_indices, indptr, starts, ends, col
):  # pragma: no cover
    """
    This is a very general algorithm to be used when more optimized methods don't apply.
    It performs a binary search for each of the requested elements.
    Consequently it roughly scales by O(n log avg(nnz)).
    """
    indices = []
    ind_list = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        inds = []
        current_row = arr_indices[start:end]
        if len(current_row) == 0:
            indptr[i + 1] = indptr[i]
            continue
        for c in range(len(col)):
            s = np.searchsorted(current_row, col[c])
            if not (s >= current_row.size or current_row[s] != col[c]):
                s += start
                inds.append(s)
                indices.append(c)
        ind_list.extend(inds)
        indptr[i + 1] = indptr[i] + len(inds)
    ind_list = np.array(ind_list, dtype=np.int64)
    indices = np.array(indices, dtype=indptr.dtype)
    data = arr_data[ind_list]
    return (data, indices, indptr)


def get_single_element(x, key):
    """
    A convience function for indexing when returning
    a single element.
    """
    key = np.array(key)[x._axis_order]  # reordering the input
    ind = np.ravel_multi_index(key, x._reordered_shape)
    row, col = np.unravel_index(ind, x._compressed_shape)
    current_row = x.indices[x.indptr[row] : x.indptr[row + 1]]
    item = np.searchsorted(current_row, col)
    if not (item >= current_row.size or current_row[item] != col):
        item += x.indptr[row]
        return x.data[item]
    return x.fill_value
