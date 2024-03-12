import numpy as np

from .._utils import can_store, check_consistent_fill_value, normalize_axis


def concatenate(arrays, axis=0, compressed_axes=None):
    from .compressed import GCXS

    check_consistent_fill_value(arrays)
    arrays = [arr if isinstance(arr, GCXS) else GCXS(arr, compressed_axes=(axis,)) for arr in arrays]
    axis = normalize_axis(axis, arrays[0].ndim)
    dim = sum(x.shape[axis] for x in arrays)
    shape = list(arrays[0].shape)
    shape[axis] = dim
    assert all(x.shape[ax] == arrays[0].shape[ax] for x in arrays for ax in set(range(arrays[0].ndim)) - {axis})
    if compressed_axes is None:
        compressed_axes = (axis,)
    if arrays[0].ndim == 1:
        from .._coo.common import concatenate as coo_concat

        arrays = [arr.tocoo() for arr in arrays]
        return coo_concat(arrays, axis=axis)
    # arrays may have different compressed_axes
    # concatenating becomes easy when compressed_axes are the same
    arrays = [arr.change_compressed_axes((axis,)) for arr in arrays]
    ptr_list = []
    for i, arr in enumerate(arrays):
        if i == 0:
            ptr_list.append(arr.indptr)
            continue
        ptr_list.append(arr.indptr[1:])
    indptr = np.concatenate(ptr_list)
    indices = np.concatenate([arr.indices for arr in arrays])
    data = np.concatenate([arr.data for arr in arrays])
    ptr_len = arrays[0].indptr.shape[0]
    nnz = arrays[0].nnz
    total_nnz = sum(int(arr.nnz) for arr in arrays)
    if not can_store(indptr.dtype, total_nnz):
        indptr = indptr.astype(np.min_scalar_type(total_nnz))
    for i in range(1, len(arrays)):
        indptr[ptr_len:] += nnz
        nnz = arrays[i].nnz
        ptr_len += arrays[i].indptr.shape[0] - 1
    return GCXS(
        (data, indices, indptr),
        shape=tuple(shape),
        compressed_axes=arrays[0].compressed_axes,
        fill_value=arrays[0].fill_value,
    ).change_compressed_axes(compressed_axes)


def stack(arrays, axis=0, compressed_axes=None):
    from .compressed import GCXS

    check_consistent_fill_value(arrays)
    arrays = [arr if isinstance(arr, GCXS) else GCXS(arr, compressed_axes=(axis,)) for arr in arrays]
    axis = normalize_axis(axis, arrays[0].ndim + 1)
    assert all(x.shape[ax] == arrays[0].shape[ax] for x in arrays for ax in set(range(arrays[0].ndim)) - {axis})
    if compressed_axes is None:
        compressed_axes = (axis,)
    if arrays[0].ndim == 1:
        from .._coo.common import stack as coo_stack

        arrays = [arr.tocoo() for arr in arrays]
        return coo_stack(arrays, axis=axis)
    # arrays may have different compressed_axes
    # stacking becomes easy when compressed_axes are the same
    ptr_list = []
    for i in range(len(arrays)):
        shape = list(arrays[i].shape)
        shape.insert(axis, 1)
        arrays[i] = arrays[i].reshape(shape).change_compressed_axes((axis,))
        if i == 0:
            ptr_list.append(arrays[i].indptr)
            continue
        ptr_list.append(arrays[i].indptr[1:])

    shape[axis] = len(arrays)
    indptr = np.concatenate(ptr_list)
    indices = np.concatenate([arr.indices for arr in arrays])
    data = np.concatenate([arr.data for arr in arrays])
    ptr_len = arrays[0].indptr.shape[0]
    nnz = arrays[0].nnz
    total_nnz = sum(int(arr.nnz) for arr in arrays)
    if not can_store(indptr.dtype, total_nnz):
        indptr = indptr.astype(np.min_scalar_type(total_nnz))
    for i in range(1, len(arrays)):
        indptr[ptr_len:] += nnz
        nnz = arrays[i].nnz
        ptr_len += arrays[i].indptr.shape[0] - 1
    return GCXS(
        (data, indices, indptr),
        shape=tuple(shape),
        compressed_axes=arrays[0].compressed_axes,
        fill_value=arrays[0].fill_value,
    ).change_compressed_axes(compressed_axes)
