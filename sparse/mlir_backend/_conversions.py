import dataclasses
import functools

import numpy as np

from ._array import Array
from ._common import _hold_ref, numpy_to_ranked_memref, ranked_memref_to_numpy
from .levels import Level, LevelFormat, LevelProperties, StorageFormat, get_storage_format

try:
    import scipy.sparse as sps

    ScipySparseArray = sps.sparray | sps.spmatrix
except ImportError:
    sps = None
    ScipySparseArray = None


def _guard_scipy(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if sps is None:
            raise RuntimeError("Could not import `scipy.sparse`. Please install `scipy`.")

        return f(*args, **kwargs)

    return wrapped


def _from_numpy(arr: np.ndarray, copy: bool | None = None) -> Array:
    shape = arr.shape
    arr_flat = np.asarray(arr, order="C").flatten()
    if copy and arr_flat.base is arr:
        arr_flat = arr_flat.copy()
    levels = (Level(LevelFormat.Dense),) * len(shape)
    dense_format = get_storage_format(
        levels=levels,
        order="C",
        pos_width=64,
        crd_width=64,
        dtype=arr.dtype,
        owns_memory=False,
    )
    storage = dense_format._get_ctypes_type()(numpy_to_ranked_memref(arr_flat))
    _hold_ref(storage, arr_flat)
    return Array(storage=storage, shape=shape)


def to_numpy(arr):
    storage = arr._storage
    storage_format: StorageFormat = storage.get_storage_format()

    if not all(LevelFormat.Dense == level.format for level in storage_format.levels):
        raise TypeError(f"Cannot convert a non-dense array to NumPy. `{storage_format=}`")

    data = ranked_memref_to_numpy(arr._storage.values)
    _hold_ref(data, storage)
    arg_order = [0] * storage_format.storage_rank
    for i, o in enumerate(storage_format.order):
        arg_order[o] = i
    arg_order = tuple(arg_order)
    storage_shape = tuple(int(arr.shape[o]) for o in arg_order)
    return data.reshape(storage_shape).transpose(arg_order)


@_guard_scipy
def _from_scipy(arr: ScipySparseArray, copy: bool | None = None) -> Array:
    if not isinstance(arr, ScipySparseArray):
        raise TypeError(f"`arr` is not a `scipy.sparse` array, `{type(arr)=}`.")
    match arr.format:
        case "csr" | "csc":
            pos_width = arr.indptr.dtype.itemsize * 8
            crd_width = arr.indices.dtype.itemsize * 8
            csr_format = get_storage_format(
                levels=(
                    Level(LevelFormat.Dense),
                    Level(
                        LevelFormat.Compressed,
                        LevelProperties(0)
                        if arr.has_canonical_format
                        else LevelProperties.NonUnique | LevelProperties.NonOrdered,
                    ),
                ),
                order=(0, 1) if "csr" in type(arr).__name__ else (1, 0),
                pos_width=pos_width,
                crd_width=crd_width,
                dtype=arr.dtype,
                owns_memory=False,
            )

            indptr_np = arr.indptr
            indices_np = arr.indices
            data_np = arr.data

            if copy:
                indptr_np = indptr_np.copy()
                indices_np = indices_np.copy()
                data_np = data_np.copy()

            indptr = numpy_to_ranked_memref(indptr_np)
            indices = numpy_to_ranked_memref(indices_np)
            data = numpy_to_ranked_memref(data_np)

            storage = csr_format._get_ctypes_type()(indptr, indices, data)
            _hold_ref(storage, indptr_np)
            _hold_ref(storage, indices_np)
            _hold_ref(storage, data_np)
            return Array(storage=storage, shape=arr.shape)
        case "coo":
            if copy is not None and not copy:
                raise RuntimeError(f"`scipy.sparse.{type(arr.__name__)}` cannot be zero-copy converted.")
            coords_np = np.stack([arr.row, arr.col], axis=1)
            pos_np = np.array([0, arr.nnz], dtype=np.int64)
            pos_width = pos_np.dtype.itemsize * 8
            crd_width = coords_np.dtype.itemsize * 8
            data_np = arr.data.copy()

            level_props = LevelProperties(0)
            if not arr.has_canonical_format:
                level_props |= LevelProperties.NonOrdered | LevelProperties.NonUnique

            coo_format = get_storage_format(
                levels=(
                    Level(LevelFormat.Compressed, level_props | LevelProperties.NonUnique),
                    Level(LevelFormat.Singleton, level_props),
                ),
                order=(0, 1),
                pos_width=pos_width,
                crd_width=crd_width,
                dtype=arr.dtype,
                owns_memory=False,
            )

            pos = numpy_to_ranked_memref(pos_np)
            crd = numpy_to_ranked_memref(coords_np)
            data = numpy_to_ranked_memref(data_np)

            storage = coo_format._get_ctypes_type()(pos, crd, data)
            _hold_ref(storage, pos_np)
            _hold_ref(storage, coords_np)
            _hold_ref(storage, data_np)
            return Array(storage=storage, shape=arr.shape)
        case _:
            raise NotImplementedError(f"No conversion implemented for `scipy.sparse.{type(arr.__name__)}`.")


@_guard_scipy
def to_scipy(arr) -> ScipySparseArray:
    storage = arr._storage
    storage_format: StorageFormat = storage.get_storage_format()

    match storage_format.levels:
        case (Level(LevelFormat.Dense, _), Level(LevelFormat.Compressed, _)):
            data = ranked_memref_to_numpy(storage.values)
            indices = ranked_memref_to_numpy(storage.indices_1)
            indptr = ranked_memref_to_numpy(storage.pointers_to_1)
            if storage_format.order == (0, 1):
                sps_arr = sps.csr_array((data, indices, indptr), shape=arr.shape)
            else:
                sps_arr = sps.csc_array((data, indices, indptr), shape=arr.shape)
        case (Level(LevelFormat.Compressed, _), Level(LevelFormat.Singleton, _)):
            data = ranked_memref_to_numpy(storage.values)
            coords = ranked_memref_to_numpy(storage.indices_1)
            sps_arr = sps.coo_array((data, (coords[:, 0], coords[:, 1])), shape=arr.shape)
        case _:
            raise RuntimeError(f"No conversion implemented for `{storage_format=}`.")

    _hold_ref(sps_arr, storage)
    return sps_arr


def asarray(arr, copy: bool | None = None) -> Array:
    if sps is not None and isinstance(arr, ScipySparseArray):
        return _from_scipy(arr, copy=copy)
    if isinstance(arr, np.ndarray):
        return _from_numpy(arr, copy=copy)

    if isinstance(arr, Array):
        if copy:
            arr = arr.copy()
        return arr

    return _from_numpy(np.asarray(arr, copy=copy), copy=None)


def from_constituent_arrays(*, format: StorageFormat, arrays: tuple[np.ndarray, ...], shape: tuple[int, ...]) -> Array:
    storage_format: StorageFormat = dataclasses.replace(format, owns_memory=False)
    storage = storage_format._get_ctypes_type().from_constituent_arrays(arrays)
    return Array(storage=storage, shape=shape)
