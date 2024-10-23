import functools

import numpy as np

from ._array import Array
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
    if copy is not None and not copy and not arr.flags["C_CONTIGUOUS"]:
        raise NotImplementedError("Cannot only convert C-contiguous arrays at the moment.")
    if copy:
        arr = arr.copy(order="C")
    arr_flat = np.ascontiguousarray(arr).reshape(-1)
    levels = (Level(LevelFormat.Dense),) * arr.ndim
    dense_format = get_storage_format(
        levels=levels,
        order="C",
        pos_width=64,
        crd_width=64,
        dtype=arr.dtype,
    )
    return from_constituent_arrays(format=dense_format, arrays=(arr_flat,), shape=arr.shape)


def to_numpy(arr: Array) -> np.ndarray:
    storage_format: StorageFormat = arr.format

    if not all(LevelFormat.Dense == level.format for level in storage_format.levels):
        raise TypeError(f"Cannot convert a non-dense array to NumPy. `{storage_format=}`")

    (data,) = arr.get_constituent_arrays()
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
            csx_format = get_storage_format(
                levels=(
                    Level(LevelFormat.Dense),
                    Level(
                        LevelFormat.Compressed,
                        LevelProperties(0)
                        if arr.has_canonical_format
                        else LevelProperties.NonUnique | LevelProperties.NonOrdered,
                    ),
                ),
                order=(0, 1) if arr.format == "csr" else (1, 0),
                pos_width=pos_width,
                crd_width=crd_width,
                dtype=arr.dtype,
            )

            indptr = arr.indptr
            indices = arr.indices
            data = arr.data

            if copy:
                indptr = indptr.copy()
                indices = indices.copy()
                data = data.copy()

            return from_constituent_arrays(format=csx_format, arrays=(indptr, indices, data), shape=arr.shape)
        case "coo":
            if copy is not None and not copy:
                raise RuntimeError(f"`scipy.sparse.{type(arr.__name__)}` cannot be zero-copy converted.")
            coords = np.stack([arr.row, arr.col], axis=1)
            pos = np.array([0, arr.nnz], dtype=np.int64)
            pos_width = pos.dtype.itemsize * 8
            crd_width = coords.dtype.itemsize * 8
            data = arr.data
            if copy:
                data = arr.data.copy()

            level_props = LevelProperties(0)
            if not arr.has_canonical_format:
                level_props |= LevelProperties.NonOrdered

            coo_format = get_storage_format(
                levels=(
                    Level(LevelFormat.Compressed, level_props | LevelProperties.NonUnique),
                    Level(LevelFormat.Singleton, level_props),
                ),
                order=(0, 1),
                pos_width=pos_width,
                crd_width=crd_width,
                dtype=arr.dtype,
            )

            return from_constituent_arrays(format=coo_format, arrays=(pos, coords, data), shape=arr.shape)
        case _:
            raise NotImplementedError(f"No conversion implemented for `scipy.sparse.{type(arr.__name__)}`.")


@_guard_scipy
def to_scipy(arr: Array) -> ScipySparseArray:
    storage_format = arr.format

    match storage_format.levels:
        case (Level(LevelFormat.Dense, _), Level(LevelFormat.Compressed, _)):
            indptr, indices, data = arr.get_constituent_arrays()
            if storage_format.order == (0, 1):
                return sps.csr_array((data, indices, indptr), shape=arr.shape)
            return sps.csc_array((data, indices, indptr), shape=arr.shape)
        case (Level(LevelFormat.Compressed, _), Level(LevelFormat.Singleton, _)):
            _, coords, data = arr.get_constituent_arrays()
            return sps.coo_array((data, (coords[:, 0], coords[:, 1])), shape=arr.shape)
        case _:
            raise RuntimeError(f"No conversion implemented for `{storage_format=}`.")


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
    storage = format._get_ctypes_type().from_constituent_arrays(arrays)
    return Array(storage=storage, shape=shape)
