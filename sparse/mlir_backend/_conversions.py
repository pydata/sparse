import functools

import numpy as np

from ._array import Array
from .formats import ConcreteFormat, Coo, Csf, Dense, Level, LevelFormat

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
    dense_format = Dense().with_ndim(arr.ndim).with_dtype(arr.dtype).build()
    return from_constituent_arrays(format=dense_format, arrays=(arr_flat,), shape=arr.shape)


def to_numpy(arr: Array) -> np.ndarray:
    if not Dense.is_this_format(arr.format):
        raise TypeError(f"Cannot convert a non-dense array to NumPy. `{arr.format=}`")

    (data,) = arr.get_constituent_arrays()
    arg_order = [0] * arr.format.storage_rank
    for i, o in enumerate(arr.format.order):
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
            order = (0, 1) if arr.format == "csr" else (1, 0)
            pos_width = arr.indptr.dtype.itemsize * 8
            crd_width = arr.indices.dtype.itemsize * 8
            csx_format = (
                Csf()
                .with_ndim(2, canonical=arr.has_canonical_format)
                .with_dtype(arr.dtype)
                .with_crd_width(crd_width)
                .with_pos_width(pos_width)
                .with_order(order)
                .build()
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
            row, col = arr.row, arr.col
            if row.dtype != col.dtype:
                raise RuntimeError(f"`row` and `col` dtypes must be the same: {row.dtype} != {col.dtype}.")
            pos = np.array([0, arr.nnz], dtype=np.int64)
            pos_width = pos.dtype.itemsize * 8
            crd_width = row.dtype.itemsize * 8
            data = arr.data
            if copy:
                data = data.copy()
                row = row.copy()
                col = col.copy()

            coo_format = (
                Coo()
                .with_ndim(2, canonical=arr.has_canonical_format)
                .with_dtype(arr.dtype)
                .with_pos_width(pos_width)
                .with_crd_width(crd_width)
                .build()
            )

            return from_constituent_arrays(format=coo_format, arrays=(pos, row, col, data), shape=arr.shape)
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
            _, row, col, data = arr.get_constituent_arrays()
            return sps.coo_array((data, (row, col)), shape=arr.shape)
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

    if copy is not None and not copy and not isinstance(arr, np.ndarray):
        raise ValueError("Cannot non-copy convert this object.")

    return _from_numpy(np.asarray(arr), copy=copy)


def from_constituent_arrays(*, format: ConcreteFormat, arrays: tuple[np.ndarray, ...], shape: tuple[int, ...]) -> Array:
    storage = format._get_ctypes_type().from_constituent_arrays(arrays)
    return Array(storage=storage, shape=shape)
