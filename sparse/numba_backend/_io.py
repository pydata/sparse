import numpy as np

from ._common import _check_device
from ._compressed import CSC, CSR, GCXS
from ._coo.core import COO
from ._sparse_array import SparseArray


def save_npz(filename, matrix, compressed=True):
    """Save a sparse matrix to disk in numpy's `.npz` format.
    Note: This is not binary compatible with scipy's `save_npz()`.
    This binary format is not currently stable. Will save a file
    that can only be opend with this package's `load_npz()`.

    Parameters
    ----------
    filename : string or file
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string or a Path, the
        `.npz` extension will be appended to the file name if it is not
        already there
    matrix : SparseArray
        The matrix to save to disk
    compressed : bool
        Whether to save in compressed or uncompressed mode

    Examples
    --------
    Store sparse matrix to disk, and load it again:

    >>> import os
    >>> import sparse
    >>> import numpy as np
    >>> dense_mat = np.array([[[0.0, 0.0], [0.0, 0.70677779]], [[0.0, 0.0], [0.0, 0.86522495]]])
    >>> mat = sparse.COO(dense_mat)
    >>> mat
    <COO: shape=(2, 2, 2), dtype=float64, nnz=2, fill_value=0.0>
    >>> sparse.save_npz("mat.npz", mat)
    >>> loaded_mat = sparse.load_npz("mat.npz")
    >>> loaded_mat
    <COO: shape=(2, 2, 2), dtype=float64, nnz=2, fill_value=0.0>
    >>> os.remove("mat.npz")

    See Also
    --------
    - [`sparse.load_npz`][]
    - [`scipy.sparse.save_npz`][]
    - [`scipy.sparse.load_npz`][]
    - [`numpy.savez`][]
    - [`numpy.load`][]

    """

    nodes = {
        "data": matrix.data,
        "shape": matrix.shape,
        "fill_value": matrix.fill_value,
    }

    if type(matrix) is COO:
        nodes["coords"] = matrix.coords
    elif type(matrix) is GCXS:
        nodes["indices"] = matrix.indices
        nodes["indptr"] = matrix.indptr
        nodes["compressed_axes"] = matrix.compressed_axes

    if compressed:
        np.savez_compressed(filename, **nodes)
    else:
        np.savez(filename, **nodes)


def load_npz(filename):
    """Load a sparse matrix in numpy's `.npz` format from disk.
    Note: This is not binary compatible with scipy's `save_npz()`
    output. This binary format is not currently stable.
    Will only load files saved by this package.

    Parameters
    ----------
    filename : file-like object, string, or pathlib.Path
        The file to read. File-like objects must support the
        `seek()` and `read()` methods.

    Returns
    -------
    SparseArray
        The sparse matrix at path `filename`.

    Examples
    --------
    See [`sparse.save_npz`][] for usage examples.

    See Also
    --------
    - [`sparse.save_npz`][]
    - [`scipy.sparse.save_npz`][]
    - [`scipy.sparse.load_npz`][]
    - [`numpy.savez`][]
    - [`numpy.load`][]

    """

    with np.load(filename) as fp:
        try:
            coords = fp["coords"]
            data = fp["data"]
            shape = tuple(fp["shape"])
            fill_value = fp["fill_value"][()]
            return COO(
                coords=coords,
                data=data,
                shape=shape,
                sorted=True,
                has_duplicates=False,
                fill_value=fill_value,
            )
        except KeyError:
            pass
        try:
            data = fp["data"]
            indices = fp["indices"]
            indptr = fp["indptr"]
            comp_axes = fp["compressed_axes"]
            shape = tuple(fp["shape"])
            fill_value = fp["fill_value"][()]
            return GCXS(
                (data, indices, indptr),
                shape=shape,
                fill_value=fill_value,
                compressed_axes=comp_axes,
            )
        except KeyError as e:
            raise RuntimeError(f"The file {filename!s} does not contain a valid sparse matrix") from e


@_check_device
def from_binsparse(arr, /, *, device=None, copy: bool | None = None) -> SparseArray:
    desc, arrs = arr.__binsparse__()

    desc = desc["binsparse"]
    version_tuple: tuple[int, ...] = tuple(int(v) for v in desc["version"].split("."))
    if version_tuple != (0, 1):
        raise RuntimeError("Unsupported `__binsparse__` protocol version.")

    format = desc["format"]
    format_err_str = f"Unsupported format: `{format!r}`."

    if isinstance(format, str):
        match format:
            case "COO" | "COOR":
                desc["format"] = {
                    "custom": {
                        "transpose": [0, 1],
                        "level": {
                            "level_desc": "sparse",
                            "rank": 2,
                            "level": {
                                "level_desc": "element",
                            },
                        },
                    }
                }
            case "CSC" | "CSR":
                desc["format"] = {
                    "custom": {
                        "transpose": [0, 1] if format == "CSR" else [0, 1],
                        "level": {
                            "level_desc": "dense",
                            "level": {
                                "level_desc": "sparse",
                                "level": {
                                    "level_desc": "element",
                                },
                            },
                        },
                    },
                }
            case _:
                raise RuntimeError(format_err_str)

    format = desc["format"]["custom"]
    rank = 0
    level = format
    while "level" in level:
        if "rank" not in level:
            level["rank"] = 1
        rank += level["rank"]
        level = level["level"]
    if "transpose" not in format:
        format["transpose"] = list(range(rank))

    match desc:
        case {
            "format": {
                "custom": {
                    "transpose": transpose,
                    "level": {
                        "level_desc": "sparse",
                        "rank": ndim,
                        "level": {
                            "level_desc": "element",
                        },
                    },
                },
            },
            "shape": shape,
            "number_of_stored_values": nnz,
            "data_types": {
                "pointers_to_1": _,
                "indices_1": coords_dtype,
                "values": value_dtype,
            },
            **_kwargs,
        }:
            if transpose != list(range(ndim)):
                raise RuntimeError(format_err_str)

            ptr_arr: np.ndarray = np.from_dlpack(arrs[0])
            start, end = ptr_arr
            if copy is False and not (start == 0 or end == nnz):
                raise RuntimeError(format_err_str)

            coord_arr: np.ndarray = np.from_dlpack(arrs[1])
            value_arr: np.ndarray = np.from_dlpack(arrs[2])

            _check_binsparse_dt(coord_arr, coords_dtype)
            _check_binsparse_dt(value_arr, value_dtype)

            return COO(
                coord_arr[:, start:end],
                value_arr,
                shape=shape,
                has_duplicates=False,
                sorted=True,
                prune=False,
                idx_dtype=coord_arr.dtype,
            )
        case {
            "format": {
                "custom": {
                    "transpose": transpose,
                    "level": {
                        "level_desc": "dense",
                        "rank": 1,
                        "level": {
                            "level_desc": "sparse",
                            "rank": 1,
                            "level": {
                                "level_desc": "element",
                            },
                        },
                    },
                },
            },
            "shape": shape,
            "number_of_stored_values": nnz,
            "data_types": {
                "pointers_to_1": ptr_dtype,
                "indices_1": crd_dtype,
                "values": val_dtype,
            },
            **_kwargs,
        }:
            crd_arr = np.from_dlpack(arrs[0])
            _check_binsparse_dt(crd_arr, crd_dtype)
            ptr_arr = np.from_dlpack(arrs[1])
            _check_binsparse_dt(ptr_arr, ptr_dtype)
            val_arr = np.from_dlpack(arrs[2])
            _check_binsparse_dt(val_arr, val_dtype)

            match transpose:
                case [0, 1]:
                    sparse_type = CSR
                case [1, 0]:
                    sparse_type = CSC
                case _:
                    raise RuntimeError(format_err_str)

            return sparse_type((val_arr, ptr_arr, crd_arr), shape=shape)
        case _:
            raise RuntimeError(format_err_str)


def _convert_binsparse_dtype(dt: str) -> np.dtype:
    if dt.startswith("complex[float") and dt.endswith("]"):
        complex_bits = 2 * int(dt[len("complex[float") : -len("]")])
        dt: str = f"complex{complex_bits}"

    return np.dtype(dt)


def _check_binsparse_dt(arr: np.ndarray, dt: str) -> None:
    invalid_dtype_str = "Invalid dtype: `{dtype!s}`, expected `{expected!s}`."
    dt = _convert_binsparse_dtype(dt)
    if dt != arr.dtype:
        raise BufferError(
            invalid_dtype_str.format(
                dtype=arr.dtype,
                expected=dt,
            )
        )
