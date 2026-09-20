import numpy as np

from ._compressed.compressed import CSC, CSR, GCXS
from ._coo.core import COO
from ._dok import DOK


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

    if isinstance(matrix, COO):
        nodes = {
            "format": "coo",
            "shape": matrix.shape,
            "data": matrix.data,
            "coords": matrix.coords,
            "fill_value": matrix.fill_value,
        }
    elif isinstance(matrix, (CSR, CSC)):
        fmt = "csr" if isinstance(matrix, CSR) else "csc"
        nodes = {
            "format": fmt,
            "shape": matrix.shape,
            "data": matrix.data,
            "indices": matrix.indices,
            "indptr": matrix.indptr,
            "compressed_axes": np.array(matrix.compressed_axes, dtype=np.intp),
            "fill_value": matrix.fill_value,
        }
    elif isinstance(matrix, GCXS):
        comp_axes = (
            np.array([], dtype=np.intp)
            if matrix.compressed_axes is None
            else np.array(matrix.compressed_axes, dtype=np.intp)
        )
        nodes = {
            "format": "gcxs",
            "shape": matrix.shape,
            "data": matrix.data,
            "indices": matrix.indices,
            "indptr": matrix.indptr,
            "compressed_axes": comp_axes,
            "fill_value": matrix.fill_value,
        }
    elif isinstance(matrix, DOK):
        coo = matrix.to_coo()
        nodes = {
            "format": "dok",
            "shape": coo.shape,
            "data": coo.data,
            "coords": coo.coords,
            "fill_value": coo.fill_value,
        }
    else:
        raise ValueError(f"Cannot save array of type {type(matrix).__name__} to npz")

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
        if "format" in fp:
            fmt = str(fp["format"][()])
            try:
                if fmt == "coo":
                    return COO(
                        coords=fp["coords"],
                        data=fp["data"],
                        shape=tuple(fp["shape"]),
                        sorted=True,
                        has_duplicates=False,
                        fill_value=fp["fill_value"][()],
                    )
                if fmt == "csr":
                    return CSR(
                        (fp["data"], fp["indices"], fp["indptr"]),
                        shape=tuple(fp["shape"]),
                        fill_value=fp["fill_value"][()],
                    )
                if fmt == "csc":
                    return CSC(
                        (fp["data"], fp["indices"], fp["indptr"]),
                        shape=tuple(fp["shape"]),
                        fill_value=fp["fill_value"][()],
                    )
                if fmt == "gcxs":
                    comp_axes = tuple(fp["compressed_axes"]) if fp["compressed_axes"].size > 0 else None
                    return GCXS(
                        (fp["data"], fp["indices"], fp["indptr"]),
                        shape=tuple(fp["shape"]),
                        fill_value=fp["fill_value"][()],
                        compressed_axes=comp_axes,
                    )
                if fmt == "dok":
                    coo = COO(
                        coords=fp["coords"],
                        data=fp["data"],
                        shape=tuple(fp["shape"]),
                        sorted=True,
                        has_duplicates=False,
                        fill_value=fp["fill_value"][()],
                    )
                    return DOK.from_coo(coo)
            except KeyError as e:
                raise RuntimeError(f"The file {filename!s} does not contain a valid sparse matrix") from e
            raise RuntimeError(f"Unknown sparse format {fmt!r} in {filename!s}")

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
            comp_axes = tuple(fp["compressed_axes"]) if fp["compressed_axes"].size > 0 else None
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
