import numpy as np

from sparse.coo.core import COO


def save_npz(filename, matrix, compressed=True):
    """ Save a sparse matrix to disk in numpy's ``.npz`` format

    Examples
    --------
    Store sparse matrix to disk, and load it again:

    >>> import sparse
    >>> import numpy as np
    >>> dense_mat = np.array([[[0., 0.], [0., 0.70677779]], [[0., 0.], [0., 0.86522495]]])
    >>> mat = sparse.COO(dense_mat)
    >>> mat
    <COO: shape=(2, 2, 2), dtype=float64, nnz=2>
    >>> sparse.io.save_npz('mat.npz', mat)
    >>> loaded_mat = sparse.io.load_npz('mat.npz')
    >>> loaded_mat
    <COO: shape=(2, 2, 2), dtype=float64, nnz=2>
    >>> loaded_mat.todense()
    array([[[0.        , 0.        ],
            [0.        , 0.70677779]],
    <BLANKLINE>
           [[0.        , 0.        ],
            [0.        , 0.86522495]]])

    :param filename: string or file
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string or a Path, the
        ``.npz`` extension will be appended to the file name if it is not
        already there
    :param matrix: coo matrix
        The matrix to save to disk
    :param compressed: bool
        Whether to save in compressed or uncompressed mode

    """

    nodes = {'data':   matrix.data,
             'coords': matrix.coords}

    if compressed:
        np.savez_compressed(filename, **nodes)
    else:
        np.savez(filename, **nodes)


def load_npz(filename):
    """ Load a sparse matrix in numpy's ``.npz`` format from disk

        Examples
        --------
        Store sparse matrix to disk, and load it again:

        >>> import sparse
        >>> import numpy as np
        >>> dense_mat = np.array([[[0., 0.], [0., 0.70677779]], [[0., 0.], [0., 0.86522495]]])
        >>> mat = sparse.COO(dense_mat)
        >>> mat
        <COO: shape=(2, 2, 2), dtype=float64, nnz=2>
        >>> sparse.io.save_npz('mat.npz', mat)
        >>> loaded_mat = sparse.io.load_npz('mat.npz')
        >>> loaded_mat
        <COO: shape=(2, 2, 2), dtype=float64, nnz=2>
        >>> loaded_mat.todense()
        array([[[0.        , 0.        ],
                [0.        , 0.70677779]],
        <BLANKLINE>
               [[0.        , 0.        ],
                [0.        , 0.86522495]]])

        :param filename: file-like object, string, or pathlib.Path
            The file to read. File-like objects must support the
            ``seek()`` and ``read()`` methods.
        :param matrix: coo matrix
            The matrix to save to disk
        :param compressed: bool
            Whether to save in compressed or uncompressed mode
        :return: coo matrix
            The sparse matrix at path ``filename``
        """
    with np.load(filename) as fp:
        try:
            coords = fp['coords']
            data = fp['data']
        except KeyError:
            raise RuntimeError('The file {} does not contain a valid sparse matrix'.format(filename))

    return COO(coords=coords, data=data)
