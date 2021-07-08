from math import ceil
from numbers import Integral
from collections.abc import Iterable

import numpy as np
import scipy.sparse
from numpy.lib.mixins import NDArrayOperatorsMixin

from ._slicing import normalize_index
from ._utils import equivalent
from ._sparse_array import SparseArray


class DOK(SparseArray, NDArrayOperatorsMixin):
    """
    A class for building sparse multidimensional arrays.

    Parameters
    ----------
    shape : tuple[int] (DOK.ndim,)
        The shape of the array.
    data : dict, optional
        The key-value pairs for the data in this array.
    dtype : np.dtype, optional
        The data type of this array. If left empty, it is inferred from
        the first element.
    fill_value : scalar, optional
        The fill value of this array.

    Attributes
    ----------
    dtype : numpy.dtype
        The datatype of this array. Can be :code:`None` if no elements
        have been set yet.
    shape : tuple[int]
        The shape of this array.
    data : dict
        The keys of this dictionary contain all the indices and the values
        contain the nonzero entries.

    See Also
    --------
    COO : A read-only sparse array.

    Examples
    --------
    You can create :obj:`DOK` objects from Numpy arrays.

    >>> x = np.eye(5, dtype=np.uint8)
    >>> x[2, 3] = 5
    >>> s = DOK.from_numpy(x)
    >>> s
    <DOK: shape=(5, 5), dtype=uint8, nnz=6, fill_value=0>

    You can also create them from just shapes, and use slicing assignment.

    >>> s2 = DOK((5, 5), dtype=np.int64)
    >>> s2[1:3, 1:3] = [[4, 5], [6, 7]]
    >>> s2
    <DOK: shape=(5, 5), dtype=int64, nnz=4, fill_value=0>

    You can convert :obj:`DOK` arrays to :obj:`COO` arrays, or :obj:`numpy.ndarray`
    objects.

    >>> from sparse import COO
    >>> s3 = COO(s2)
    >>> s3
    <COO: shape=(5, 5), dtype=int64, nnz=4, fill_value=0>
    >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0, 0, 0],
           [0, 4, 5, 0, 0],
           [0, 6, 7, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])

    >>> s4 = COO.from_numpy(np.eye(4, dtype=np.uint8))
    >>> s4
    <COO: shape=(4, 4), dtype=uint8, nnz=4, fill_value=0>
    >>> s5 = DOK.from_coo(s4)
    >>> s5
    <DOK: shape=(4, 4), dtype=uint8, nnz=4, fill_value=0>

    You can also create :obj:`DOK` arrays from a shape and a dict of
    values. Zeros are automatically ignored.

    >>> values = {
    ...     (1, 2, 3): 4,
    ...     (3, 2, 1): 0,
    ... }
    >>> s6 = DOK((5, 5, 5), values)
    >>> s6
    <DOK: shape=(5, 5, 5), dtype=int64, nnz=1, fill_value=0.0>
    """

    def __init__(self, shape, data=None, dtype=None, fill_value=None):
        from ._coo import COO

        self.data = dict()

        if isinstance(shape, COO):
            ar = DOK.from_coo(shape)
            self._make_shallow_copy_of(ar)
            return

        if isinstance(shape, np.ndarray):
            ar = DOK.from_numpy(shape)
            self._make_shallow_copy_of(ar)
            return

        if isinstance(shape, scipy.sparse.spmatrix):
            ar = DOK.from_scipy_sparse(shape)
            self._make_shallow_copy_of(ar)
            return

        self.dtype = np.dtype(dtype)

        if not data:
            data = dict()

        super().__init__(shape, fill_value=fill_value)

        if isinstance(data, dict):
            if not dtype:
                if not len(data):
                    self.dtype = np.dtype("float64")
                else:
                    self.dtype = np.result_type(
                        *map(lambda x: np.asarray(x).dtype, data.values())
                    )

            for c, d in data.items():
                self[c] = d
        else:
            raise ValueError("data must be a dict.")

    @classmethod
    def from_scipy_sparse(cls, x):
        """
        Create a :obj:`DOK` array from a :obj:`scipy.sparse.spmatrix`.

        Parameters
        ----------
        x : scipy.sparse.spmatrix
            The matrix to convert.

        Returns
        -------
        DOK
            The equivalent :obj:`DOK` array.

        Examples
        --------
        >>> x = scipy.sparse.rand(6, 3, density=0.2)
        >>> s = DOK.from_scipy_sparse(x)
        >>> np.array_equal(x.todense(), s.todense())
        True
        """
        from sparse import COO

        return COO.from_scipy_sparse(x).asformat(cls)

    @classmethod
    def from_coo(cls, x):
        """
        Get a :obj:`DOK` array from a :obj:`COO` array.

        Parameters
        ----------
        x : COO
            The array to convert.

        Returns
        -------
        DOK
            The equivalent :obj:`DOK` array.

        Examples
        --------
        >>> from sparse import COO
        >>> s = COO.from_numpy(np.eye(4))
        >>> s2 = DOK.from_coo(s)
        >>> s2
        <DOK: shape=(4, 4), dtype=float64, nnz=4, fill_value=0.0>
        """
        ar = cls(x.shape, dtype=x.dtype, fill_value=x.fill_value)

        for c, d in zip(x.coords.T, x.data):
            ar.data[tuple(c)] = d

        return ar

    def to_coo(self):
        """
        Convert this :obj:`DOK` array to a :obj:`COO` array.

        Returns
        -------
        COO
            The equivalent :obj:`COO` array.

        Examples
        --------
        >>> s = DOK((5, 5))
        >>> s[1:3, 1:3] = [[4, 5], [6, 7]]
        >>> s
        <DOK: shape=(5, 5), dtype=float64, nnz=4, fill_value=0.0>
        >>> s2 = s.to_coo()
        >>> s2
        <COO: shape=(5, 5), dtype=float64, nnz=4, fill_value=0.0>
        """
        from ._coo import COO

        return COO(self)

    @classmethod
    def from_numpy(cls, x):
        """
        Get a :obj:`DOK` array from a Numpy array.

        Parameters
        ----------
        x : np.ndarray
            The array to convert.

        Returns
        -------
        DOK
            The equivalent :obj:`DOK` array.

        Examples
        --------
        >>> s = DOK.from_numpy(np.eye(4))
        >>> s
        <DOK: shape=(4, 4), dtype=float64, nnz=4, fill_value=0.0>
        """
        ar = cls(x.shape, dtype=x.dtype)

        coords = np.nonzero(x)
        data = x[coords]

        for c in zip(data, *coords):
            d, c = c[0], c[1:]
            ar.data[c] = d

        return ar

    @property
    def nnz(self):
        """
        The number of nonzero elements in this array.

        Returns
        -------
        int
            The number of nonzero elements.

        See Also
        --------
        COO.nnz : Equivalent :obj:`COO` array property.
        numpy.count_nonzero : A similar Numpy function.
        scipy.sparse.dok_matrix.nnz : The Scipy equivalent property.

        Examples
        --------
        >>> values = {
        ...     (1, 2, 3): 4,
        ...     (3, 2, 1): 0,
        ... }
        >>> s = DOK((5, 5, 5), values)
        >>> s.nnz
        1
        """
        return len(self.data)

    @property
    def format(self):
        """
        The storage format of this array.

        Returns
        -------
        str
            The storage format of this array.
        See Also
        -------
        COO.format : Equivalent :obj:`COO` array property.
        GCXS.format : Equivalent :obj:`GCXS` array property.
        scipy.sparse.dok_matrix.format : The Scipy equivalent property.
        Examples
        -------
        >>> import sparse
        >>> s = sparse.random((5,5), density=0.2, format='dok')
        >>> s.format
        'dok'
        """
        return "dok"

    @property
    def nbytes(self):
        """
        The number of bytes taken up by this object. Note that for small arrays,
        this may undercount the number of bytes due to the large constant overhead.

        Returns
        -------
        int
            The approximate bytes of memory taken by this object.

        See Also
        --------
        numpy.ndarray.nbytes : The equivalent Numpy property.

        Examples
        --------
        >>> import sparse
        >>> x = sparse.random((100,100),density=.1,format='dok')
        >>> x.nbytes
        8000
        """
        return self.nnz * self.dtype.itemsize

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        if all(isinstance(k, Iterable) for k in key):
            if len(key) != self.ndim:
                raise NotImplementedError(
                    f"Index sequences for all {self.ndim} array dimensions needed!"
                )
            if not all(len(key[0]) == len(k) for k in key):
                raise IndexError("Unequal length of index sequences!")
            return self._fancy_getitem(key)

        key = normalize_index(key, self.shape)

        ret = self.asformat("coo")[key]
        if isinstance(ret, SparseArray):
            ret = ret.asformat("dok")

        return ret

    def _fancy_getitem(self, key):
        """Subset of fancy indexing, when all dimensions are accessed"""
        new_data = {}
        for i, k in enumerate(zip(*key)):
            if k in self.data:
                new_data[i] = self.data[k]
        return DOK(
            shape=(len(key[0])),
            data=new_data,
            dtype=self.dtype,
            fill_value=self.fill_value,
        )

    def __setitem__(self, key, value):
        value = np.asarray(value, dtype=self.dtype)

        # 1D fancy indexing
        if (
            self.ndim == 1
            and isinstance(key, Iterable)
            and all(isinstance(i, (int, np.integer)) for i in key)
        ):
            key = (key,)

        if isinstance(key, tuple) and all(isinstance(k, Iterable) for k in key):
            if len(key) != self.ndim:
                raise NotImplementedError(
                    f"Index sequences for all {self.ndim} array dimensions needed!"
                )
            if not all(len(key[0]) == len(k) for k in key):
                raise IndexError("Unequal length of index sequences!")
            self._fancy_setitem(key, value)
            return

        key = normalize_index(key, self.shape)

        key_list = [int(k) if isinstance(k, Integral) else k for k in key]

        self._setitem(key_list, value)

    def _fancy_setitem(self, idxs, values):
        idxs = tuple(np.asanyarray(idxs) for idxs in idxs)
        if not all(np.issubdtype(k.dtype, np.integer) for k in idxs):
            raise IndexError("Indices must be sequences of integer types!")
        if idxs[0].ndim != 1:
            raise IndexError("Indices are not 1d sequences!")
        if values.ndim == 0:
            values = np.full(idxs[0].size, values, self.dtype)
        elif values.ndim > 1:
            raise ValueError(f"Dimension of values ({values.ndim}) must be 0 or 1!")
        if not idxs[0].shape == values.shape:
            raise ValueError(
                f"Shape mismatch of indices ({idxs[0].shape}) and values ({values.shape})!"
            )
        fill_value = self.fill_value
        data = self.data
        for idx, value in zip(zip(*idxs), values):
            if not value == fill_value:
                data[idx] = value
            elif idx in data:
                del data[idx]

    def _setitem(self, key_list, value):
        value_missing_dims = (
            len([ind for ind in key_list if isinstance(ind, slice)]) - value.ndim
        )

        if value_missing_dims < 0:
            raise ValueError("setting an array element with a sequence.")

        for i, ind in enumerate(key_list):
            if isinstance(ind, slice):
                step = ind.step if ind.step is not None else 1
                if step > 0:
                    start = ind.start if ind.start is not None else 0
                    start = max(start, 0)
                    stop = ind.stop if ind.stop is not None else self.shape[i]
                    stop = min(stop, self.shape[i])
                    if start > stop:
                        start = stop
                else:
                    start = ind.start or self.shape[i] - 1
                    stop = ind.stop if ind.stop is not None else -1
                    start = min(start, self.shape[i] - 1)
                    stop = max(stop, -1)
                    if start < stop:
                        start = stop

                key_list_temp = key_list[:]
                for v_idx, ki in enumerate(range(start, stop, step)):
                    key_list_temp[i] = ki
                    vi = (
                        value
                        if value_missing_dims > 0
                        else (value[0] if value.shape[0] == 1 else value[v_idx])
                    )
                    self._setitem(key_list_temp, vi)

                return
            elif not isinstance(ind, Integral):
                raise IndexError(
                    "All indices must be slices or integers when setting an item."
                )

        key = tuple(key_list)
        if not equivalent(value, self.fill_value):
            self.data[key] = value[()]
        elif key in self.data:
            del self.data[key]

    def __str__(self):
        return "<DOK: shape={!s}, dtype={!s}, nnz={:d}, fill_value={!s}>".format(
            self.shape, self.dtype, self.nnz, self.fill_value
        )

    __repr__ = __str__

    def todense(self):
        """
        Convert this :obj:`DOK` array into a Numpy array.

        Returns
        -------
        numpy.ndarray
            The equivalent dense array.

        See Also
        --------
        COO.todense : Equivalent :obj:`COO` array method.
        scipy.sparse.dok_matrix.todense : Equivalent Scipy method.

        Examples
        --------
        >>> s = DOK((5, 5))
        >>> s[1:3, 1:3] = [[4, 5], [6, 7]]
        >>> s.todense()  # doctest: +SKIP
        array([[0., 0., 0., 0., 0.],
               [0., 4., 5., 0., 0.],
               [0., 6., 7., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])
        """
        result = np.full(self.shape, self.fill_value, self.dtype)

        for c, d in self.data.items():
            result[c] = d

        return result

    def asformat(self, format, **kwargs):
        """
        Convert this sparse array to a given format.

        Parameters
        ----------
        format : str
            A format string.

        Returns
        -------
        out : SparseArray
            The converted array.

        Raises
        ------
        NotImplementedError
            If the format isn't supported.
        """
        from sparse._utils import convert_format

        format = convert_format(format)

        if format == "dok":
            return self

        if format == "coo":
            from ._coo import COO

            if len(kwargs) != 0:
                raise ValueError(f"Extra kwargs found: {kwargs}")
            return COO.from_iter(
                self.data,
                shape=self.shape,
                fill_value=self.fill_value,
                dtype=self.dtype,
            )

        return self.asformat("coo").asformat(format, **kwargs)


def to_slice(k):
    """Convert integer indices to one-element slices for consistency"""
    if isinstance(k, Integral):
        return slice(k, k + 1, 1)
    return k
