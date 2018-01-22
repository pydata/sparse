import six

import numpy as np

# Zip with Python 2/3 compat
# Consumes less memory than Py2 zip
from six.moves import zip, range

from numbers import Integral
from collections import Iterable

from .slicing import normalize_index
from .utils import _zero_of_dtype

try:  # Windows compatibility
    int = long
except NameError:
    pass


class DOK(object):
    """
    A class for building sparse multidimensional arrays.

    Parameters
    ----------
    shape : tuple[int]
        The shape of the array
    data : dict, optional
        The key-value pairs for the data in this array.
    dtype : np.dtype, optional
        The data type of this array. If left empty, it is inferred from
        the first element.

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
    <DOK: shape=(5, 5), dtype=uint8, nnz=6>

    You can also create them from just shapes, and use slicing assignment.

    >>> s2 = DOK((5, 5), dtype=np.int64)
    >>> s2[1:3, 1:3] = [[4, 5], [6, 7]]
    >>> s2
    <DOK: shape=(5, 5), dtype=int64, nnz=4>

    You can convert :obj:`DOK` arrays to :obj:`COO` arrays, or :obj:`numpy.ndarray`
    objects.

    >>> from sparse import COO
    >>> s3 = COO(s2)
    >>> s3
    <COO: shape=(5, 5), dtype=int64, nnz=4, sorted=False, duplicates=False>
    >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0, 0, 0],
           [0, 4, 5, 0, 0],
           [0, 6, 7, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])

    >>> s4 = COO.from_numpy(np.eye(4, dtype=np.uint8))
    >>> s4
    <COO: shape=(4, 4), dtype=uint8, nnz=4, sorted=True, duplicates=False>
    >>> s5 = DOK.from_coo(s4)
    >>> s5
    <DOK: shape=(4, 4), dtype=uint8, nnz=4>

    You can also create :obj:`DOK` arrays from a shape and a dict of
    values. Zeros are automatically ignored.

    >>> values = {
    ...     (1, 2, 3): 4,
    ...     (3, 2, 1): 0,
    ... }
    >>> s6 = DOK((5, 5, 5), values)
    >>> s6
    <DOK: shape=(5, 5, 5), dtype=int64, nnz=1>
    """

    def __init__(self, shape, data=None, dtype=None):
        from .coo import COO
        self.data = {}

        if isinstance(shape, COO):
            ar = DOK.from_coo(shape)
            self.shape = ar.shape
            self.dtype = ar.dtype
            self.data = ar.data
            return

        if isinstance(shape, np.ndarray):
            ar = DOK.from_numpy(shape)
            self.shape = ar.shape
            self.dtype = ar.dtype
            self.data = ar.data
            return

        self.dtype = np.dtype(dtype)
        if isinstance(shape, Integral):
            self.shape = (int(shape),)
        elif isinstance(shape, Iterable):
            if not all(isinstance(l, Integral) or int(l) < 0 for l in shape):
                raise ValueError('shape must be an iterable of non-negative integers.')

            self.shape = tuple(shape)

        if not data:
            data = {}

        if isinstance(data, dict):
            if not dtype:
                if not len(data):
                    self.dtype = np.dtype('float64')
                else:
                    self.dtype = np.result_type(*map(lambda x: np.asarray(x).dtype, six.itervalues(data)))

            for c, d in six.iteritems(data):
                self[c] = d
        else:
            raise ValueError('data must be a dict.')

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
        <DOK: shape=(4, 4), dtype=float64, nnz=4>
        """
        ar = cls(x.shape, dtype=x.dtype)

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
        <DOK: shape=(5, 5), dtype=float64, nnz=4>
        >>> s2 = s.to_coo()
        >>> s2
        <COO: shape=(5, 5), dtype=float64, nnz=4, sorted=False, duplicates=False>
        """
        from .coo import COO
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
        <DOK: shape=(4, 4), dtype=float64, nnz=4>
        """
        ar = cls(x.shape, dtype=x.dtype)

        coords = np.nonzero(x)
        data = x[coords]

        for c in zip(data, *coords):
            d, c = c[0], c[1:]
            ar.data[c] = d

        return ar

    @property
    def ndim(self):
        """
        The number of dimensions in this array.

        Returns
        -------
        int
            The number of dimensions.

        See Also
        --------
        COO.ndim : Equivalent property for :obj:`COO` arrays.
        numpy.ndarray.ndim : Numpy equivalent property.

        Examples
        --------
        >>> s = DOK((1, 2, 3))
        >>> s.ndim
        3
        """
        return len(self.shape)

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

    def __getitem__(self, key):
        key = normalize_index(key, self.shape)

        if not all(isinstance(i, Integral) for i in key):
            raise NotImplementedError('All indices must be integers'
                                      ' when getting an item.')

        if len(key) != self.ndim:
            raise NotImplementedError('Can only get single elements. '
                                      'Expected key of length %d, got %s'
                                      % (self.ndim, str(key)))

        key = tuple(int(k) for k in key)

        if key in self.data:
            return self.data[key]
        else:
            return _zero_of_dtype(self.dtype)[()]

    def __setitem__(self, key, value):
        key = normalize_index(key, self.shape)
        value = np.asanyarray(value)

        value = value.astype(self.dtype)

        key_list = [int(k) if isinstance(k, Integral) else k for k in key]

        self._setitem(key_list, value)

    def _setitem(self, key_list, value):
        value_missing_dims = len([ind for ind in key_list if isinstance(ind, slice)]) - value.ndim

        if value_missing_dims < 0:
            raise ValueError('setting an array element with a sequence.')

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
                    vi = value if value_missing_dims > 0 else \
                        (value[0] if value.shape[0] == 1 else value[v_idx])
                    self._setitem(key_list_temp, vi)

                return
            elif not isinstance(ind, Integral):
                raise IndexError('All indices must be slices or integers'
                                 ' when setting an item.')

        if value != _zero_of_dtype(self.dtype):
            self.data[tuple(key_list)] = value[()]

    def __str__(self):
        return "<DOK: shape=%s, dtype=%s, nnz=%d>" % (self.shape, self.dtype, self.nnz)

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
        result = np.zeros(self.shape, dtype=self.dtype)

        for c, d in six.iteritems(self.data):
            result[c] = d

        return result
