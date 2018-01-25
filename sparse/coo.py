from __future__ import absolute_import, division, print_function

from collections import Iterable, defaultdict, deque
from functools import reduce, partial
import numbers
import operator

import numpy as np
import scipy.sparse

from .slicing import normalize_index
from .utils import _zero_of_dtype

# zip_longest with Python 2/3 compat
from six.moves import range, zip_longest

try:  # Windows compatibility
    int = long
except NameError:
    pass


class COO(object):
    """
    A sparse multidimensional array.

    This is stored in COO format.  It depends on NumPy and Scipy.sparse for
    computation, but supports arrays of arbitrary dimension.

    Parameters
    ----------
    coords : numpy.ndarray (COO.ndim, COO.nnz)
        An array holding the index locations of every value
        Should have shape (number of dimensions, number of non-zeros)
    data : numpy.ndarray (COO.nnz,)
        An array of Values
    shape : tuple[int] (COO.ndim,), optional
        The shape of the array
    has_duplicates : bool, optional
        A value indicating whether the supplied value for :code:`coords` has
        duplicates. Note that setting this to `False` when :code:`coords` does have
        duplicates may result in undefined behaviour. See :obj:`COO.sum_duplicates`
    sorted : bool, optional
        A value indicating whether the values in `coords` are sorted. Note
        that setting this to `False` when :code:`coords` isn't sorted may
        result in undefined behaviour. See :obj:`COO.sort_indices`.
    cache : bool, optional
        Whether to enable cacheing for various operations. See
        :obj:`COO.enable_caching`

    Attributes
    ----------
    coords : numpy.ndarray (ndim, nnz)
        An array holding the coordinates of every nonzero element.
    data : numpy.ndarray (nnz,)
        An array holding the values corresponding to :obj:`COO.coords`.
    shape : tuple[int] (ndim,)
        The dimensions of this array.

    See Also
    --------
    DOK : A mostly write-only sparse array.

    Examples
    --------
    You can create :obj:`COO` objects from Numpy arrays.

    >>> x = np.eye(4, dtype=np.uint8)
    >>> x[2, 3] = 5
    >>> s = COO.from_numpy(x)
    >>> s
    <COO: shape=(4, 4), dtype=uint8, nnz=5, sorted=True, duplicates=False>
    >>> s.data  # doctest: +NORMALIZE_WHITESPACE
    array([1, 1, 1, 5, 1], dtype=uint8)
    >>> s.coords  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 2, 2, 3],
           [0, 1, 2, 3, 3]], dtype=uint8)

    :obj:`COO` objects support basic arithmetic and binary operations.

    >>> x2 = np.eye(4, dtype=np.uint8)
    >>> x2[3, 2] = 5
    >>> s2 = COO.from_numpy(x2)
    >>> (s + s2).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[2, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 2, 5],
           [0, 0, 5, 2]], dtype=uint8)
    >>> (s * s2).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]], dtype=uint8)

    Binary operations support broadcasting.

    >>> x3 = np.zeros((4, 1), dtype=np.uint8)
    >>> x3[2, 0] = 1
    >>> s3 = COO.from_numpy(x3)
    >>> (s * s3).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 1, 5],
           [0, 0, 0, 0]], dtype=uint8)

    :obj:`COO` objects also support dot products and reductions.

    >>> s.dot(s.T).sum(axis=0).todense()   # doctest: +NORMALIZE_WHITESPACE
    array([ 1,  1, 31,  6], dtype=uint64)

    You can use Numpy :code:`ufunc` operations on :obj:`COO` arrays as well.

    >>> np.sum(s, axis=1).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([1, 1, 6, 1], dtype=uint64)
    >>> np.round(np.sqrt(s, dtype=np.float64), decimals=1).todense()   # doctest: +SKIP
    array([[ 1. ,  0. ,  0. ,  0. ],
           [ 0. ,  1. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  2.2],
           [ 0. ,  0. ,  0. ,  1. ]])

    Operations that will result in a dense array will raise a :obj:`ValueError`,
    such as the following.

    >>> np.exp(s)
    Traceback (most recent call last):
        ...
    ValueError: Performing this operation would produce a dense result: <ufunc 'exp'>

    You can also create :obj:`COO` arrays from coordinates and data.

    >>> coords = [[0, 0, 0, 1, 1],
    ...           [0, 1, 2, 0, 3],
    ...           [0, 3, 2, 0, 1]]
    >>> data = [1, 2, 3, 4, 5]
    >>> s4 = COO(coords, data, shape=(3, 4, 5))
    >>> s4
    <COO: shape=(3, 4, 5), dtype=int64, nnz=5, sorted=False, duplicates=True>

    Following scipy.sparse conventions you can also pass these as a tuple with
    rows and columns

    >>> rows = [0, 1, 2, 3, 4]
    >>> cols = [0, 0, 0, 1, 1]
    >>> data = [10, 20, 30, 40, 50]
    >>> z = COO((data, (rows, cols)))
    >>> z.todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[10,  0],
           [20,  0],
           [30,  0],
           [ 0, 40],
           [ 0, 50]])

    You can also pass a dictionary or iterable of index/value pairs. Repeated
    indices imply summation:

    >>> d = {(0, 0, 0): 1, (1, 2, 3): 2, (1, 1, 0): 3}
    >>> COO(d)
    <COO: shape=(2, 3, 4), dtype=int64, nnz=3, sorted=False, duplicates=False>
    >>> L = [((0, 0), 1),
    ...      ((1, 1), 2),
    ...      ((0, 0), 3)]
    >>> COO(L).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[4, 0],
           [0, 2]])

    You can convert :obj:`DOK` arrays to :obj:`COO` arrays.

    >>> from sparse import DOK
    >>> s5 = DOK((5, 5), dtype=np.int64)
    >>> s5[1:3, 1:3] = [[4, 5], [6, 7]]
    >>> s5
    <DOK: shape=(5, 5), dtype=int64, nnz=4>
    >>> s6 = COO(s5)
    >>> s6
    <COO: shape=(5, 5), dtype=int64, nnz=4, sorted=False, duplicates=False>
    >>> s6.todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0, 0, 0],
           [0, 4, 5, 0, 0],
           [0, 6, 7, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    """
    __array_priority__ = 12

    def __init__(self, coords, data=None, shape=None, has_duplicates=True,
                 sorted=False, cache=False):
        self._cache = None
        if cache:
            self.enable_caching()
        if data is None:
            from .dok import DOK

            if isinstance(coords, COO):
                self.coords = coords.coords
                self.data = coords.data
                self.has_duplicates = coords.has_duplicates
                self.sorted = coords.sorted
                self.shape = coords.shape
                return

            if isinstance(coords, DOK):
                shape = coords.shape
                coords = coords.data

            # {(i, j, k): x, (i, j, k): y, ...}
            if isinstance(coords, dict):
                coords = list(coords.items())
                has_duplicates = False

            if isinstance(coords, np.ndarray):
                result = COO.from_numpy(coords)
                self.coords = result.coords
                self.data = result.data
                self.has_duplicates = result.has_duplicates
                self.sorted = result.sorted
                self.shape = result.shape
                return

            if isinstance(coords, scipy.sparse.spmatrix):
                result = COO.from_scipy_sparse(coords)
                self.coords = result.coords
                self.data = result.data
                self.has_duplicates = result.has_duplicates
                self.sorted = result.sorted
                self.shape = result.shape
                return

            # []
            if not coords:
                data = []
                coords = []

            # [((i, j, k), value), (i, j, k), value), ...]
            elif isinstance(coords[0][0], Iterable):
                if coords:
                    assert len(coords[0]) == 2
                data = [x[1] for x in coords]
                coords = [x[0] for x in coords]
                coords = np.asarray(coords).T

            # (data, (row, col, slab, ...))
            else:
                data = coords[0]
                coords = np.stack(coords[1], axis=0)

        self.data = np.asarray(data)
        self.coords = np.asarray(coords)
        if self.coords.ndim == 1:
            self.coords = self.coords[None, :]

        if shape and not self.coords.size:
            self.coords = np.zeros((len(shape), 0), dtype=np.uint64)

        if shape is None:
            if self.coords.nbytes:
                shape = tuple((self.coords.max(axis=1) + 1).tolist())
            else:
                shape = ()

        if isinstance(shape, numbers.Integral):
            shape = (int(shape),)

        self.shape = tuple(shape)
        if self.shape:
            dtype = np.min_scalar_type(max(self.shape))
        else:
            dtype = np.int_
        self.coords = self.coords.astype(dtype)
        assert not self.shape or len(data) == self.coords.shape[1]
        self.has_duplicates = has_duplicates
        self.sorted = sorted

    def enable_caching(self):
        """ Enable caching of reshape, transpose, and tocsr/csc operations

        This enables efficient iterative workflows that make heavy use of
        csr/csc operations, such as tensordot.  This maintains a cache of
        recent results of reshape and transpose so that operations like
        tensordot (which uses both internally) store efficiently stored
        representations for repeated use.  This can significantly cut down on
        computational costs in common numeric algorithms.

        However, this also assumes that neither this object, nor the downstream
        objects will have their data mutated.

        Examples
        --------
        >>> s.enable_caching()  # doctest: +SKIP
        >>> csr1 = s.transpose((2, 0, 1)).reshape((100, 120)).tocsr()  # doctest: +SKIP
        >>> csr2 = s.transpose((2, 0, 1)).reshape((100, 120)).tocsr()  # doctest: +SKIP
        >>> csr1 is csr2  # doctest: +SKIP
        True
        """
        self._cache = defaultdict(lambda: deque(maxlen=3))
        return self

    @classmethod
    def from_numpy(cls, x):
        """
        Convert the given :obj:`numpy.ndarray` to a :obj:`COO` object.

        Parameters
        ----------
        x : np.ndarray
            The dense array to convert.

        Returns
        -------
        COO
            The converted COO array.

        Examples
        --------
        >>> x = np.eye(5)
        >>> s = COO.from_numpy(x)
        >>> s
        <COO: shape=(5, 5), dtype=float64, nnz=5, sorted=True, duplicates=False>
        """
        x = np.asanyarray(x)
        if x.shape:
            coords = np.where(x)
            data = x[coords]
            coords = np.vstack(coords)
        else:
            coords = np.empty((0, 1), dtype=np.uint8)
            data = np.array(x, ndmin=1)
        return cls(coords, data, shape=x.shape, has_duplicates=False,
                   sorted=True)

    def todense(self):
        """
        Convert this :obj:`COO` array to a dense :obj:`numpy.ndarray`. Note that
        this may take a large amount of memory if the :obj:`COO` object's :code:`shape`
        is large.

        Returns
        -------
        numpy.ndarray
            The converted dense array.

        See Also
        --------
        DOK.todense : Equivalent :obj:`DOK` array method.
        scipy.sparse.coo_matrix.todense : Equivalent Scipy method.

        Examples
        --------
        >>> x = np.random.randint(100, size=(7, 3))
        >>> s = COO.from_numpy(x)
        >>> x2 = s.todense()
        >>> np.array_equal(x, x2)
        True
        """
        self.sum_duplicates()
        x = np.zeros(shape=self.shape, dtype=self.dtype)

        coords = tuple([self.coords[i, :] for i in range(self.ndim)])
        data = self.data

        if coords != ():
            x[coords] = data
        else:
            if len(data) != 0:
                x[coords] = data

        return x

    @classmethod
    def from_scipy_sparse(cls, x):
        """
        Construct a :obj:`COO` array from a :obj:`scipy.sparse.spmatrix`

        Parameters
        ----------
        x : scipy.sparse.spmatrix
            The sparse matrix to construct the array from.

        Returns
        -------
        COO
            The converted :obj:`COO` object.

        Examples
        --------
        >>> x = scipy.sparse.rand(6, 3, density=0.2)
        >>> s = COO.from_scipy_sparse(x)
        >>> np.array_equal(x.todense(), s.todense())
        True
        """
        x = scipy.sparse.coo_matrix(x)
        coords = np.empty((2, x.nnz), dtype=x.row.dtype)
        coords[0, :] = x.row
        coords[1, :] = x.col
        return COO(coords, x.data, shape=x.shape,
                   has_duplicates=not x.has_canonical_format,
                   sorted=x.has_canonical_format)

    @property
    def dtype(self):
        """
        The datatype of this array.

        Returns
        -------
        numpy.dtype
            The datatype of this array.

        See Also
        --------
        numpy.ndarray.dtype : Numpy equivalent property.
        scipy.sparse.coo_matrix.dtype : Scipy equivalent property.

        Examples
        --------
        >>> x = (200 * np.random.rand(5, 4)).astype(np.int32)
        >>> s = COO.from_numpy(x)
        >>> s.dtype
        dtype('int32')
        >>> x.dtype == s.dtype
        True
        """
        return self.data.dtype

    @property
    def ndim(self):
        """
        The number of dimensions of this array.

        Returns
        -------
        int
            The number of dimensions of this array.

        See Also
        --------
        DOK.ndim : Equivalent property for :obj:`DOK` arrays.
        numpy.ndarray.ndim : Numpy equivalent property.

        Examples
        --------
        >>> x = np.random.rand(1, 2, 3, 1, 2)
        >>> s = COO.from_numpy(x)
        >>> s.ndim
        5
        >>> s.ndim == x.ndim
        True
        """
        return len(self.shape)

    @property
    def nnz(self):
        """
        The number of nonzero elements in this array. Note that any duplicates in
        :code:`coords` are counted multiple times. To avoid this, call :obj:`COO.sum_duplicates`.

        Returns
        -------
        int
            The number of nonzero elements in this array.

        See Also
        --------
        DOK.nnz : Equivalent :obj:`DOK` array property.
        numpy.count_nonzero : A similar Numpy function.
        scipy.sparse.coo_matrix.nnz : The Scipy equivalent property.

        Examples
        --------
        >>> x = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 0])
        >>> np.count_nonzero(x)
        6
        >>> s = COO.from_numpy(x)
        >>> s.nnz
        6
        >>> np.count_nonzero(x) == s.nnz
        True
        """
        return self.coords.shape[1]

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
        >>> data = np.arange(6, dtype=np.uint8)
        >>> coords = np.random.randint(1000, size=(3, 6), dtype=np.uint16)
        >>> s = COO(coords, data, shape=(1000, 1000, 1000))
        >>> s.nbytes
        42
        """
        return self.data.nbytes + self.coords.nbytes

    def __len__(self):
        """
        Get "length" of array, which is by definition the size of the first
        dimension.

        Returns
        -------
        int
            The size of the first dimension.

        See Also
        --------
        numpy.ndarray.__len__ : Numpy equivalent property.

        Examples
        --------
        >>> x = np.zeros((10, 10))
        >>> s = COO.from_numpy(x)
        >>> len(s)
        10
        """
        return self.shape[0]

    @property
    def size(self):
        """
        The number of all elements (including zeros) in this array.

        Returns
        -------
        int
            The number of elements.

        See Also
        --------
        numpy.ndarray.size : Numpy equivalent property.

        Examples
        --------
        >>> x = np.zeros((10, 10))
        >>> s = COO.from_numpy(x)
        >>> s.size
        100
        """
        return np.prod(self.shape)

    @property
    def density(self):
        """
        The ratio of nonzero to all elements in this array.

        Returns
        -------
        float
            The ratio of nonzero to all elements.

        See Also
        --------
        COO.size : Number of elements.
        COO.nnz : Number of nonzero elements.

        Examples
        --------
        >>> x = np.zeros((8, 8))
        >>> x[0, :] = 1
        >>> s = COO.from_numpy(x)
        >>> s.density
        0.125
        """
        return self.nnz / self.size

    def __sizeof__(self):
        return self.nbytes

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            if isinstance(index, str):
                data = self.data[index]
                idx = np.where(data)
                coords = list(self.coords[:, idx[0]])
                coords.extend(idx[1:])

                return COO(coords, data[idx].flatten(),
                           shape=self.shape + self.data.dtype[index].shape,
                           has_duplicates=self.has_duplicates,
                           sorted=self.sorted)
            else:
                index = (index,)

        last_ellipsis = len(index) > 0 and index[-1] is Ellipsis
        index = normalize_index(index, self.shape)
        if len(index) != 0 and all(not isinstance(ind, Iterable) and ind == slice(None) for ind in index):
            return self
        mask = np.ones(self.nnz, dtype=np.bool)
        for i, ind in enumerate([i for i in index if i is not None]):
            if not isinstance(ind, Iterable) and ind == slice(None):
                continue
            mask &= _mask(self.coords[i], ind, self.shape[i])

        n = mask.sum()
        coords = []
        shape = []
        i = 0
        for ind in index:
            if isinstance(ind, numbers.Integral):
                i += 1
                continue
            elif isinstance(ind, slice):
                step = ind.step if ind.step is not None else 1
                if step > 0:
                    start = ind.start if ind.start is not None else 0
                    start = max(start, 0)
                    stop = ind.stop if ind.stop is not None else self.shape[i]
                    stop = min(stop, self.shape[i])
                    if start > stop:
                        start = stop
                    shape.append((stop - start + step - 1) // step)
                else:
                    start = ind.start or self.shape[i] - 1
                    stop = ind.stop if ind.stop is not None else -1
                    start = min(start, self.shape[i] - 1)
                    stop = max(stop, -1)
                    if start < stop:
                        start = stop
                    shape.append((start - stop - step - 1) // (-step))

                dt = np.min_scalar_type(min(-(dim - 1) if dim != 0 else -1 for dim in shape))
                coords.append((self.coords[i, mask].astype(dt) - start) // step)
                i += 1
            elif isinstance(ind, Iterable):
                old = self.coords[i][mask]
                new = np.empty(shape=old.shape, dtype=old.dtype)
                for j, item in enumerate(ind):
                    new[old == item] = j
                coords.append(new)
                shape.append(len(ind))
                i += 1
            elif ind is None:
                coords.append(np.zeros(n))
                shape.append(1)

        for j in range(i, self.ndim):
            coords.append(self.coords[j][mask])
            shape.append(self.shape[j])

        if coords:
            coords = np.stack(coords, axis=0)
        else:
            if last_ellipsis:
                coords = np.empty((0, np.sum(mask)), dtype=np.uint8)
            else:
                if np.sum(mask) != 0:
                    return self.data[mask][0]
                else:
                    return _zero_of_dtype(self.dtype)[()]
        shape = tuple(shape)
        data = self.data[mask]

        return COO(coords, data, shape=shape,
                   has_duplicates=self.has_duplicates,
                   sorted=self.sorted)

    def __str__(self):
        return "<COO: shape=%s, dtype=%s, nnz=%d, sorted=%s, duplicates=%s>" % (
            self.shape, self.dtype, self.nnz, self.sorted,
            self.has_duplicates)

    __repr__ = __str__

    @staticmethod
    def _reduce(method, *args, **kwargs):
        assert len(args) == 1

        self = args[0]
        if isinstance(self, scipy.sparse.spmatrix):
            self = COO.from_scipy_sparse(self)

        return self.reduce(method, **kwargs)

    def reduce(self, method, axis=None, keepdims=False, **kwargs):
        """
        Performs a reduction operation on this array.

        Parameters
        ----------
        method : numpy.ufunc
            The method to use for performing the reduction.
        axis : Union[int, Iterable[int]], optional
            The axes along which to perform the reduction. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        kwargs : dict
            Any extra arguments to pass to the reduction operation.

        Returns
        -------
        COO
            The result of the reduction operation.

        Raises
        ------
        ValueError
            If reducing an all-zero axis would produce a nonzero result.

        Notes
        -----
        This function internally calls :obj:`COO.sum_duplicates` to bring the array into
        canonical form.

        See Also
        --------
        numpy.ufunc.reduce : A similar Numpy method.

        Examples
        --------
        You can use the :obj:`COO.reduce` method to apply a reduction operation to
        any Numpy :code:`ufunc`.

        >>> x = np.ones((5, 5), dtype=np.int)
        >>> s = COO.from_numpy(x)
        >>> s2 = s.reduce(np.add, axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([5, 5, 5, 5, 5])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        reduction.

        >>> s3 = s.reduce(np.add, axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        You can also pass in any keyword argument that :obj:`numpy.ufunc.reduce` supports.
        For example, :code:`dtype`. Note that :code:`out` isn't supported.

        >>> s4 = s.reduce(np.add, axis=1, dtype=np.float16)
        >>> s4.dtype
        dtype('float16')

        By default, this reduces the array down to one number, reducing along all axes.

        >>> s.reduce(np.add)
        25
        """
        zero_reduce_result = method.reduce([_zero_of_dtype(self.dtype)], **kwargs)

        if zero_reduce_result != _zero_of_dtype(np.dtype(zero_reduce_result)):
            raise ValueError("Performing this reduction operation would produce "
                             "a dense result: %s" % str(method))

        # Needed for more esoteric reductions like product.
        self.sum_duplicates()

        if axis is None:
            axis = tuple(range(self.ndim))

        if not isinstance(axis, tuple):
            axis = (axis,)

        if set(axis) == set(range(self.ndim)):
            result = method.reduce(self.data, **kwargs)
            if self.nnz != self.size:
                result = method(result, _zero_of_dtype(self.dtype)[()], **kwargs)
        else:
            axis = tuple(axis)
            neg_axis = tuple(ax for ax in range(self.ndim) if ax not in axis)

            a = self.transpose(neg_axis + axis)
            a = a.reshape((np.prod([self.shape[d] for d in neg_axis]),
                           np.prod([self.shape[d] for d in axis])))
            a.sort_indices()

            result, inv_idx, counts = _grouped_reduce(a.data, a.coords[0], method, **kwargs)
            missing_counts = counts != a.shape[1]
            result[missing_counts] = method(result[missing_counts],
                                            _zero_of_dtype(self.dtype), **kwargs)
            coords = a.coords[0:1, inv_idx]
            a = COO(coords, result, shape=(a.shape[0],),
                    has_duplicates=False, sorted=True)

            a = a.reshape([self.shape[d] for d in neg_axis])
            result = a

        if keepdims:
            result = _keepdims(self, result, axis)
        return result

    def sum(self, axis=None, keepdims=False, dtype=None, out=None):
        """
        Performs a sum operation along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to sum. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        COO
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.sum` : Equivalent numpy function.
        scipy.sparse.coo_matrix.sum : Equivalent Scipy function.

        Notes
        -----
        * This function internally calls :obj:`COO.sum_duplicates` to bring the array into
          canonical form.
        * The :code:`out` parameter is provided just for compatibility with Numpy and
          isn't actually supported.

        Examples
        --------
        You can use :obj:`COO.sum` to sum an array across any dimension.

        >>> x = np.ones((5, 5), dtype=np.int)
        >>> s = COO.from_numpy(x)
        >>> s2 = s.sum(axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([5, 5, 5, 5, 5])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        sum.

        >>> s3 = s.sum(axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        You can pass in an output datatype, if needed.

        >>> s4 = s.sum(axis=1, dtype=np.float16)
        >>> s4.dtype
        dtype('float16')

        By default, this reduces the array down to one number, summing along all axes.

        >>> s.sum()
        25
        """
        assert out is None
        return self.reduce(np.add, axis=axis, keepdims=keepdims, dtype=dtype)

    def max(self, axis=None, keepdims=False, out=None):
        """
        Maximize along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to maximize. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        COO
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.max` : Equivalent numpy function.
        scipy.sparse.coo_matrix.max : Equivalent Scipy function.

        Notes
        -----
        * This function internally calls :obj:`COO.sum_duplicates` to bring the array into
          canonical form.
        * The :code:`out` parameter is provided just for compatibility with Numpy and
          isn't actually supported.

        Examples
        --------
        You can use :obj:`COO.max` to maximize an array across any dimension.

        >>> x = np.add.outer(np.arange(5), np.arange(5))
        >>> x  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 1, 2, 3, 4],
               [1, 2, 3, 4, 5],
               [2, 3, 4, 5, 6],
               [3, 4, 5, 6, 7],
               [4, 5, 6, 7, 8]])
        >>> s = COO.from_numpy(x)
        >>> s2 = s.max(axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([4, 5, 6, 7, 8])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        maximization.

        >>> s3 = s.max(axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        By default, this reduces the array down to one number, maximizing along all axes.

        >>> s.max()
        8
        """
        assert out is None
        return self.reduce(np.maximum, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False, out=None):
        """
        Minimize along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to minimize. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        COO
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.min` : Equivalent numpy function.
        scipy.sparse.coo_matrix.min : Equivalent Scipy function.

        Notes
        -----
        * This function internally calls :obj:`COO.sum_duplicates` to bring the array into
          canonical form.
        * The :code:`out` parameter is provided just for compatibility with Numpy and
          isn't actually supported.

        Examples
        --------
        You can use :obj:`COO.min` to minimize an array across any dimension.

        >>> x = np.add.outer(np.arange(5), np.arange(5))
        >>> x  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 1, 2, 3, 4],
               [1, 2, 3, 4, 5],
               [2, 3, 4, 5, 6],
               [3, 4, 5, 6, 7],
               [4, 5, 6, 7, 8]])
        >>> s = COO.from_numpy(x)
        >>> s2 = s.min(axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([0, 1, 2, 3, 4])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        minimization.

        >>> s3 = s.min(axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        By default, this reduces the array down to one number, minimizing along all axes.

        >>> s.min()
        0
        """
        assert out is None
        return self.reduce(np.minimum, axis=axis, keepdims=keepdims)

    def prod(self, axis=None, keepdims=False, dtype=None, out=None):
        """
        Performs a product operation along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to multiply. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        COO
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.prod` : Equivalent numpy function.

        Notes
        -----
        * This function internally calls :obj:`COO.sum_duplicates` to bring the array into
          canonical form.
        * The :code:`out` parameter is provided just for compatibility with Numpy and
          isn't actually supported.

        Examples
        --------
        You can use :obj:`COO.prod` to multiply an array across any dimension.

        >>> x = np.add.outer(np.arange(5), np.arange(5))
        >>> x  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 1, 2, 3, 4],
               [1, 2, 3, 4, 5],
               [2, 3, 4, 5, 6],
               [3, 4, 5, 6, 7],
               [4, 5, 6, 7, 8]])
        >>> s = COO.from_numpy(x)
        >>> s2 = s.prod(axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([   0,  120,  720, 2520, 6720])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        reduction.

        >>> s3 = s.prod(axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        You can pass in an output datatype, if needed.

        >>> s4 = s.prod(axis=1, dtype=np.float16)
        >>> s4.dtype
        dtype('float16')

        By default, this reduces the array down to one number, multiplying along all axes.

        >>> s.prod()
        0
        """
        assert out is None
        return self.reduce(np.multiply, axis=axis, keepdims=keepdims, dtype=dtype)

    def transpose(self, axes=None):
        """
        Returns a new array which has the order of the axes switched.

        Parameters
        ----------
        axes : Iterable[int], optional
            The new order of the axes compared to the previous one. Reverses the axes
            by default.

        Returns
        -------
        COO
            The new array with the axes in the desired order.

        See Also
        --------
        :obj:`COO.T` : A quick property to reverse the order of the axes.
        numpy.ndarray.transpose : Numpy equivalent function.

        Examples
        --------
        We can change the order of the dimensions of any :obj:`COO` array with this
        function.

        >>> x = np.add.outer(np.arange(5), np.arange(5)[::-1])
        >>> x  # doctest: +NORMALIZE_WHITESPACE
        array([[4, 3, 2, 1, 0],
               [5, 4, 3, 2, 1],
               [6, 5, 4, 3, 2],
               [7, 6, 5, 4, 3],
               [8, 7, 6, 5, 4]])
        >>> s = COO.from_numpy(x)
        >>> s.transpose((1, 0)).todense()  # doctest: +NORMALIZE_WHITESPACE
        array([[4, 5, 6, 7, 8],
               [3, 4, 5, 6, 7],
               [2, 3, 4, 5, 6],
               [1, 2, 3, 4, 5],
               [0, 1, 2, 3, 4]])

        Note that by default, this reverses the order of the axes rather than switching
        the last and second-to-last axes as required by some linear algebra operations.

        >>> x = np.random.rand(2, 3, 4)
        >>> s = COO.from_numpy(x)
        >>> s.transpose().shape
        (4, 3, 2)
        """
        if axes is None:
            axes = list(reversed(range(self.ndim)))

        # Normalize all axe indices to posivite values
        axes = np.array(axes)
        axes[axes < 0] += self.ndim

        if np.any(axes >= self.ndim) or np.any(axes < 0):
            raise ValueError("invalid axis for this array")

        if len(np.unique(axes)) < len(axes):
            raise ValueError("repeated axis in transpose")

        if not len(axes) == self.ndim:
            raise ValueError("axes don't match array")

        # Normalize all axe indices to posivite values
        try:
            axes = np.arange(self.ndim)[list(axes)]
        except IndexError:
            raise ValueError("invalid axis for this array")

        if len(np.unique(axes)) < len(axes):
            raise ValueError("repeated axis in transpose")

        if not len(axes) == self.ndim:
            raise ValueError("axes don't match array")

        axes = tuple(axes)

        if axes == tuple(range(self.ndim)):
            return self

        if self._cache is not None:
            for ax, value in self._cache['transpose']:
                if ax == axes:
                    return value

        shape = tuple(self.shape[ax] for ax in axes)
        result = COO(self.coords[axes, :], self.data, shape,
                     has_duplicates=self.has_duplicates,
                     cache=self._cache is not None)

        if self._cache is not None:
            self._cache['transpose'].append((axes, result))
        return result

    @property
    def T(self):
        """
        Returns a new array which has the order of the axes reversed.

        Returns
        -------
        COO
            The new array with the axes in the desired order.

        See Also
        --------
        :obj:`COO.transpose` : A method where you can specify the order of the axes.
        numpy.ndarray.T : Numpy equivalent property.

        Examples
        --------
        We can change the order of the dimensions of any :obj:`COO` array with this
        function.

        >>> x = np.add.outer(np.arange(5), np.arange(5)[::-1])
        >>> x  # doctest: +NORMALIZE_WHITESPACE
        array([[4, 3, 2, 1, 0],
               [5, 4, 3, 2, 1],
               [6, 5, 4, 3, 2],
               [7, 6, 5, 4, 3],
               [8, 7, 6, 5, 4]])
        >>> s = COO.from_numpy(x)
        >>> s.T.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([[4, 5, 6, 7, 8],
               [3, 4, 5, 6, 7],
               [2, 3, 4, 5, 6],
               [1, 2, 3, 4, 5],
               [0, 1, 2, 3, 4]])

        Note that by default, this reverses the order of the axes rather than switching
        the last and second-to-last axes as required by some linear algebra operations.

        >>> x = np.random.rand(2, 3, 4)
        >>> s = COO.from_numpy(x)
        >>> s.T.shape
        (4, 3, 2)
        """
        return self.transpose(tuple(range(self.ndim))[::-1])

    def dot(self, other):
        """
        Performs the equivalent of :code:`x.dot(y)` for :obj:`COO`.

        Parameters
        ----------
        other : Union[COO, numpy.ndarray, scipy.sparse.spmatrix]
            The second operand of the dot product operation.

        Returns
        -------
        {COO, numpy.ndarray}
            The result of the dot product. If the result turns out to be dense,
            then a dense array is returned, otherwise, a sparse array.

        See Also
        --------
        dot : Equivalent function for two arguments.
        :obj:`numpy.dot` : Numpy equivalent function.
        scipy.sparse.coo_matrix.dot : Scipy equivalent function.

        Examples
        --------
        >>> x = np.arange(4).reshape((2, 2))
        >>> s = COO.from_numpy(x)
        >>> s.dot(s) # doctest: +SKIP
        array([[ 2,  3],
               [ 6, 11]], dtype=int64)
        """
        return dot(self, other)

    def __matmul__(self, other):
        try:
            return dot(self, other)
        except NotImplementedError:
            return NotImplemented

    def __rmatmul__(self, other):
        try:
            return dot(other, self)
        except NotImplementedError:
            return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            return COO._elemwise(ufunc, *inputs, **kwargs)
        elif method == 'reduce':
            return COO._reduce(ufunc, *inputs, **kwargs)
        else:
            return NotImplemented

    def __array__(self, dtype=None, **kwargs):
        x = self.todense()
        if dtype and x.dtype != dtype:
            x = x.astype(dtype)
        return x

    def linear_loc(self, signed=False):
        """
        The nonzero coordinates of a flattened version of this array. Note that
        the coordinates may be out of order.

        Parameters
        ----------
        signed : bool, optional
            Whether to use a signed datatype for the output array. :code:`False`
            by default.

        Returns
        -------
        numpy.ndarray
            The flattened coordinates.

        See Also
        --------
        :obj:`numpy.flatnonzero` : Equivalent Numpy function.

        Examples
        --------
        >>> x = np.eye(5)
        >>> s = COO.from_numpy(x)
        >>> s.linear_loc()  # doctest: +NORMALIZE_WHITESPACE
        array([ 0,  6, 12, 18, 24], dtype=uint8)
        >>> np.array_equal(np.flatnonzero(x), s.linear_loc())
        True
        """
        return _linear_loc(self.coords, self.shape, signed)

    def reshape(self, shape):
        """
        Returns a new :obj:`COO` array that is a reshaped version of this array.

        Parameters
        ----------
        shape : tuple[int]
            The desired shape of the output array.

        Returns
        -------
        COO
            The reshaped output array.

        See Also
        --------
        numpy.ndarray.reshape : The equivalent Numpy function.

        Examples
        --------
        >>> s = COO.from_numpy(np.arange(25))
        >>> s2 = s.reshape((5, 5))
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])
        """
        if self.shape == shape:
            return self
        if any(d == -1 for d in shape):
            extra = int(self.size /
                        np.prod([d for d in shape if d != -1]))
            shape = tuple([d if d != -1 else extra for d in shape])

        if self.shape == shape:
            return self

        if self._cache is not None:
            for sh, value in self._cache['reshape']:
                if sh == shape:
                    return value

        # TODO: this self.size enforces a 2**64 limit to array size
        linear_loc = self.linear_loc()

        max_shape = max(shape) if len(shape) != 0 else 1
        coords = np.empty((len(shape), self.nnz), dtype=np.min_scalar_type(max_shape - 1))
        strides = 1
        for i, d in enumerate(shape[::-1]):
            coords[-(i + 1), :] = (linear_loc // strides) % d
            strides *= d

        result = COO(coords, self.data, shape,
                     has_duplicates=self.has_duplicates,
                     sorted=self.sorted, cache=self._cache is not None)

        if self._cache is not None:
            self._cache['reshape'].append((shape, result))
        return result

    def to_scipy_sparse(self):
        """
        Converts this :obj:`COO` object into a :obj:`scipy.sparse.coo_matrix`.

        Returns
        -------
        :obj:`scipy.sparse.coo_matrix`
            The converted Scipy sparse matrix.

        Raises
        ------
        ValueError
            If the array is not two-dimensional.

        See Also
        --------
        COO.tocsr : Convert to a :obj:`scipy.sparse.csr_matrix`.
        COO.tocsc : Convert to a :obj:`scipy.sparse.csc_matrix`.
        """
        if self.ndim != 2:
            raise ValueError("Can only convert a 2-dimensional array to a Scipy sparse matrix.")

        result = scipy.sparse.coo_matrix((self.data,
                                          (self.coords[0],
                                           self.coords[1])),
                                         shape=self.shape)
        result.has_canonical_format = (not self.has_duplicates and self.sorted)
        return result

    def _tocsr(self):
        if self.ndim != 2:
            raise ValueError('This array must be two-dimensional for this conversion '
                             'to work.')

        # Pass 1: sum duplicates
        self.sum_duplicates()

        # Pass 2: sort indices
        self.sort_indices()
        row, col = self.coords

        # Pass 3: count nonzeros in each row
        indptr = np.zeros(self.shape[0] + 1, dtype=np.int64)
        np.cumsum(np.bincount(row, minlength=self.shape[0]), out=indptr[1:])

        return scipy.sparse.csr_matrix((self.data, col, indptr), shape=self.shape)

    def tocsr(self):
        """
        Converts this array to a :obj:`scipy.sparse.csr_matrix`.

        Returns
        -------
        scipy.sparse.csr_matrix
            The result of the conversion.

        Raises
        ------
        ValueError
            If the array is not two-dimensional.

        See Also
        --------
        COO.tocsc : Convert to a :obj:`scipy.sparse.csc_matrix`.
        COO.to_scipy_sparse : Convert to a :obj:`scipy.sparse.coo_matrix`.
        scipy.sparse.coo_matrix.tocsr : Equivalent Scipy function.
        """
        if self._cache is not None:
            try:
                return self._csr
            except AttributeError:
                pass
            try:
                self._csr = self._csc.tocsr()
                return self._csr
            except AttributeError:
                pass

            self._csr = csr = self._tocsr()
        else:
            csr = self._tocsr()
        return csr

    def tocsc(self):
        """
        Converts this array to a :obj:`scipy.sparse.csc_matrix`.

        Returns
        -------
        scipy.sparse.csc_matrix
            The result of the conversion.

        Raises
        ------
        ValueError
            If the array is not two-dimensional.

        See Also
        --------
        COO.tocsr : Convert to a :obj:`scipy.sparse.csr_matrix`.
        COO.to_scipy_sparse : Convert to a :obj:`scipy.sparse.coo_matrix`.
        scipy.sparse.coo_matrix.tocsc : Equivalent Scipy function.
        """
        if self._cache is not None:
            try:
                return self._csc
            except AttributeError:
                pass
            try:
                self._csc = self._csr.tocsc()
                return self._csc
            except AttributeError:
                pass

            self._csc = csc = self.tocsr().tocsc()
        else:
            csc = self.tocsr().tocsc()

        return csc

    def sort_indices(self):
        """
        Sorts the :obj:`COO.coords` attribute. Also sorts the data in
        :obj:`COO.data` to match.

        Examples
        --------
        >>> coords = np.array([[1, 2, 0]], dtype=np.uint8)
        >>> data = np.array([4, 1, 3], dtype=np.uint8)
        >>> s = COO(coords, data)
        >>> s.sort_indices()
        >>> s.coords  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 1, 2]], dtype=uint8)
        >>> s.data  # doctest: +NORMALIZE_WHITESPACE
        array([3, 4, 1], dtype=uint8)
        """
        if self.sorted:
            return

        linear = self.linear_loc(signed=True)

        if (np.diff(linear) > 0).all():  # already sorted
            self.sorted = True
            return

        order = np.argsort(linear)
        self.coords = self.coords[:, order]
        self.data = self.data[order]
        self.sorted = True

    def sum_duplicates(self):
        """
        Sums data corresponding to duplicates in :obj:`COO.coords`.

        See Also
        --------
        scipy.sparse.coo_matrix.sum_duplicates : Equivalent Scipy function.

        Examples
        --------
        >>> coords = np.array([[0, 1, 1, 2]], dtype=np.uint8)
        >>> data = np.array([6, 5, 2, 2], dtype=np.uint8)
        >>> s = COO(coords, data)
        >>> s.sum_duplicates()
        >>> s.coords  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 1, 2]], dtype=uint8)
        >>> s.data  # doctest: +NORMALIZE_WHITESPACE
        array([6, 7, 2], dtype=uint8)
        """
        # Inspired by scipy/sparse/coo.py::sum_duplicates
        # See https://github.com/scipy/scipy/blob/master/LICENSE.txt
        if not self.has_duplicates and self.sorted:
            return
        if not self.coords.size:
            return

        self.sort_indices()

        linear = self.linear_loc()
        unique_mask = np.diff(linear) != 0

        if unique_mask.sum() == len(unique_mask):  # already unique
            self.has_duplicates = False
            return

        unique_mask = np.append(True, unique_mask)

        coords = self.coords[:, unique_mask]
        (unique_inds,) = np.nonzero(unique_mask)
        data = np.add.reduceat(self.data, unique_inds, dtype=self.data.dtype)

        self.data = data
        self.coords = coords
        self.has_duplicates = False

    def __add__(self, other):
        return self.elemwise(operator.add, other)

    def __radd__(self, other):
        return self.elemwise(_reverse_self_other(operator.add), other)

    def __neg__(self):
        return self.elemwise(operator.neg)

    def __sub__(self, other):
        return self.elemwise(operator.sub, other)

    def __rsub__(self, other):
        return self.elemwise(_reverse_self_other(operator.sub), other)

    def __mul__(self, other):
        return self.elemwise(operator.mul, other)

    def __rmul__(self, other):
        return self.elemwise(_reverse_self_other(operator.mul), other)

    def __truediv__(self, other):
        return self.elemwise(operator.truediv, other)

    def __rtruediv__(self, other):
        return self.elemwise(_reverse_self_other(operator.truediv), other)

    def __floordiv__(self, other):
        return self.elemwise(operator.floordiv, other)

    def __rfloordiv__(self, other):
        return self.elemwise(_reverse_self_other(operator.floordiv), other)

    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __pow__(self, other):
        return self.elemwise(operator.pow, other)

    def __rpow__(self, other):
        return self.elemwise(_reverse_self_other(operator.pow), other)

    def __mod__(self, other):
        return self.elemwise(operator.mod, other)

    def __rmod__(self, other):
        return self.elemwise(_reverse_self_other(operator.mod), other)

    def __and__(self, other):
        return self.elemwise(operator.and_, other)

    def __rand__(self, other):
        return self.elemwise(_reverse_self_other(operator.and_), other)

    def __xor__(self, other):
        return self.elemwise(operator.xor, other)

    def __rxor__(self, other):
        return self.elemwise(_reverse_self_other(operator.xor), other)

    def __or__(self, other):
        return self.elemwise(operator.or_, other)

    def __ror__(self, other):
        return self.elemwise(_reverse_self_other(operator.or_), other)

    def __invert__(self):
        return self.elemwise(operator.invert)

    def __gt__(self, other):
        return self.elemwise(operator.gt, other)

    def __ge__(self, other):
        return self.elemwise(operator.ge, other)

    def __lt__(self, other):
        return self.elemwise(operator.lt, other)

    def __le__(self, other):
        return self.elemwise(operator.le, other)

    def __eq__(self, other):
        return self.elemwise(operator.eq, other)

    def __ne__(self, other):
        return self.elemwise(operator.ne, other)

    def __lshift__(self, other):
        return self.elemwise(operator.lshift, other)

    def __rlshift__(self, other):
        return self.elemwise(_reverse_self_other(operator.lshift), other)

    def __rshift__(self, other):
        return self.elemwise(operator.rshift, other)

    def __rrshift__(self, other):
        return self.elemwise(_reverse_self_other(operator.rshift), other)

    @staticmethod
    def _elemwise(func, *args, **kwargs):
        if len(args) == 0:
            return func()

        self = args[0]
        if isinstance(self, scipy.sparse.spmatrix):
            self = COO.from_numpy(self)
        elif np.isscalar(self) or (isinstance(self, np.ndarray)
                                   and self.ndim == 0):
            func = partial(func, self)
            other = args[1]
            if isinstance(other, scipy.sparse.spmatrix):
                other = COO.from_scipy_sparse(other)
            return _elemwise_unary(func, other, *args[2:], **kwargs)

        if len(args) == 1:
            return _elemwise_unary(func, self, *args[1:], **kwargs)
        else:
            other = args[1]
            if isinstance(other, scipy.sparse.spmatrix):
                other = COO.from_scipy_sparse(other)

            if isinstance(other, COO) or isinstance(other, np.ndarray):
                return _elemwise_binary(func, self, other, *args[2:], **kwargs)
            else:
                return _elemwise_unary(func, self, *args[1:], **kwargs)

    def elemwise(self, func, *args, **kwargs):
        """
        Apply a function to one or two arguments.

        Parameters
        ----------
        func : Callable
            The function to apply to one or two arguments.
        args : tuple, optional
            The extra arguments to pass to the function. If :code:`args[0]` is a COO object,
            a scipy.sparse.spmatrix or a scalar; the function will be treated as a binary
            function. Otherwise, it will be treated as a unary function.
        kwargs : dict, optional
            The kwargs to pass to the function.

        Returns
        -------
        COO
            The result of applying the function.

        Raises
        ------
        ValueError
            If the operation would result in a dense matrix.

        See Also
        --------
        :obj:`numpy.ufunc` : A similar Numpy construct. Note that any :code:`ufunc` can be used
            as the :code:`func` input to this function.
        """
        return COO._elemwise(func, self, *args, **kwargs)

    def broadcast_to(self, shape):
        """
        Performs the equivalent of :obj:`numpy.broadcast_to` for :obj:`COO`. Note that
        this function returns a new array instead of a view.

        Parameters
        ----------
        shape : tuple[int]
            The shape to broadcast the data to.

        Returns
        -------
        COO
            The broadcasted sparse array.

        Raises
        ------
        ValueError
            If the operand cannot be broadcast to the given shape.

        See also
        --------
        :obj:`numpy.broadcast_to` : NumPy equivalent function
        """
        result_shape = _get_broadcast_shape(self.shape, shape, is_result=True)
        params = _get_broadcast_parameters(self.shape, result_shape)
        coords, data = _get_expanded_coords_data(self.coords, self.data, params, result_shape)

        return COO(coords, data, shape=result_shape, has_duplicates=self.has_duplicates,
                   sorted=self.sorted)

    def __abs__(self):
        """
        Calculate the absolute value element-wise.

        See also
        --------
        :obj:`numpy.absolute` : NumPy equivalent ufunc.
        """
        return self.elemwise(abs)

    abs = __abs__

    def exp(self, out=None):
        """
        Calculate the exponential of all elements in the array.

        See also
        --------
        :obj:`numpy.exp` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.exp)

    def expm1(self, out=None):
        """
        Calculate :code:`exp(x) - 1` for all elements in the array.

        See also
        --------
        scipy.sparse.coo_matrix.expm1 : SciPy sparse equivalent function
        :obj:`numpy.expm1` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.expm1)

    def log1p(self, out=None):
        """
        Return the natural logarithm of one plus the input array, element-wise.

        Calculates :code:`log(1 + x)`.

        See also
        --------
        scipy.sparse.coo_matrix.log1p : SciPy sparse equivalent function
        :obj:`numpy.log1p` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.log1p)

    def sin(self, out=None):
        """
        Trigonometric sine, element-wise.

        See also
        --------
        scipy.sparse.coo_matrix.sin : SciPy sparse equivalent function
        :obj:`numpy.sin` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.sin)

    def sinh(self, out=None):
        """
        Hyperbolic sine, element-wise.

        See also
        --------
        scipy.sparse.coo_matrix.sinh : SciPy sparse equivalent function
        :obj:`numpy.sinh` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.sinh)

    def tan(self, out=None):
        """
        Compute tangent element-wise.

        See also
        --------
        scipy.sparse.coo_matrix.tan : SciPy sparse equivalent function
        :obj:`numpy.tan` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.
        """
        assert out is None
        return self.elemwise(np.tan)

    def tanh(self, out=None):
        """
        Compute hyperbolic tangent element-wise.

        See also
        --------
        scipy.sparse.coo_matrix.tanh : SciPy sparse equivalent function
        :obj:`numpy.tanh` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.tanh)

    def sqrt(self, out=None):
        """
        Return the positive square-root of an array, element-wise.

        See also
        --------
        scipy.sparse.coo_matrix.sqrt : SciPy sparse equivalent function
        :obj:`numpy.sqrt` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.sqrt)

    def ceil(self, out=None):
        """
        Return the ceiling of the input, element-wise.

        See also
        --------
        scipy.sparse.coo_matrix.ceil : SciPy sparse equivalent function
        :obj:`numpy.ceil` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.ceil)

    def floor(self, out=None):
        """
        Return the floor of the input, element-wise.

        See also
        --------
        scipy.sparse.coo_matrix.floor : SciPy sparse equivalent function
        :obj:`numpy.floor` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.floor)

    def round(self, decimals=0, out=None):
        """
        Evenly round to the given number of decimals.

        See also
        --------
        :obj:`numpy.round` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.round, decimals)

    def rint(self, out=None):
        """
        Round elements of the array to the nearest integer.

        See also
        --------
        scipy.sparse.coo_matrix.rint : SciPy sparse equivalent function
        :obj:`numpy.rint` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.rint)

    def conj(self, out=None):
        """
        Return the complex conjugate, element-wise.

        See also
        --------
        conjugate : Equivalent function
        scipy.sparse.coo_matrix.conj : SciPy sparse equivalent function
        :obj:`numpy.conj` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.conj)

    def conjugate(self, out=None):
        """
        Return the complex conjugate, element-wise.

        See also
        --------
        conj : Equivalent function
        scipy.sparse.coo_matrix.conjugate : SciPy sparse equivalent function
        :obj:`numpy.conj` : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.conjugate)

    def astype(self, dtype, out=None):
        """
        Copy of the array, cast to a specified type.

        See also
        --------
        scipy.sparse.coo_matrix.astype : SciPy sparse equivalent function
        numpy.ndarray.astype : NumPy equivalent ufunc.
        :obj:`COO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        assert out is None
        return self.elemwise(np.ndarray.astype, dtype)

    def maybe_densify(self, max_size=1000, min_density=0.25):
        """
        Converts this :obj:`COO` array to a :obj:`numpy.ndarray` if not too
        costly.

        Parameters
        ----------
        max_size : int
            Maximum number of elements in output
        min_density : float
            Minimum density of output

        Returns
        -------
        numpy.ndarray
            The dense array.

        Raises
        -------
        ValueError
            If the returned array would be too large.

        Examples
        --------
        Convert a small sparse array to a dense array.

        >>> s = COO.from_numpy(np.random.rand(2, 3, 4))
        >>> x = s.maybe_densify()
        >>> np.allclose(x, s.todense())
        True

        You can also specify the minimum allowed density or the maximum number
        of output elements. If both conditions are unmet, this method will throw
        an error.

        >>> x = np.zeros((5, 5), dtype=np.uint8)
        >>> x[2, 2] = 1
        >>> s = COO.from_numpy(x)
        >>> s.maybe_densify(max_size=5, min_density=0.25)
        Traceback (most recent call last):
            ...
        ValueError: Operation would require converting large sparse array to dense
        """
        if self.size <= max_size or self.density >= min_density:
            return self.todense()
        else:
            raise ValueError("Operation would require converting "
                             "large sparse array to dense")


def tensordot(a, b, axes=2):
    """
    Perform the equivalent of :obj:`numpy.tensordot`.

    Parameters
    ----------
    a, b : Union[COO, np.ndarray, scipy.sparse.spmatrix]
        The arrays to perform the :code:`tensordot` operation on.
    axes : tuple[Union[int, tuple[int], Union[int, tuple[int]], optional
        The axes to match when performing the sum.

    Returns
    -------
    Union[COO, numpy.ndarray]
        The result of the operation.

    See Also
    --------
    numpy.tensordot : NumPy equivalent function
    """
    # Much of this is stolen from numpy/core/numeric.py::tensordot
    # Please see license at https://github.com/numpy/numpy/blob/master/LICENSE.txt
    try:
        iter(axes)
    except TypeError:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    # a, b = asarray(a), asarray(b)  # <--- modified
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (-1, N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, -1)
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = _dot(at, bt)
    if isinstance(res, scipy.sparse.spmatrix):
        if res.nnz > reduce(operator.mul, res.shape) / 2:
            res = res.todense()
        else:
            res = COO.from_scipy_sparse(res)  # <--- modified
            res.has_duplicates = False
    if isinstance(res, np.matrix):
        res = np.asarray(res)
    return res.reshape(olda + oldb)


def dot(a, b):
    """
    Perform the equivalent of :obj:`numpy.dot` on two arrays.

    Parameters
    ----------
    a, b : Union[COO, np.ndarray, scipy.sparse.spmatrix]
        The arrays to perform the :code:`dot` operation on.

    Returns
    -------
    Union[COO, numpy.ndarray]
        The result of the operation.

    See Also
    --------
    numpy.dot : NumPy equivalent function.
    COO.dot : Equivalent function for COO objects.
    """
    if not hasattr(a, 'ndim') or not hasattr(b, 'ndim'):
        raise NotImplementedError(
            "Cannot perform dot product on types %s, %s" %
            (type(a), type(b)))
    return tensordot(a, b, axes=((a.ndim - 1,), (b.ndim - 2,)))


def _dot(a, b):
    if isinstance(a, COO):
        a.sum_duplicates()
    if isinstance(b, COO):
        b.sum_duplicates()
    if isinstance(b, COO) and not isinstance(a, COO):
        return _dot(b.T, a.T).T
    aa = a.tocsr()

    if isinstance(b, (COO, scipy.sparse.spmatrix)):
        b = b.tocsc()
    return aa.dot(b)


def _keepdims(original, new, axis):
    shape = list(original.shape)
    for ax in axis:
        shape[ax] = 1
    return new.reshape(shape)


def _mask(coords, idx, shape):
    if isinstance(idx, numbers.Integral):
        return coords == idx
    elif isinstance(idx, slice):
        step = idx.step if idx.step is not None else 1
        if step > 0:
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else shape
            return (coords >= start) & (coords < stop) & \
                   (coords % step == start % step)
        else:
            start = idx.start if idx.start is not None else (shape - 1)
            stop = idx.stop if idx.stop is not None else -1
            return (coords <= start) & (coords > stop) & \
                   (coords % step == start % step)
    elif isinstance(idx, Iterable):
        mask = np.zeros(len(coords), dtype=np.bool)
        for item in idx:
            mask |= _mask(coords, item, shape)
        return mask


def concatenate(arrays, axis=0):
    """
    Concatenate the input arrays along the given dimension.

    Parameters
    ----------
    arrays : Iterable[Union[COO, numpy.ndarray, scipy.sparse.spmatrix]]
        The input arrays to concatenate.
    axis : int, optional
        The axis along which to concatenate the input arrays. The default is zero.

    Returns
    -------
    COO
        The output concatenated array.

    See Also
    --------
    numpy.concatenate : NumPy equivalent function
    """
    arrays = [x if isinstance(x, COO) else COO(x) for x in arrays]
    if axis < 0:
        axis = axis + arrays[0].ndim
    assert all(x.shape[ax] == arrays[0].shape[ax]
               for x in arrays
               for ax in set(range(arrays[0].ndim)) - {axis})
    nnz = 0
    dim = sum(x.shape[axis] for x in arrays)
    shape = list(arrays[0].shape)
    shape[axis] = dim

    coords_dtype = np.min_scalar_type(max(shape) - 1) if len(shape) != 0 else np.uint8
    data = np.concatenate([x.data for x in arrays])
    coords = np.concatenate([x.coords for x in arrays], axis=1).astype(coords_dtype)

    dim = 0
    for x in arrays:
        if dim:
            coords[axis, nnz:x.nnz + nnz] += dim
        dim += x.shape[axis]
        nnz += x.nnz

    has_duplicates = any(x.has_duplicates for x in arrays)

    return COO(coords, data, shape=shape, has_duplicates=has_duplicates,
               sorted=(axis == 0) and all(a.sorted for a in arrays))


def stack(arrays, axis=0):
    """
    Stack the input arrays along the given dimension.

    Parameters
    ----------
    arrays : Iterable[Union[COO, numpy.ndarray, scipy.sparse.spmatrix]]
        The input arrays to stack.
    axis : int, optional
        The axis along which to stack the input arrays.

    Returns
    -------
    COO
        The output stacked array.

    See Also
    --------
    numpy.stack : NumPy equivalent function
    """
    assert len(set(x.shape for x in arrays)) == 1
    arrays = [x if isinstance(x, COO) else COO(x) for x in arrays]
    if axis < 0:
        axis = axis + arrays[0].ndim + 1
    data = np.concatenate([x.data for x in arrays])
    coords = np.concatenate([x.coords for x in arrays], axis=1)
    shape = list(arrays[0].shape)
    shape.insert(axis, len(arrays))

    coords_dtype = np.min_scalar_type(max(shape) - 1) if len(shape) != 0 else np.uint8

    nnz = 0
    dim = 0
    new = np.empty(shape=(coords.shape[1],), dtype=coords_dtype)
    for x in arrays:
        new[nnz:x.nnz + nnz] = dim
        dim += 1
        nnz += x.nnz

    has_duplicates = any(x.has_duplicates for x in arrays)
    coords = [coords[i].astype(coords_dtype) for i in range(coords.shape[0])]
    coords.insert(axis, new)
    coords = np.stack(coords, axis=0)

    return COO(coords, data, shape=shape, has_duplicates=has_duplicates,
               sorted=(axis == 0) and all(a.sorted for a in arrays))


def triu(x, k=0):
    """
    Returns an array with all elements below the k-th diagonal set to zero.

    Parameters
    ----------
    x : COO
        The input array.
    k : int, optional
        The diagonal below which elements are set to zero. The default is
        zero, which corresponds to the main diagonal.

    Returns
    -------
    COO
        The output upper-triangular matrix.

    See Also
    --------
    numpy.triu : NumPy equivalent function
    """
    if not x.ndim >= 2:
        raise NotImplementedError('sparse.triu is not implemented for scalars or 1-D arrays.')

    mask = x.coords[-2] + k <= x.coords[-1]

    coords = x.coords[:, mask]
    data = x.data[mask]

    return COO(coords, data, x.shape, x.has_duplicates, x.sorted)


def tril(x, k=0):
    """
    Returns an array with all elements above the k-th diagonal set to zero.

    Parameters
    ----------
    x : COO
        The input array.
    k : int, optional
        The diagonal above which elements are set to zero. The default is
        zero, which corresponds to the main diagonal.

    Returns
    -------
    COO
        The output lower-triangular matrix.

    See Also
    --------
    numpy.tril : NumPy equivalent function
    """
    if not x.ndim >= 2:
        raise NotImplementedError('sparse.tril is not implemented for scalars or 1-D arrays.')

    mask = x.coords[-2] + k >= x.coords[-1]

    coords = x.coords[:, mask]
    data = x.data[mask]

    return COO(coords, data, x.shape, x.has_duplicates, x.sorted)


# (c) Paul Panzer
# Taken from https://stackoverflow.com/a/47833496/774273
# License: https://creativecommons.org/licenses/by-sa/3.0/
def _match_arrays(a, b):
    """
    Finds all indexes into a and b such that a[i] = b[j]. The outputs are sorted
    in lexographical order.

    Parameters
    ----------
    a, b : np.ndarray
        The input 1-D arrays to match. If matching of multiple fields is
        needed, use np.recarrays. These two arrays must be sorted.

    Returns
    -------
    a_idx, b_idx : np.ndarray
        The output indices of every possible pair of matching elements.
    """
    if len(a) == 0 or len(b) == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)
    asw = np.r_[0, 1 + np.flatnonzero(a[:-1] != a[1:]), len(a)]
    bsw = np.r_[0, 1 + np.flatnonzero(b[:-1] != b[1:]), len(b)]
    al, bl = np.diff(asw), np.diff(bsw)
    na = len(al)
    asw, bsw = asw, bsw
    abunq = np.r_[a[asw[:-1]], b[bsw[:-1]]]
    m = np.argsort(abunq, kind='mergesort')
    mv = abunq[m]
    midx = np.flatnonzero(mv[:-1] == mv[1:])
    ai, bi = m[midx], m[midx + 1] - na
    aic = np.r_[0, np.cumsum(al[ai])]
    a_idx = np.ones((aic[-1],), dtype=np.int_)
    a_idx[aic[:-1]] = asw[ai]
    a_idx[aic[1:-1]] -= asw[ai[:-1]] + al[ai[:-1]] - 1
    a_idx = np.repeat(np.cumsum(a_idx), np.repeat(bl[bi], al[ai]))
    bi = np.repeat(bi, al[ai])
    bic = np.r_[0, np.cumsum(bl[bi])]
    b_idx = np.ones((bic[-1],), dtype=np.int_)
    b_idx[bic[:-1]] = bsw[bi]
    b_idx[bic[1:-1]] -= bsw[bi[:-1]] + bl[bi[:-1]] - 1
    b_idx = np.cumsum(b_idx)
    return a_idx, b_idx


def _grouped_reduce(x, groups, method, **kwargs):
    """
    Performs a :code:`ufunc` grouped reduce.

    Parameters
    ----------
    x : np.ndarray
        The data to reduce.
    groups : np.ndarray
        The groups the data belongs to. The groups must be
        contiguous.
    method : np.ufunc
        The :code:`ufunc` to use to perform the reduction.
    kwargs : dict
        The kwargs to pass to the :code:`ufunc`'s :code:`reduceat`
        function.

    Returns
    -------
    result : np.ndarray
        The result of the grouped reduce operation.
    inv_idx : np.ndarray
        The index of the first element where each group is found.
    counts : np.ndarray
        The number of elements in each group.
    """
    # Partial credit to @shoyer
    # Ref: https://gist.github.com/shoyer/f538ac78ae904c936844
    flag = np.concatenate(([True] if len(x) != 0 else [], groups[1:] != groups[:-1]))
    inv_idx = np.flatnonzero(flag)
    result = method.reduceat(x, inv_idx, **kwargs)
    counts = np.diff(np.concatenate((inv_idx, [len(x)])))
    return result, inv_idx, counts


def _elemwise_binary(func, self, other, *args, **kwargs):
    check = kwargs.pop('check', True)
    self_zero = _zero_of_dtype(self.dtype)
    other_zero = _zero_of_dtype(other.dtype)

    func_zero = _zero_of_dtype(func(self_zero, other_zero, *args, **kwargs).dtype)
    if check and func(self_zero, other_zero, *args, **kwargs) != func_zero:
        raise ValueError("Performing this operation would produce "
                         "a dense result: %s" % str(func))

    if not isinstance(self, COO):
        if not check or np.array_equiv(func(self, other_zero, *args, **kwargs), func_zero):
            return _elemwise_binary_self_dense(func, self, other, *args, **kwargs)
        else:
            raise ValueError("Performing this operation would produce "
                             "a dense result: %s" % str(func))

    if not isinstance(other, COO):
        if not check or np.array_equiv(func(self_zero, other, *args, **kwargs), func_zero):
            temp_func = _reverse_self_other(func)
            return _elemwise_binary_self_dense(temp_func, other, self, *args, **kwargs)
        else:
            raise ValueError("Performing this operation would produce "
                             "a dense result: %s" % str(func))

    self_shape, other_shape = self.shape, other.shape

    result_shape = _get_broadcast_shape(self_shape, other_shape)
    self_params = _get_broadcast_parameters(self.shape, result_shape)
    other_params = _get_broadcast_parameters(other.shape, result_shape)
    combined_params = [p1 and p2 for p1, p2 in zip(self_params, other_params)]
    self_reduce_params = combined_params[-self.ndim:]
    other_reduce_params = combined_params[-other.ndim:]

    self.sum_duplicates()  # TODO: document side-effect or make copy
    other.sum_duplicates()  # TODO: document side-effect or make copy

    self_coords = self.coords
    self_data = self.data

    self_reduced_coords, self_reduced_shape = \
        _get_reduced_coords(self_coords, self_shape,
                            self_reduce_params)
    self_reduced_linear = _linear_loc(self_reduced_coords, self_reduced_shape)
    i = np.argsort(self_reduced_linear)
    self_reduced_linear = self_reduced_linear[i]
    self_coords = self_coords[:, i]
    self_data = self_data[i]

    # Store coords
    other_coords = other.coords
    other_data = other.data

    other_reduced_coords, other_reduced_shape = \
        _get_reduced_coords(other_coords, other_shape,
                            other_reduce_params)
    other_reduced_linear = _linear_loc(other_reduced_coords, other_reduced_shape)
    i = np.argsort(other_reduced_linear)
    other_reduced_linear = other_reduced_linear[i]
    other_coords = other_coords[:, i]
    other_data = other_data[i]

    # Find matches between self.coords and other.coords
    matched_self, matched_other = _match_arrays(self_reduced_linear,
                                                other_reduced_linear)

    # Start with an empty list. This may reduce computation in many cases.
    data_list = []
    coords_list = []

    # Add the matched part.
    matched_coords = _get_matching_coords(self_coords[:, matched_self],
                                          other_coords[:, matched_other],
                                          self_shape, other_shape)

    data_list.append(func(self_data[matched_self],
                          other_data[matched_other],
                          *args, **kwargs))
    coords_list.append(matched_coords)

    self_func = func(self_data, other_zero, *args, **kwargs)
    # Add unmatched parts as necessary.
    if (self_func != func_zero).any():
        self_unmatched_coords, self_unmatched_func = \
            _get_unmatched_coords_data(self_coords, self_func, self_shape,
                                       result_shape, matched_self,
                                       matched_coords)

        data_list.extend(self_unmatched_func)
        coords_list.extend(self_unmatched_coords)

    other_func = func(self_zero, other_data, *args, **kwargs)

    if (other_func != func_zero).any():
        other_unmatched_coords, other_unmatched_func = \
            _get_unmatched_coords_data(other_coords, other_func, other_shape,
                                       result_shape, matched_other,
                                       matched_coords)

        coords_list.extend(other_unmatched_coords)
        data_list.extend(other_unmatched_func)

    # Concatenate matches and mismatches
    data = np.concatenate(data_list) if len(data_list) else np.empty((0,), dtype=self.dtype)
    coords = np.concatenate(coords_list, axis=1) if len(coords_list) else \
        np.empty((0, len(result_shape)), dtype=self.coords.dtype)

    nonzero = data != func_zero
    data = data[nonzero]
    coords = coords[:, nonzero]

    return COO(coords, data, shape=result_shape, has_duplicates=False)


def _elemwise_binary_self_dense(func, self, other, *args, **kwargs):
    assert isinstance(self, np.ndarray)
    assert isinstance(other, COO)

    result_shape = _get_broadcast_shape(self.shape, other.shape)

    if result_shape != other.shape:
        other = other.broadcast_to(result_shape)

    self = np.broadcast_to(self, result_shape)

    self_coords = tuple([other.coords[i, :] for i in range(other.ndim)])

    self_data = self[self_coords]

    func_data = func(self_data, other.data, *args, **kwargs)
    mask = func_data != 0
    func_data = func_data[mask]
    func_coords = other.coords[:, mask]

    return COO(func_coords, func_data, shape=result_shape,
               has_duplicates=other.has_duplicates,
               sorted=other.sorted)


def _reverse_self_other(func):
    def wrapper(*args, **kwargs):
        return func(args[1], args[0], *args[2:], **kwargs)

    return wrapper


def _get_unmatched_coords_data(coords, data, shape, result_shape, matched_idx,
                               matched_coords):
    """
    Get the unmatched coordinates and data - both those that are unmatched with
    any point of the other data as well as those which are added because of
    broadcasting.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates to get the unmatched coordinates from.
    data : np.ndarray
        The data corresponding to these coordinates.
    shape : tuple[int]
        The shape corresponding to these coordinates.
    result_shape : tuple[int]
        The result broadcasting shape.
    matched_idx : np.ndarray
        The indices into the coords array where it matches with the other array.
    matched_coords : np.ndarray
        The overall coordinates that match from both arrays.

    Returns
    -------
    coords_list : list[np.ndarray]
        The list of unmatched/broadcasting coordinates.
    data_list : list[np.ndarray]
        The data corresponding to the coordinates.
    """
    params = _get_broadcast_parameters(shape, result_shape)
    matched = np.zeros(len(data), dtype=np.bool)
    matched[matched_idx] = True
    unmatched = ~matched
    data_zero = _zero_of_dtype(data.dtype)
    nonzero = data != data_zero

    unmatched &= nonzero
    matched &= nonzero

    coords_list = []
    data_list = []

    unmatched_coords, unmatched_data = \
        _get_expanded_coords_data(coords[:, unmatched],
                                  data[unmatched],
                                  params,
                                  result_shape)

    coords_list.append(unmatched_coords)
    data_list.append(unmatched_data)

    if shape != result_shape:
        broadcast_coords, broadcast_data = \
            _get_broadcast_coords_data(coords[:, matched],
                                       matched_coords,
                                       data[matched],
                                       params,
                                       result_shape)

        coords_list.append(broadcast_coords)
        data_list.append(broadcast_data)

    return coords_list, data_list


def _get_broadcast_shape(shape1, shape2, is_result=False):
    """
    Get the overall broadcasted shape.

    Parameters
    ----------
    shape1, shape2 : tuple[int]
        The input shapes to broadcast together.
    is_result : bool
        Whether or not shape2 is also the result shape.

    Returns
    -------
    result_shape : tuple[int]
        The overall shape of the result.

    Raises
    ------
    ValueError
        If the two shapes cannot be broadcast together.
    """
    # https://stackoverflow.com/a/47244284/774273
    if not all((l1 == l2) or (l1 == 1) or ((l2 == 1) and not is_result) for l1, l2 in
               zip(shape1[::-1], shape2[::-1])):
        raise ValueError('operands could not be broadcast together with shapes %s, %s' %
                         (shape1, shape2))

    result_shape = tuple(max(l1, l2) for l1, l2 in
                         zip_longest(shape1[::-1], shape2[::-1], fillvalue=1))[::-1]

    return result_shape


def _get_broadcast_parameters(shape, broadcast_shape):
    """
    Get the broadcast parameters.

    Parameters
    ----------
    shape : tuple[int]
        The input shape.
    broadcast_shape
        The shape to broadcast to.

    Returns
    -------
    params : list
        A list containing None if the dimension isn't in the original array, False if
        it needs to be broadcast, and True if it doesn't.
    """
    params = [None if l1 is None else l1 == l2 for l1, l2
              in zip_longest(shape[::-1], broadcast_shape[::-1], fillvalue=None)][::-1]

    return params


def _get_reduced_coords(coords, shape, params):
    """
    Gets only those dimensions of the coordinates that don't need to be broadcast.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates to reduce.
    params : list
        The params from which to check which dimensions to get.

    Returns
    -------
    reduced_coords : np.ndarray
        The reduced coordinates.
    """
    reduced_params = [bool(param) for param in params]
    reduced_shape = tuple(l for l, p in zip(shape, params) if p)

    return coords[reduced_params], reduced_shape


def _get_expanded_coords_data(coords, data, params, broadcast_shape):
    """
    Expand coordinates/data to broadcast_shape. Does most of the heavy lifting for broadcast_to.
    Produces sorted output for sorted inputs.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates to expand.
    data : np.ndarray
        The data corresponding to the coordinates.
    params : list
        The broadcast parameters.
    broadcast_shape : tuple[int]
        The shape to broadcast to.

    Returns
    -------
    expanded_coords : np.ndarray
        List of 1-D arrays. Each item in the list has one dimension of coordinates.
    expanded_data : np.ndarray
        The data corresponding to expanded_coords.
    """
    first_dim = -1
    expand_shapes = []
    for d, p, l in zip(range(len(broadcast_shape)), params, broadcast_shape):
        if p and first_dim == -1:
            expand_shapes.append(coords.shape[1])
            first_dim = d

        if not p:
            expand_shapes.append(l)

    all_idx = _cartesian_product(*(np.arange(d, dtype=np.min_scalar_type(d - 1)) for d in expand_shapes))
    dt = np.result_type(*(np.min_scalar_type(l - 1) for l in broadcast_shape))

    false_dim = 0
    dim = 0

    expanded_coords = np.empty((len(broadcast_shape), all_idx.shape[1]), dtype=dt)
    expanded_data = data[all_idx[first_dim]]

    for d, p, l in zip(range(len(broadcast_shape)), params, broadcast_shape):
        if p:
            expanded_coords[d] = coords[dim, all_idx[first_dim]]
        else:
            expanded_coords[d] = all_idx[false_dim + (d > first_dim)]
            false_dim += 1

        if p is not None:
            dim += 1

    return np.asarray(expanded_coords), np.asarray(expanded_data)


# (c) senderle
# Taken from https://stackoverflow.com/a/11146645/774273
# License: https://creativecommons.org/licenses/by-sa/3.0/
def _cartesian_product(*arrays):
    """
    Get the cartesian product of a number of arrays.

    Parameters
    ----------
    arrays : Tuple[np.ndarray]
        The arrays to get a cartesian product of. Always sorted with respect
        to the original array.

    Returns
    -------
    out : np.ndarray
        The overall cartesian product of all the input arrays.
    """
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)
    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows)


def _elemwise_unary(func, self, *args, **kwargs):
    check = kwargs.pop('check', True)
    data_zero = _zero_of_dtype(self.dtype)
    func_zero = _zero_of_dtype(func(data_zero, *args, **kwargs).dtype)
    if check and func(data_zero, *args, **kwargs) != func_zero:
        raise ValueError("Performing this operation would produce "
                         "a dense result: %s" % str(func))

    data_func = func(self.data, *args, **kwargs)
    nonzero = data_func != func_zero

    return COO(self.coords[:, nonzero], data_func[nonzero],
               shape=self.shape,
               has_duplicates=self.has_duplicates,
               sorted=self.sorted)


def _get_matching_coords(coords1, coords2, shape1, shape2):
    """
    Takes in the matching coordinates in both dimensions (only those dimensions that
    don't need to be broadcast in both arrays and returns the coordinates that will
    overlap in the output array, i.e., the coordinates for which both broadcast arrays
    will be nonzero.

    Parameters
    ----------
    coords1, coords2 : np.ndarray
    shape1, shape2 : tuple[int]

    Returns
    -------
    matching_coords : np.ndarray
        The coordinates of the output array for which both inputs will be nonzero.
    """
    result_shape = _get_broadcast_shape(shape1, shape2)
    params1 = _get_broadcast_parameters(shape1, result_shape)
    params2 = _get_broadcast_parameters(shape2, result_shape)

    matching_coords = []
    dim1 = 0
    dim2 = 0

    for p1, p2 in zip(params1, params2):
        if p1:
            matching_coords.append(coords1[dim1])
        else:
            matching_coords.append(coords2[dim2])

        if p1 is not None:
            dim1 += 1

        if p2 is not None:
            dim2 += 1

    return np.asarray(matching_coords)


def _get_broadcast_coords_data(coords, matched_coords, data, params, broadcast_shape):
    """
    Get data that matched in the reduced coordinates but still had a partial overlap because of
    the broadcast, i.e., it didn't match in one of the other dimensions.

    Parameters
    ----------
    coords : np.ndarray
        The list of coordinates of the required array. Must be sorted.
    matched_coords : np.ndarray
        The list of coordinates that match. Must be sorted.
    data : np.ndarray
        The data corresponding to coords.
    params : list
        The broadcast parameters.
    broadcast_shape : tuple[int]
        The shape to get the broadcast coordinates.

    Returns
    -------
    broadcast_coords : np.ndarray
        The broadcasted coordinates. Is sorted.
    broadcasted_data : np.ndarray
        The data corresponding to those coordinates.
    """
    full_coords, full_data = _get_expanded_coords_data(coords, data, params, broadcast_shape)
    linear_full_coords = _linear_loc(full_coords, broadcast_shape)
    linear_matched_coords = _linear_loc(matched_coords, broadcast_shape)

    overlapping_coords, _ = _match_arrays(linear_full_coords, linear_matched_coords)
    mask = np.ones(full_coords.shape[1], dtype=np.bool)
    mask[overlapping_coords] = False

    return full_coords[:, mask], full_data[mask]


def _linear_loc(coords, shape, signed=False):
    n = reduce(operator.mul, shape, 1)
    if signed:
        n = -n
    dtype = np.min_scalar_type(n)
    out = np.zeros(coords.shape[1], dtype=dtype)
    tmp = np.zeros(coords.shape[1], dtype=dtype)
    strides = 1
    for i, d in enumerate(shape[::-1]):
        # out += self.coords[-(i + 1), :].astype(dtype) * strides
        np.multiply(coords[-(i + 1), :], strides, out=tmp, dtype=dtype)
        np.add(tmp, out, out=out)
        strides *= d
    return out
