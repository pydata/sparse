import numbers
from collections import Iterable, defaultdict, deque

import numba
import numpy as np
import scipy.sparse
from numpy.lib.mixins import NDArrayOperatorsMixin

from .common import dot
from .umath import elemwise, broadcast_to
from ..compatibility import int, range, zip_longest
from ..slicing import normalize_index
from ..sparse_array import SparseArray
from ..utils import _zero_of_dtype


class COO(SparseArray, NDArrayOperatorsMixin):
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
    shape : tuple[int] (COO.ndim,)
        The shape of the array.
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
            from ..dok import DOK

            if isinstance(coords, COO):
                self._make_shallow_copy_of(coords)
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
                self._make_shallow_copy_of(result)
                return

            if isinstance(coords, scipy.sparse.spmatrix):
                result = COO.from_scipy_sparse(coords)
                self._make_shallow_copy_of(result)
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

        super(COO, self).__init__(shape)
        if self.shape:
            dtype = np.min_scalar_type(max(max(self.shape) - 1, 0))
        else:
            dtype = np.uint8
        self.coords = self.coords.astype(dtype)
        assert not self.shape or len(data) == self.coords.shape[1]
        self.has_duplicates = has_duplicates
        self.sorted = sorted

    def _make_shallow_copy_of(self, other):
        self.coords = other.coords
        self.data = other.data
        self.has_duplicates = other.has_duplicates
        self.sorted = other.sorted
        super(COO, self).__init__(other.shape)

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

    def __sizeof__(self):
        return self.nbytes

    def __getitem__(self, index):
        self.sum_duplicates()

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

        if len(index) != 0 and all(not isinstance(ind, Iterable) and ind == slice(0, dim, 1)
                                   for ind, dim in zip_longest(index, self.shape)):
            return self
        mask = _mask(self.coords, index, self.shape)

        if isinstance(mask, slice):
            n = len(range(mask.start, mask.stop, mask.step))
        else:
            n = len(mask)

        coords = []
        shape = []
        i = 0
        for ind in index:
            if isinstance(ind, numbers.Integral):
                i += 1
                continue
            elif isinstance(ind, slice):
                shape.append(len(range(ind.start, ind.stop, ind.step)))
                dt = np.min_scalar_type(min(-(dim - 1) if dim != 0 else -1 for dim in shape))
                coords.append((self.coords[i, mask].astype(dt) - ind.start) // ind.step)
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
                coords = np.empty((0, n), dtype=np.uint8)
            else:
                if n != 0:
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

    def reduce(self, method, axis=(0,), keepdims=False, **kwargs):
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
        COO.nanreduce : Similar method with ``NaN`` skipping functionality.

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

        By default, this reduces the array by only the first axis.

        >>> s.reduce(np.add)
        <COO: shape=(5,), dtype=int64, nnz=5, sorted=True, duplicates=False>
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

        axis = tuple(a if a >= 0 else a + self.ndim for a in axis)

        if set(axis) == set(range(self.ndim)):
            result = method.reduce(self.data, **kwargs)
            if self.nnz != self.size:
                result = method(result, _zero_of_dtype(self.dtype)[()], **kwargs)
        else:
            axis = tuple(axis)
            neg_axis = tuple(ax for ax in range(self.ndim) if ax not in set(axis))

            a = self.transpose(neg_axis + axis)
            a = a.reshape((np.prod([self.shape[d] for d in neg_axis]),
                           np.prod([self.shape[d] for d in axis])))
            a.sort_indices()

            result, inv_idx, counts = _grouped_reduce(a.data, a.coords[0], method, **kwargs)
            missing_counts = counts != a.shape[1]
            result[missing_counts] = method(result[missing_counts],
                                            _zero_of_dtype(self.dtype), **kwargs)
            coords = a.coords[0:1, inv_idx]

            # Filter out zeros
            mask = result != _zero_of_dtype(result.dtype)
            coords = coords[:, mask]
            result = result[mask]

            a = COO(coords, result, shape=(a.shape[0],),
                    has_duplicates=False, sorted=True)

            a = a.reshape(tuple(self.shape[d] for d in neg_axis))
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
        :obj:`nansum` : Function with ``NaN`` skipping.

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
        :obj:`nanmax` : Function with ``NaN`` skipping.

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
        :obj:`nanmin` : Function with ``NaN`` skipping.

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
        :obj:`nanprod` : Function with ``NaN`` skipping.

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
        out = kwargs.pop('out', None)
        if out is not None:
            return NotImplemented

        if method == '__call__':
            return elemwise(ufunc, *inputs, **kwargs)
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
        from .common import linear_loc

        return linear_loc(self.coords, self.shape, signed)

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
        return broadcast_to(self, shape)

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
        return elemwise(np.round, self, decimals=decimals)

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
        return elemwise(np.ndarray.astype, self, dtype=dtype)

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


def _keepdims(original, new, axis):
    shape = list(original.shape)
    for ax in axis:
        shape[ax] = 1
    return new.reshape(shape)


def _mask(coords, indices, shape):
    indices = _prune_indices(indices, shape)

    ind_ar = np.empty((len(indices), 3), dtype=np.intp)

    for i, idx in enumerate(indices):
        if isinstance(idx, slice):
            ind_ar[i] = [idx.start, idx.stop, idx.step]
        else:
            ind_ar[i] = [idx, idx + 1, 1]

    mask, is_slice = _get_mask(coords, ind_ar)

    if is_slice:
        return slice(mask[0], mask[1], 1)
    else:
        return mask


def _prune_indices(indices, shape):
    """
    Gets rid of the indices that do not contribute to the
    overall mask, e.g. None and full slices.

    Parameters
    ----------
    indices : tuple
        The indices to the array.
    shape : tuple[int]
        The shape of the array.

    Returns
    -------
    indices : tuple
        The filtered indices.
    """
    indices = [idx for idx in indices if idx is not None]
    i = 0
    for idx, l in zip(indices[::-1], shape[::-1]):
        if not isinstance(idx, slice):
            break

        if idx.start == 0 and idx.stop == l and idx.step == 1:
            i += 1
            continue

        if idx.start == l - 1 and idx.stop == -1 and idx.step == -1:
            i += 1
            continue

        break
    if i != 0:
        indices = indices[:-i]
    return indices


@numba.jit(nopython=True)
def _get_mask(coords, indices):  # pragma: no cover
    """
    Gets the start-end pairs for the mask.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates of the array.
    indices : np.ndarray
        The indices in the form of slices.

    Returns
    -------
    starts, stops : np.ndarray
        The starts and stops in the mask.
    """
    starts = [0]
    stops = [coords.shape[1]]
    n_matches = coords.shape[1]

    i = 0
    while i < len(indices):
        n_current_slices = _get_slice_len(indices[i]) * len(starts) + 2
        if 10.0 * n_current_slices * np.log(n_current_slices / len(starts)) > n_matches:
            break

        starts, stops, n_matches = _get_mask_pairs_inner(starts, stops, coords[i], indices[i])

        i += 1

    starts, stops = _join_adjacent_pairs(starts, stops)

    if i == len(indices) and len(starts) == 1:
        return np.array([starts[0], stops[0]]), True

    mask = _cat_pairs(starts, stops)

    while i < len(indices):
        mask = _filter_mask(coords[i], mask, indices[i])
        i += 1

    return np.array(mask, dtype=np.intp), False


@numba.jit(nopython=True)
def _get_slice_len(idx):
    """
    Get the number of elements in a slice.

    Parameters
    ----------
    idx : np.ndarray
        A (3,) shaped array containing start, stop, step

    Returns
    -------
    n : int
        The length of the slice.
    """
    start, stop, step = idx[0], idx[1], idx[2]

    if step > 0:
        return (stop - start + step - 1) // step
    else:
        return (start - stop - step - 1) // (-step)


@numba.jit(nopython=True)
def _get_mask_pairs_inner(starts_old, stops_old, c, idx):  # pragma: no cover
    """
    Gets the starts/stops for a an index given the starts/stops
    from the previous index.

    Parameters
    ----------
    starts_old, stops_old : list[int]
        The starts and stops from the previous index.
    c : np.ndarray
        The coords for this index's dimension.
    idx : np.ndarray
        The index in the form of a slice.
        idx[0], idx[1], idx[2] = start, stop, step

    Returns
    -------
    starts, stops: np.ndarray
        The starts and stops after applying the current index.
    """
    starts = []
    stops = []
    n_matches = 0

    for j in range(len(starts_old)):
        for p_match in range(idx[0], idx[1], idx[2]):
            start = np.searchsorted(c[starts_old[j]:stops_old[j]], p_match) + starts_old[j]
            stop = np.searchsorted(c[starts_old[j]:stops_old[j]], p_match + 1) + starts_old[j]

            if start != stop:
                starts.append(start)
                stops.append(stop)
                n_matches += stop - start

    return starts, stops, n_matches


@numba.jit(nopython=True)
def _join_adjacent_pairs(starts_old, stops_old):  # pragma: no cover
    """
    Joins adjacent pairs into one. For example, 2-5 and 5-7
    will reduce to 2-7 (a single pair). This may help in
    returning a slice in the end which could be faster.

    Parameters
    ----------
    starts_old, stops_old : list[int]
        The input starts and stops

    Returns
    -------
    starts, stops : list[int]
        The reduced starts and stops.
    """
    if len(starts_old) <= 1:
        return starts_old, stops_old

    starts = [starts_old[0]]
    stops = []

    for i in range(1, len(starts_old)):
        if starts_old[i] != stops_old[i - 1]:
            starts.append(starts_old[i])
            stops.append(stops_old[i - 1])

    stops.append(stops_old[-1])

    return starts, stops


@numba.jit(nopython=True)
def _cat_pairs(starts, stops):  # pragma: no cover
    """
    Converts all the pairs into a single integer mask.

    Parameters
    ----------
    starts, stops : np.ndarray
        The starts and stops to convert into an array.

    Returns
    -------
    mask : list
        The output integer mask.
    """
    mask = []
    for i in range(len(starts)):
        for match in range(starts[i], stops[i]):
            mask.append(match)

    return mask


@numba.jit(nopython=True)
def _filter_mask(c, mask_old, idx):
    """
    Filters the mask given coordinates and a slice.

    Parameters
    ----------
    c : np.ndarray
        The coordinates for the current axis.
    mask_old : list
        The old mask.
    idx : np.ndarray
        The slice to filter by.

    Returns
    -------
    mask : list
        The filtered mask.
    """
    mask = []

    for i in mask_old:
        elem = c[i]
        if (elem - idx[0]) % idx[2] == 0 and \
                ((idx[2] > 0 and idx[0] <= elem < idx[1])
                 or (idx[2] < 0 and idx[0] >= elem > idx[1])):
            mask.append(i)

    return mask


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
