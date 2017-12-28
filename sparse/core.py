from __future__ import absolute_import, division, print_function

from collections import Iterable, defaultdict, deque
from functools import reduce
import numbers
import operator

import numpy as np
import scipy.sparse

# zip_longest with Python 2/3 compat
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

try:  # Windows compatibility
    int = long
except NameError:
    pass


class COO(object):
    """ A Sparse Multidimensional Array

    This is stored in COO format.  It depends on NumPy and Scipy.sparse for
    computation, but supports arrays of arbitrary dimension.

    Parameters
    ----------
    coords: np.ndarray (ndim, nnz)
        An array holding the index locations of every value
        Should have shape (number of dimensions, number of non-zeros)
    data: np.array (nnz,)
        An array of Values
    shape: tuple (ndim,), optional
        The shape of the array

    Examples
    --------
    >>> x = np.eye(4)
    >>> x[2, 3] = 5
    >>> s = COO(x)
    >>> s
    <COO: shape=(4, 4), dtype=float64, nnz=5, sorted=True, duplicates=False>
    >>> s.data
    array([ 1.,  1.,  1.,  5.,  1.])
    >>> s.coords
    array([[0, 1, 2, 2, 3],
           [0, 1, 2, 3, 3]], dtype=uint8)

    >>> s.dot(s.T).sum(axis=0).todense()
    array([  1.,   1.,  31.,   6.])

    Make a sparse array by passing in an array of coordinates and an array of
    values.

    >>> coords = [[0, 0, 0, 1, 1],
    ...           [0, 1, 2, 0, 3],
    ...           [0, 3, 2, 0, 1]]
    >>> data = [1, 2, 3, 4, 5]
    >>> y = COO(coords, data, shape=(3, 4, 5))
    >>> y
    <COO: shape=(3, 4, 5), dtype=int64, nnz=5, sorted=False, duplicates=True>
    >>> tensordot(s, y, axes=(0, 1))
    <COO: shape=(4, 3, 5), dtype=float64, nnz=6, sorted=False, duplicates=False>

    Following scipy.sparse conventions you can also pass these as a tuple with
    rows and columns

    >>> rows = [0, 1, 2, 3, 4]
    >>> cols = [0, 0, 0, 1, 1]
    >>> data = [10, 20, 30, 40, 50]
    >>> z = COO((data, (rows, cols)))
    >>> z.todense()
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
    >>> COO(L).todense()
    array([[4, 0],
           [0, 2]])

    See Also
    --------
    COO.from_numpy
    COO.from_scipy_sparse
    """
    __array_priority__ = 12

    def __init__(self, coords, data=None, shape=None, has_duplicates=True,
                 sorted=False, cache=False):
        self._cache = None
        if cache:
            self.enable_caching()
        if data is None:
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

        if shape and not np.prod(self.coords.shape):
            self.coords = np.zeros((len(shape), 0), dtype=np.uint64)

        if shape is None:
            if self.coords.nbytes:
                shape = tuple((self.coords.max(axis=1) + 1).tolist())
            else:
                shape = ()

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
        >>> x.enable_caching()  # doctest: +SKIP
        >>> csr1 = x.transpose((2, 0, 1)).reshape((100, 120)).tocsr()  # doctest: +SKIP
        >>> csr2 = x.transpose((2, 0, 1)).reshape((100, 120)).tocsr()  # doctest: +SKIP
        >>> csr1 is csr2  # doctest: +SKIP
        True
        """
        self._cache = defaultdict(lambda: deque(maxlen=3))
        return self

    @classmethod
    def from_numpy(cls, x):
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
        x = scipy.sparse.coo_matrix(x)
        coords = np.empty((2, x.nnz), dtype=x.row.dtype)
        coords[0, :] = x.row
        coords[1, :] = x.col
        return COO(coords, x.data, shape=x.shape,
                   has_duplicates=not x.has_canonical_format,
                   sorted=x.has_canonical_format)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nnz(self):
        return self.coords.shape[1]

    @property
    def nbytes(self):
        return self.data.nbytes + self.coords.nbytes

    def __sizeof__(self):
        return self.nbytes

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) - index.count(None) - index.count(Ellipsis) > self.ndim:
            raise IndexError("too many indices for array")
        if index.count(Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        index = tuple(ind + self.shape[i]  # this fails for newaxis slices
                      if isinstance(ind, numbers.Integral) and ind < 0
                      else ind
                      for i, ind in enumerate(index))
        if any(ind is Ellipsis for ind in index):
            loc = index.index(Ellipsis)
            n = self.ndim - (len(index) - 1 - index.count(None))
            index = index[:loc] + (slice(None, None),) * n + index[loc + 1:]
        if all(ind == slice(None) for ind in index):
            return self
        mask = np.ones(self.nnz, dtype=bool)
        for i, ind in enumerate([i for i in index if i is not None]):
            if ind == slice(None, None):
                continue
            mask &= _mask(self.coords[i], ind)

        n = mask.sum()
        coords = []
        shape = []
        i = 0
        for ind in index:
            if isinstance(ind, numbers.Integral):
                i += 1
                continue
            elif isinstance(ind, slice):
                start = ind.start or 0
                stop = ind.stop if ind.stop is not None else self.shape[i]
                shape.append(min(stop, self.shape[i]) - start)
                coords.append(self.coords[i][mask] - start)
                i += 1
            elif isinstance(ind, list):
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
            coords = np.empty((0, np.sum(mask)), dtype=np.uint8)
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

    def reduce(self, method, axis=None, keepdims=False, dtype=None):
        # Needed for more esoteric reductions like product.
        self.sum_duplicates()

        if axis is None:
            axis = tuple(range(self.ndim))

        kwargs = {}
        if dtype:
            kwargs['dtype'] = dtype

        if not isinstance(axis, tuple):
            axis = (axis,)

        if set(axis) == set(range(self.ndim)):
            result = method.reduce(self.data, **kwargs)
        else:
            axis = tuple(axis)
            neg_axis = tuple(ax for ax in range(self.ndim) if ax not in axis)

            a = self.transpose(neg_axis + axis)
            a = a.reshape((np.prod([self.shape[d] for d in neg_axis]),
                           np.prod([self.shape[d] for d in axis])))
            a.sort_indices()

            flag = np.concatenate(([True], a.coords[0, 1:] != a.coords[0, :-1]))
            # Partial credit to @shoyer
            # Ref: https://gist.github.com/shoyer/f538ac78ae904c936844
            inv_idx, = flag.nonzero()
            result = method.reduceat(a.data, inv_idx, **kwargs)
            counts = np.diff(np.concatenate(np.nonzero(flag) + ([a.nnz],)))
            missing_counts = counts != a.shape[1]
            result[missing_counts] = method(result[missing_counts],
                                            _zero_of_dtype(result.dtype))

            a = COO(np.asarray([a.coords[0, inv_idx]]), result, shape=(np.prod(neg_axis),),
                    has_duplicates=False, sorted=True)

            a = a.reshape([self.shape[d] for d in neg_axis])
            result = a

        if keepdims:
            result = _keepdims(self, result, axis)
        return result

    def sum(self, axis=None, keepdims=False, dtype=None, out=None):
        assert out is None
        return self.reduce(np.add, axis=axis, keepdims=keepdims, dtype=dtype)

    def max(self, axis=None, keepdims=False, out=None):
        assert out is None
        return self.reduce(np.maximum, axis=axis, keepdims=keepdims)

    def transpose(self, axes=None):
        if axes is None:
            axes = reversed(range(self.ndim))

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
        return self.transpose(list(range(self.ndim))[::-1])

    def dot(self, other):
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

    def linear_loc(self, signed=False):
        """ Index location of every piece of data in a flattened array

        This is used internally to check for duplicates, re-order, reshape,
        etc..
        """
        return self._linear_loc(self.coords, self.shape, signed)

    @staticmethod
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

    def reshape(self, shape):
        if self.shape == shape:
            return self
        if any(d == -1 for d in shape):
            extra = int(np.prod(self.shape) /
                        np.prod([d for d in shape if d != -1]))
            shape = tuple([d if d != -1 else extra for d in shape])

        if self.shape == shape:
            return self

        if self._cache is not None:
            for sh, value in self._cache['reshape']:
                if sh == shape:
                    return value

        # TODO: this np.prod(self.shape) enforces a 2**64 limit to array size
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
        assert self.ndim == 2
        result = scipy.sparse.coo_matrix((self.data,
                                          (self.coords[0],
                                           self.coords[1])),
                                         shape=self.shape)
        result.has_canonical_format = (not self.has_duplicates and self.sorted)
        return result

    def _tocsr(self):
        assert self.ndim == 2

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
        if self.sorted:
            return

        linear = self.linear_loc(signed=True)

        if (np.diff(linear) > 0).all():  # already sorted
            self.sorted = True
            return self

        order = np.argsort(linear)
        self.coords = self.coords[:, order]
        self.data = self.data[order]
        self.sorted = True
        return self

    def sum_duplicates(self):
        # Inspired by scipy/sparse/coo.py::sum_duplicates
        # See https://github.com/scipy/scipy/blob/master/LICENSE.txt
        if not self.has_duplicates:
            return self
        if not np.prod(self.coords.shape):
            return self

        self.sort_indices()

        linear = self.linear_loc()
        unique_mask = np.diff(linear) != 0

        if unique_mask.sum() == len(unique_mask):  # already unique
            self.has_duplicates = False
            return self

        unique_mask = np.append(True, unique_mask)

        coords = self.coords[:, unique_mask]
        (unique_inds,) = np.nonzero(unique_mask)
        data = np.add.reduceat(self.data, unique_inds, dtype=self.data.dtype)

        self.data = data
        self.coords = coords
        self.has_duplicates = False

        return self

    def __add__(self, other):
        return self.elemwise(operator.add, other)

    __radd__ = __add__

    def __neg__(self):
        return self.elemwise(operator.neg)

    def __sub__(self, other):
        return self.elemwise(operator.sub, other)

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        return self.elemwise(operator.mul, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.elemwise(operator.truediv, other)

    def __floordiv__(self, other):
        return self.elemwise(operator.floordiv, other)

    __div__ = __truediv__

    def __pow__(self, other):
        return self.elemwise(operator.pow, other)

    def __and__(self, other):
        return self.elemwise(operator.and_, other)

    def __xor__(self, other):
        return self.elemwise(operator.xor, other)

    def __or__(self, other):
        return self.elemwise(operator.or_, other)

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

    def __rshift__(self, other):
        return self.elemwise(operator.rshift, other)

    @staticmethod
    def _elemwise(func, *args, **kwargs):
        assert len(args) >= 1
        self = args[0]
        if isinstance(self, scipy.sparse.spmatrix):
            self = COO.from_numpy(self)

        if len(args) == 1:
            return self._elemwise_unary(func, *args[1:], **kwargs)
        else:
            other = args[1]
            if isinstance(other, COO):
                return self._elemwise_binary(func, *args[1:], **kwargs)
            elif isinstance(other, scipy.sparse.spmatrix):
                other = COO.from_scipy_sparse(other)
                return self._elemwise_binary(func, other, *args[2:], **kwargs)
            else:
                return self._elemwise_unary(func, *args[1:], **kwargs)

    def elemwise(self, func, *args, **kwargs):
        """
        Apply a function to one or two arguments.

        Parameters
        ----------
        func
            The function to apply to one or two arguments.
        args : tuple, optional
            The extra arguments to pass to the function. If args[0] is a COO object
            or a scipy.sparse.spmatrix, the function will be treated as a binary
            function. Otherwise, it will be treated as a unary function.
        kwargs : dict, optional
            The kwargs to pass to the function.

        Returns
        -------
        COO
            The result of applying the function.
        """
        return COO._elemwise(func, self, *args, **kwargs)

    def _elemwise_unary(self, func, *args, **kwargs):
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

    def _elemwise_binary(self, func, other, *args, **kwargs):
        assert isinstance(other, COO)
        check = kwargs.pop('check', True)
        self_zero = _zero_of_dtype(self.dtype)
        other_zero = _zero_of_dtype(other.dtype)
        func_zero = _zero_of_dtype(func(self_zero, other_zero, *args, **kwargs).dtype)
        if check and func(self_zero, other_zero, *args, **kwargs) != func_zero:
            raise ValueError("Performing this operation would produce "
                             "a dense result: %s" % str(func))
        self_shape, other_shape = self.shape, other.shape

        result_shape = self._get_broadcast_shape(self_shape, other_shape)
        self_params = self._get_broadcast_parameters(self.shape, result_shape)
        other_params = self._get_broadcast_parameters(other.shape, result_shape)
        combined_params = [p1 and p2 for p1, p2 in zip(self_params, other_params)]
        self_reduce_params = combined_params[-self.ndim:]
        other_reduce_params = combined_params[-other.ndim:]

        self.sum_duplicates()  # TODO: document side-effect or make copy
        other.sum_duplicates()  # TODO: document side-effect or make copy

        self_coords = self.coords
        self_data = self.data

        self_reduced_coords, self_reduced_shape = \
            self._get_reduced_coords(self_coords, self_shape,
                                     self_reduce_params)
        self_reduced_linear = self._linear_loc(self_reduced_coords, self_reduced_shape)
        i = np.argsort(self_reduced_linear)
        self_reduced_linear = self_reduced_linear[i]
        self_coords = self_coords[:, i]
        self_data = self_data[i]

        # Store coords
        other_coords = other.coords
        other_data = other.data

        other_reduced_coords, other_reduced_shape = \
            self._get_reduced_coords(other_coords, other_shape,
                                     other_reduce_params)
        other_reduced_linear = self._linear_loc(other_reduced_coords, other_reduced_shape)
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
        matched_coords = self._get_matching_coords(self_coords[:, matched_self],
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
                self._get_unmatched_coords_data(self_coords, self_func, self_shape,
                                                result_shape, matched_self,
                                                matched_coords)

            data_list.extend(self_unmatched_func)
            coords_list.extend(self_unmatched_coords)

        other_func = func(self_zero, other_data, *args, **kwargs)

        if (other_func != func_zero).any():
            other_unmatched_coords, other_unmatched_func = \
                self._get_unmatched_coords_data(other_coords, other_func, other_shape,
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

    @staticmethod
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
        params = COO._get_broadcast_parameters(shape, result_shape)
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
            COO._get_expanded_coords_data(coords[:, unmatched],
                                          data[unmatched],
                                          params,
                                          result_shape)

        coords_list.append(unmatched_coords)
        data_list.append(unmatched_data)

        if shape != result_shape:
            broadcast_coords, broadcast_data = \
                COO._get_broadcast_coords_data(coords[:, matched],
                                               matched_coords,
                                               data[matched],
                                               params,
                                               result_shape)

            coords_list.append(broadcast_coords)
            data_list.append(broadcast_data)

        return coords_list, data_list

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

        all_idx = COO._cartesian_product(*(np.arange(d, dtype=np.min_scalar_type(d - 1)) for d in expand_shapes))
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
    @staticmethod
    def _cartesian_product(*arrays):
        """
        Get the cartesian product of a number of arrays.

        Parameters
        ----------
        arrays : Iterable[np.ndarray]
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

    def broadcast_to(self, shape):
        """
        Performs the equivalent of np.broadcast_to for COO.

        Parameters
        ----------
        shape : tuple[int]
            The shape to broadcast the data to.

        Returns
        -------
            The broadcasted sparse array.

        Raises
        ------
        ValueError
            If the operand cannot be broadcast to the given shape.
        """
        result_shape = self._get_broadcast_shape(self.shape, shape, is_result=True)
        params = self._get_broadcast_parameters(self.shape, result_shape)
        coords, data = self._get_expanded_coords_data(self.coords, self.data, params, result_shape)

        return COO(coords, data, shape=result_shape, has_duplicates=self.has_duplicates,
                   sorted=self.sorted)

    @staticmethod
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
        result_shape = COO._get_broadcast_shape(shape1, shape2)
        params1 = COO._get_broadcast_parameters(shape1, result_shape)
        params2 = COO._get_broadcast_parameters(shape2, result_shape)

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

    @staticmethod
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
        full_coords, full_data = COO._get_expanded_coords_data(coords, data, params, broadcast_shape)
        linear_full_coords = COO._linear_loc(full_coords, broadcast_shape)
        linear_matched_coords = COO._linear_loc(matched_coords, broadcast_shape)

        overlapping_coords, _ = _match_arrays(linear_full_coords, linear_matched_coords)
        mask = np.ones(full_coords.shape[1], dtype=np.bool)
        mask[overlapping_coords] = False

        return full_coords[:, mask], full_data[mask]

    def __abs__(self):
        return self.elemwise(abs)

    def exp(self, out=None):
        assert out is None
        return self.elemwise(np.exp)

    def expm1(self, out=None):
        assert out is None
        return self.elemwise(np.expm1)

    def log1p(self, out=None):
        assert out is None
        return self.elemwise(np.log1p)

    def sin(self, out=None):
        assert out is None
        return self.elemwise(np.sin)

    def sinh(self, out=None):
        assert out is None
        return self.elemwise(np.sinh)

    def tan(self, out=None):
        assert out is None
        return self.elemwise(np.tan)

    def tanh(self, out=None):
        assert out is None
        return self.elemwise(np.tanh)

    def sqrt(self, out=None):
        assert out is None
        return self.elemwise(np.sqrt)

    def ceil(self, out=None):
        assert out is None
        return self.elemwise(np.ceil)

    def floor(self, out=None):
        assert out is None
        return self.elemwise(np.floor)

    def round(self, decimals=0, out=None):
        assert out is None
        return self.elemwise(np.round, decimals)

    def rint(self, out=None):
        assert out is None
        return self.elemwise(np.rint)

    def conj(self, out=None):
        assert out is None
        return self.elemwise(np.conj)

    def conjugate(self, out=None):
        assert out is None
        return self.elemwise(np.conjugate)

    def astype(self, dtype, out=None):
        assert out is None
        return self.elemwise(np.ndarray.astype, dtype)

    def maybe_densify(self, allowed_nnz=1e3, allowed_fraction=0.25):
        """ Convert to a dense numpy array if not too costly.  Err othrewise """
        if reduce(operator.mul, self.shape) <= allowed_nnz or self.nnz >= np.prod(self.shape) * allowed_fraction:
            return self.todense()
        else:
            raise NotImplementedError("Operation would require converting "
                                      "large sparse array to dense")


def tensordot(a, b, axes=2):
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


def _mask(coords, idx):
    if isinstance(idx, numbers.Integral):
        return coords == idx
    elif isinstance(idx, slice):
        if idx.step not in (1, None):
            raise NotImplementedError("Steped slices not implemented")
        start = idx.start if idx.start is not None else 0
        stop = idx.stop if idx.stop is not None else np.inf
        return (coords >= start) & (coords < stop)
    elif isinstance(idx, list):
        mask = np.zeros(len(coords), dtype=bool)
        for item in idx:
            mask |= coords == item
        return mask


def concatenate(arrays, axis=0):
    arrays = [x if type(x) is COO else COO(x) for x in arrays]
    if axis < 0:
        axis = axis + arrays[0].ndim
    assert all(x.shape[ax] == arrays[0].shape[ax]
               for x in arrays
               for ax in set(range(arrays[0].ndim)) - {axis})
    data = np.concatenate([x.data for x in arrays])
    coords = np.concatenate([x.coords for x in arrays], axis=1)

    nnz = 0
    dim = 0
    for x in arrays:
        if dim:
            coords[axis, nnz:x.nnz + nnz] += dim
        dim += x.shape[axis]
        nnz += x.nnz

    shape = list(arrays[0].shape)
    shape[axis] = dim
    has_duplicates = any(x.has_duplicates for x in arrays)

    return COO(coords, data, shape=shape, has_duplicates=has_duplicates,
               sorted=(axis == 0) and all(a.sorted for a in arrays))


def stack(arrays, axis=0):
    assert len(set(x.shape for x in arrays)) == 1
    arrays = [x if type(x) is COO else COO(x) for x in arrays]
    if axis < 0:
        axis = axis + arrays[0].ndim + 1
    data = np.concatenate([x.data for x in arrays])
    coords = np.concatenate([x.coords for x in arrays], axis=1)

    nnz = 0
    dim = 0
    new = np.empty(shape=(coords.shape[1],), dtype=coords.dtype)
    for x in arrays:
        new[nnz:x.nnz + nnz] = dim
        dim += 1
        nnz += x.nnz

    shape = list(arrays[0].shape)
    shape.insert(axis, len(arrays))
    has_duplicates = any(x.has_duplicates for x in arrays)
    coords = [coords[i] for i in range(coords.shape[0])]
    coords.insert(axis, new)
    coords = np.stack(coords, axis=0)

    return COO(coords, data, shape=shape, has_duplicates=has_duplicates,
               sorted=(axis == 0) and all(a.sorted for a in arrays))


def triu(x, k=0):
    """
    Calculates the equivalent of np.triu(x, k) for COO.

    Parameters
    ----------
    x : COO
        The input array.
    k : int
        The diagonal below which elements are set to zero.

    Returns
    -------
    COO
        The output upper-triangular matrix.
    """
    if not x.ndim >= 2:
        raise NotImplementedError('sparse.triu is not implemented for scalars or 1-D arrays.')

    mask = x.coords[-2] + k <= x.coords[-1]

    coords = x.coords[:, mask]
    data = x.data[mask]

    return COO(coords, data, x.shape, x.has_duplicates, x.sorted)


def tril(x, k=0):
    """
    Calculates the equivalent of np.tril(x, k) for COO.

    Parameters
    ----------
    x : COO
        The input array.
    k : int
        The diagonal above which elements are set to zero.

    Returns
    -------
    COO
        The output lower-triangular matrix.
    """
    if not x.ndim >= 2:
        raise NotImplementedError('sparse.tril is not implemented for scalars or 1-D arrays.')

    mask = x.coords[-2] + k >= x.coords[-1]

    coords = x.coords[:, mask]
    data = x.data[mask]

    return COO(coords, data, x.shape, x.has_duplicates, x.sorted)


def _zero_of_dtype(dtype):
    """
    Creates a ()-shaped 0-sized array of a given dtype
    Parameters
    ----------
    dtype : np.dtype
        The dtype for the array.
    Returns
    -------
    The zero array.
    """
    return np.zeros((), dtype=dtype)


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
