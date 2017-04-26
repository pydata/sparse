from __future__ import absolute_import, division, print_function

from collections import Iterable
from functools import reduce
from numbers import Number
import operator

import numpy as np
import scipy.sparse


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
    >>> s = COO.from_numpy(x)
    >>> s
    <COO: shape=(4, 4), dtype=float64, nnz=5>
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
    <COO: shape=(3, 4, 5), dtype=int64, nnz=5>
    >>> tensordot(s, y, axes=(0, 1))
    <COO: shape=(4, 3, 5), dtype=float64, nnz=6>

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
    <COO: shape=(2, 3, 4), dtype=int64, nnz=3>

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

    def __init__(self, coords, data=None, shape=None, has_duplicates=True):
        if data is None:
            # {(i, j, k): x, (i, j, k): y, ...}
            if isinstance(coords, dict):
                coords = list(coords.items())

            if isinstance(coords, np.ndarray):
                result = COO.from_numpy(coords)
                self.coords = result.coords
                self.data = result.data
                self.has_duplicates = result.has_duplicates
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
            self.coords = np.zeros((len(shape), 0), dtype=int)

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

    @classmethod
    def from_numpy(cls, x):
        coords = np.where(x)
        data = x[coords]
        coords = np.vstack(coords)
        return cls(coords, data, shape=x.shape)

    def todense(self):
        self = self.sum_duplicates()
        x = np.zeros(shape=self.shape, dtype=self.dtype)

        coords = tuple([self.coords[i, :] for i in range(self.ndim)])
        x[coords] = self.data
        return x

    @classmethod
    def from_scipy_sparse(cls, x):
        x = scipy.sparse.coo_matrix(x)
        coords = np.empty((2, x.nnz), dtype=x.row.dtype)
        coords[0, :] = x.row
        coords[1, :] = x.col
        return COO(coords, x.data, shape=x.shape, has_duplicates=not x.has_canonical_format)

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
        index = tuple(ind + self.shape[i] if isinstance(ind, int) and ind < 0 else ind
                      for i, ind in enumerate(index))
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
            if isinstance(ind, int):
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

        coords = np.stack(coords, axis=0)
        shape = tuple(shape)
        data = self.data[mask]

        return COO(coords, data, shape=shape, has_duplicates=self.has_duplicates)

    def __str__(self):
        return "<COO: shape=%s, dtype=%s, nnz=%d>" % (self.shape, self.dtype,
                self.nnz)

    __repr__ = __str__

    def reduction(self, method, axis=None, keepdims=False, dtype=None):
        if axis is None:
            axis = tuple(range(self.ndim))

        kwargs = {}
        if dtype:
            kwargs['dtype'] = dtype

        if isinstance(axis, int):
            axis = (axis,)

        if set(axis) == set(range(self.ndim)):
            result = getattr(self.data, method)(**kwargs)
        else:
            axis = tuple(axis)

            neg_axis = list(range(self.ndim))
            for ax in axis:
                neg_axis.remove(ax)
            neg_axis = tuple(neg_axis)

            a = self.transpose(axis + neg_axis)
            a = a.reshape((np.prod([self.shape[d] for d in axis]),
                           np.prod([self.shape[d] for d in neg_axis])))

            a = a.to_scipy_sparse()
            a = getattr(a, method)(axis=0, **kwargs)
            a = COO.from_scipy_sparse(a)
            a = a.reshape([self.shape[d] for d in neg_axis])
            result = a

        if keepdims:
            result = _keepdims(self, result, axis)
        return result

    def sum(self, axis=None, keepdims=False, dtype=None, out=None):
        return self.reduction('sum', axis=axis, keepdims=keepdims, dtype=dtype)

    def max(self, axis=None, keepdims=False, out=None):
        x = self.reduction('max', axis=axis, keepdims=keepdims)
        # TODO: verify that there are some missing elements in each entry
        if isinstance(x, COO):
            x.data[x.data < 0] = 0
            return x
        elif isinstance(x, np.ndarray):
            x[x < 0] = 0
            return x
        else:
            return np.max(x, 0)

    def transpose(self, axes=None):
        if axes is None:
            axes = reversed(range(self.ndim))

        axes = tuple(axes)

        if axes == tuple(range(self.ndim)):
            return self

        shape = tuple(self.shape[ax] for ax in axes)
        result = COO(self.coords[axes, :], self.data, shape,
                     has_duplicates=self.has_duplicates)

        if axes == (1, 0):
            try:
                result._csc = self._csr.T
            except AttributeError:
                pass
            try:
                result._csr = self._csc.T
            except AttributeError:
                pass
        return result

    @property
    def T(self):
        return self.transpose(list(range(self.ndim))[::-1])

    def dot(self, other):
        return dot(self, other)

    def reshape(self, shape):
        if self.shape == shape:
            return self
        if any(d == -1 for d in shape):
            extra = int(np.prod(self.shape) /
                        np.prod([d for d in shape if d != -1]))
            shape = tuple([d if d != -1 else extra for d in shape])
        if self.shape == shape:
            return self
        # TODO: this np.prod(self.shape) enforces a 2**64 limit to array size
        dtype = np.min_scalar_type(np.prod(self.shape))
        linear_loc = np.zeros(self.nnz, dtype=dtype)
        strides = 1
        for i, d in enumerate(self.shape[::-1]):
            linear_loc += self.coords[-(i + 1), :].astype(dtype) * strides
            strides *= d

        coords = np.empty((len(shape), self.nnz), dtype=np.min_scalar_type(max(self.shape)))
        strides = 1
        for i, d in enumerate(shape[::-1]):
            coords[-(i + 1), :] = (linear_loc // strides) % d
            strides *= d

        return COO(coords, self.data, shape, has_duplicates=self.has_duplicates)

    def to_scipy_sparse(self):
        assert self.ndim == 2
        result = scipy.sparse.coo_matrix((self.data,
                                          (self.coords[0],
                                           self.coords[1])),
                                          shape=self.shape)
        result.has_canonical_format = not self.has_duplicates
        return result

    def tocsr(self):
        try:
            return self._csr
        except AttributeError:
            pass
        try:
            self._csr = self._csc.tocsr()
            return self._csr
        except AttributeError:
            pass

        coo = self.to_scipy_sparse()
        csr = coo.tocsr()
        self._csr = csr
        return csr

    def tocsc(self):
        try:
            return self._csc
        except AttributeError:
            pass
        try:
            self._csc = self._csr.tocsc()
            return self._csc
        except AttributeError:
            pass
        coo = self.to_scipy_sparse()
        csc = coo.tocsc()
        self._csc = csc
        return csc

    def sum_duplicates(self):
        # Inspired by scipy/sparse/coo.py::sum_duplicates
        # See https://github.com/scipy/scipy/blob/master/LICENSE.txt
        if not self.has_duplicates:
            return self
        if not np.prod(self.coords.shape):
            return self
        order = np.lexsort(self.coords)
        coords = self.coords[:, order]
        data = self.data[order]
        unique_mask = (coords[:, 1:] != coords[:, :-1]).any(axis=0)
        unique_mask = np.append(True, unique_mask)

        coords = coords[:, unique_mask]
        (unique_inds,) = np.nonzero(unique_mask)
        data = np.add.reduceat(data, unique_inds, dtype=data.dtype)

        self.data = data
        self.coords = coords
        self.has_duplicates = False

        return self

    def __add__(self, other):
        if not isinstance(other, COO):
            return self.maybe_densify() + other
        if self.shape == other.shape:
            return COO(np.concatenate([self.coords, other.coords], axis=1),
                       np.concatenate([self.data, other.data]),
                       self.shape, has_duplicates=True)
        else:
            raise NotImplementedError("Broadcasting not yet supported")

    def __neg__(self):
        return COO(self.coords, -self.data, self.shape, self.has_duplicates)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, COO):
            return self.elemwise_binary(operator.mul, other)
        else:
            return self.elemwise(operator.mul, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.elemwise(operator.truediv, other)

    def __floordiv__(self, other):
        return self.elemwise(operator.floordiv, other)

    __div__ = __truediv__

    def __pow__(self, other):
        return self.elemwise(operator.pow, other)

    def elemwise(self, func, *args, **kwargs):
        if kwargs.pop('check', True) and func(0, *args, **kwargs) != 0:
            raise ValueError("Performing this operation would produce "
                    "a dense result: %s" % str(func))
        return COO(self.coords, func(self.data, *args, **kwargs),
                   shape=self.shape, has_duplicates=self.has_duplicates)

    def elemwise_binary(self, func, other, *args, **kwargs):
        assert isinstance(other, COO)
        if kwargs.pop('check', True) and func(0, 0, *args, **kwargs) != 0:
            raise ValueError("Performing this operation would produce "
                    "a dense result: %s" % str(func))
        if self.shape != other.shape:
            raise NotImplementedError("Broadcasting is not supported")
        self.sum_duplicates()  # TODO: document side-effect or make copy
        other.sum_duplicates()  # TODO: document side-effect or make copy

        # Sort self.coords in lexographical order using record arrays
        self_coords = np.rec.fromarrays(self.coords)
        i = np.argsort(self_coords)
        self_coords = self_coords[i]
        self_data = self.data[i]

        # Convert other.coords to a record array
        other_coords = np.rec.fromarrays(other.coords)
        other_data = other.data

        # Find matches between self.coords and other.coords
        j = np.searchsorted(self_coords, other_coords)
        if len(self_coords):
            matched_other = (other_coords == self_coords[j % len(self_coords)])
        else:
            matched_other = np.zeros(shape=(0,), dtype=bool)
        matched_self = j[matched_other]

        # Locate coordinates without a match
        unmatched_other = ~matched_other
        unmatched_self = np.ones(len(self_coords), dtype=bool)
        unmatched_self[matched_self] = 0

        # Concatenate matches and mismatches
        data = np.concatenate([func(self_data[matched_self],
                                    other_data[matched_other],
                                    *args, **kwargs),
                               func(self_data[unmatched_self], 0,
                                    *args, **kwargs),
                               func(0, other_data[unmatched_other],
                                    *args, **kwargs)])
        coords = np.concatenate([self_coords[matched_self],
                                 self_coords[unmatched_self],
                                 other_coords[unmatched_other]])

        nonzero = data != 0
        data = data[nonzero]
        coords = coords[nonzero]

        # record array to ND array
        coords = np.asarray(coords.view(coords.dtype[0]).reshape(len(coords), self.ndim)).T

        return COO(coords, data, shape=self.shape, has_duplicates=False)

    def __abs__(self):
        return self.elemwise(abs)

    def exp(self):
        return np.exp(self.maybe_densify())

    def expm1(self):
        return self.elemwise(np.expm1)

    def log1p(self):
        return self.elemwise(np.log1p)

    def sin(self):
        return self.elemwise(np.sin)

    def sinh(self):
        return self.elemwise(np.sinh)

    def tan(self):
        return self.elemwise(np.tan)

    def tanh(self):
        return self.elemwise(np.tanh)

    def sqrt(self):
        return self.elemwise(np.sqrt)

    def ceil(self):
        return self.elemwise(np.ceil)

    def floor(self):
        return self.elemwise(np.floor)

    def round(self, decimals=0):
        return self.elemwise(np.round, decimals)

    def rint(self):
        return self.elemwise(np.rint)

    def conj(self):
        return self.elemwise(np.conj)

    def conjugate(self):
        return self.elemwise(np.conjugate)

    def astype(self, dtype):
        return self.elemwise(np.ndarray.astype, dtype, check=False)

    def __gt__(self, other):
        if not isinstance(other, Number):
            raise NotImplementedError("Only scalars supported")
        if other < 0:
            raise ValueError("Comparison with negative number would produce "
                             "dense result")
        return self.elemwise(operator.gt, other)

    def __ge__(self, other):
        if not isinstance(other, Number):
            raise NotImplementedError("Only scalars supported")
        if other <= 0:
            raise ValueError("Comparison with negative number would produce "
                             "dense result")
        return self.elemwise(operator.ge, other)

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
    except:
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
    if isinstance(res, np.matrix):
        res = np.asarray(res)
    return res.reshape(olda + oldb)


def dot(a, b):
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
    if isinstance(idx, int):
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

    return COO(coords, data, shape=shape, has_duplicates=has_duplicates)


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

    return COO(coords, data, shape=shape, has_duplicates=has_duplicates)
