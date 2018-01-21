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


class DOK:
    def __init__(self, shape, values=None, dtype=None):
        from .coo import COO
        self.dict = {}

        if isinstance(shape, COO):
            self.dtype = shape.dtype
            self.shape = shape.shape

            for c, d in zip(shape.coords.T, shape.data):
                self.dict[tuple(c)] = d

            return

        if isinstance(shape, np.ndarray):
            coords = np.nonzero(shape)
            data = shape[coords]
            self.dtype = shape.dtype
            self.shape = shape.shape

            for c in zip(data, *coords):
                d, c = c[0], c[1:]
                self.dict[c] = d

            return

        self.dtype = dtype
        if isinstance(shape, Integral):
            self.shape = (int(shape),)
        elif isinstance(shape, Iterable):
            if not all(isinstance(l, Integral) or int(l) < 0 for l in shape):
                raise ValueError('shape must be an iterable of non-negative integers.')

            self.shape = tuple(shape)

        if not values:
            values = {}

        if isinstance(values, dict):
            for c, d in six.iteritems(values):
                self[c] = d
        else:
            raise ValueError('values must be a dict.')

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nnz(self):
        return len(self.dict)

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

        if key in self.dict:
            return self.dict[key]
        else:
            return _zero_of_dtype(self.dtype)[()]

    def __setitem__(self, key, value):
        key = normalize_index(key, self.shape)
        value = np.asanyarray(value)

        if not self.dtype:
            self.dtype = value.dtype
        else:
            value = value.astype(self.dtype)

        key_list = [int(k) if isinstance(k, Integral) else k for k in key]

        self._setitem(key_list, value)

    def _setitem(self, key_list, value):
        value_missing_dims = len([ind for ind in key_list if isinstance(ind, slice)]) - value.ndim

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

                for v_idx, ki in enumerate(range(start, stop, step)):
                    key_list_temp = list(key_list)
                    key_list_temp[i] = ki
                    vi = value if value_missing_dims > 0 else value[v_idx]
                    self._setitem(key_list_temp, vi)

                return
            elif not isinstance(ind, Integral):
                raise IndexError('All indices must be slices or integers'
                                 ' when setting an item.')

        self.dict[tuple(key_list)] = value[()]

    def todense(self):
        result = np.zeros(self.shape, dtype=self.dtype)

        for c, d in six.iteritems(self.dict):
            result[c] = d

        return result
