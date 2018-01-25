import sys
from numbers import Integral, Number
from collections import Iterable

import numpy as np

from .utils import _get_broadcast_shape, SparseArray

try:  # Windows compatibility
    int = long
except NameError:
    pass

try:
    _max_size = sys.maxsize
except NameError:
    _max_size = sys.maxint


class DensificationConfig(object):
    def __init__(self, densify=None, max_size=None, min_density=None):
        if isinstance(densify, DensificationConfig):
            self.densify = densify.densify
            self.max_size = densify.max_size
            self.min_density = densify.min_density
            return

        if max_size is None:
            max_size = 10000

        if min_density is None:
            min_density = 0.25

        if not isinstance(densify, bool) and densify is not None:
            raise ValueError('always_densify must be a bool or None.')

        if not isinstance(max_size, Integral) or max_size < 0:
            raise ValueError("max_nnz must be a non-negative integer.")

        if not isinstance(min_density, Number) or not (0.0 <= min_density <= 1.0):
            raise ValueError('min_density must be a number between 0 and 1.')

        self.densify = densify
        self.max_size = int(max_size)
        self.min_density = float(min_density)

    def _should_densify(self, size, density):
        if self.densify:
            return True
        elif self.densify is False:
            return False
        else:
            return self.max_size >= size or self.min_density <= density

    def _raise_if_fails(self, size, density, name):
        if not self._should_densify(size, density):
            raise ValueError("Performing this operation would produce "
                             "a dense result: %s" % name)

    def check(self, name, *arrays):
        result_shape = ()
        density = 1.0

        for arr in arrays:
            if isinstance(arr, SparseArray):
                density = min(density, arr.density)
                result_shape = _get_broadcast_shape(result_shape, arr.shape)

        size = np.prod(result_shape, dtype=np.uint64)

        self._raise_if_fails(size, density, name)

    @staticmethod
    def validate(configs):
        if isinstance(configs, Iterable):
            for config in configs:
                DensificationConfig.validate(config)

            return

        if not isinstance(configs, DensificationConfig):
            raise ValueError('Invalid DensificationConfig.')

    @staticmethod
    def combine(*configs):
        result = set()

        for config in configs:
            if isinstance(config, Iterable):
                DensificationConfig.validate(config)
                result.update(config)
            elif isinstance(config, DensificationConfig):
                DensificationConfig.validate(config)
                result.add(config)

        return result

    @staticmethod
    def from_many(managers):
        densify = True
        max_size = _max_size
        min_density = 1.0

        for manager in managers:
            densify = _three_way_and(densify, manager.densify)
            max_size = max(max_size, manager.max_size)
            min_density = min(min_density, manager.min_density)

        return DensificationConfig(densify, max_size, min_density)


def _three_way_and(flag1, flag2):
    if flag1:
        return flag2
    elif flag1 is None:
        return False if flag2 is False else None
    else:
        return False
