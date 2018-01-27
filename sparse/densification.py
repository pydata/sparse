import sys
from numbers import Integral, Number
from collections import Iterable
import threading

import numpy as np

from .utils import _get_broadcast_shape, SparseArray, TriState

try:  # Windows compatibility
    int = long
except NameError:
    pass

try:
    _max_size = sys.maxsize
except NameError:
    _max_size = sys.maxint


class DensificationConfig(object):
    def __init__(self, densify=None, max_size=10000, min_density=0.25):
        if not isinstance(densify, bool) and densify is not None:
            raise ValueError('always_densify must be a bool or None.')

        if not isinstance(max_size, Integral) or max_size < 0:
            raise ValueError("max_nnz must be a non-negative integer.")

        if not isinstance(min_density, Number) or not (0.0 <= min_density <= 1.0):
            raise ValueError('min_density must be a number between 0 and 1.')

        self.densify = TriState(densify)
        self.max_size = int(max_size)
        self.min_density = float(min_density)
        self.children = None
        self.parents = None
        self._disconnected_parents = 0
        self._parents_with_children = 0
        self._lock = threading.Lock()

    def _should_densify(self, size, density):
        if self.densify.value is True:
            return True
        elif self.densify.value is False:
            return False
        else:
            return self.max_size >= size or self.min_density <= density

    def _raise_if_fails(self, size, density, name):
        if not self._should_densify(size, density):
            raise ValueError("Performing this operation would produce "
                             "a dense result: %s" % name)

    def check(self, name, *arrays):
        config = self._reduce_from_parents()
        result_shape = ()
        density = 1.0

        for arr in arrays:
            if isinstance(arr, SparseArray):
                density = min(density, arr.density)
                result_shape = _get_broadcast_shape(result_shape, arr.shape)

        size = np.prod(result_shape, dtype=np.uint64)

        config._raise_if_fails(size, density, name)

    @staticmethod
    def validate(configs):
        if isinstance(configs, Iterable):
            for config in configs:
                DensificationConfig._validate_single(config)

            return

        DensificationConfig._validate_single(configs)

    @staticmethod
    def _validate_single(config):
        if not isinstance(config, DensificationConfig):
            raise ValueError('Invalid DensificationConfig.')

    @staticmethod
    def from_parents(parents):
        root_parents = set()

        for parent in parents:
            DensificationConfig._validate_single(parent)
            root_parents.update(parent._get_all_parents())

            result = DensificationConfig()
            result.parents = root_parents

            for parent in root_parents:
                if isinstance(parent.children, set):
                    parent.children.add(result)
                    result._parents_with_children += 1

            if not result._parents_with_children:
                result._reduce_from_parents(in_place=True)

            return result

    def _get_all_parents(self):
        if isinstance(self.parents, Iterable):
            return self.parents

        parents = set()
        parents.add(self)

        return parents

    def _reduce_from_parents(self, in_place=False):
        if not isinstance(self.parents, Iterable):
            return self

        max_size = _max_size
        min_density = 0.0
        densify = TriState(True)

        for parent in self.parents:
            max_size = min(max_size, parent.max_size)
            min_density = max(min_density, parent.min_density)
            densify = min(densify, parent.densify)

        if in_place:
            self.parents = None
            self._disconnected_parents = 0
            self._parents_with_children = 0
            self.max_size = max_size
            self.min_density = min_density
            self.densify = densify
            return self

        return DensificationConfig(densify, max_size, min_density)

    def __enter__(self):
        self.children = set()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for child in self.children:
            child._lock.acquire()
            child._disconnected_parents += 1
            if child._disconnected_parents == child._parents_with_children:
                self._reduce_from_parents(in_place=True)

            child._lock.release()

        self.children = None

    def __str__(self):
        if isinstance(self.parents, set):
            return '<DensificationConfig: parents=%s>' % len(self.parents)
        elif isinstance(self.densify.value, bool):
            return '<DensificationConfig: densify=%s>' % self.densify.value
        else:
            return '<DensificationConfig: max_size=%s, min_density=%s>' % \
                   (self.max_size, self.min_density)
