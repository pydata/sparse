# Prototype reference implementation of numpy NEP 21 (https://github.com/numpy/numpy/pull/6256)
# Copied from https://gist.github.com/shoyer/c700193625347eb68fee4d1f0dc8c0c8

# Copyright 2017 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def is_contiguous(positions):
    """Given a non-empty list, does it consist of contiguous integers?"""
    previous = positions[0]
    for current in positions[1:]:
        if current != previous + 1:
            return False
        previous = current
    return True


def advanced_indexer_subspaces(key):
    """Indices of the advanced indexes subspaces for mixed indexing and vindex."""
    if not isinstance(key, tuple):
        key = (key,)
    advanced_index_positions = [
        i for i, k in enumerate(key) if not isinstance(k, slice)
    ]

    if not advanced_index_positions or not is_contiguous(advanced_index_positions):
        # nothing to reorder
        return (), ()

    non_slices = [k for k in key if not isinstance(k, slice)]
    ndim = len(np.broadcast(*non_slices).shape)
    mixed_positions = advanced_index_positions[0] + np.arange(ndim)
    vindex_positions = np.arange(ndim)
    return mixed_positions, vindex_positions


class VectorizedIndexer(object):
    def __init__(self, array):
        self._array = array

    def __getitem__(self, key):
        mixed_positions, vindex_positions = advanced_indexer_subspaces(key)
        return np.moveaxis(self._array[key], mixed_positions, vindex_positions)

    def __setitem__(self, key, value):
        mixed_positions, vindex_positions = advanced_indexer_subspaces(key)
        self._array[key] = np.moveaxis(value, vindex_positions, mixed_positions)


class VindexArray(np.ndarray):
    @property
    def vindex(self):
        return VectorizedIndexer(self)
