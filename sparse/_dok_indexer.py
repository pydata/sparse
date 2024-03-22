# Copyright 2019 Steffen Schroeder
# All the contents of the file have been copied from here
# https://gist.github.com/stschroe/7c4660774835258c06b2be968eb49204#file-sparse_dok_indexer-py

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import product, repeat
from numbers import Integral

import numpy as np
from sparse import DOK, SparseArray
from sparse._slicing import normalize_index


class DOKVectorIndexer:
    """Vector indexing
    Allows so called "fancy" or advanced indexing with one dimensional
    sequences (here named as vectors) as array indices. As opposed to
    numpy legacy indexing, the vector result dimensions are always the
    first axes of the result array. This is inspired by NEP 21:
    https://www.numpy.org/neps/nep-0021-advanced-indexing.html
    It is not fully implemented yet. Especially the setting of array
    elements.
    """

    def __init__(self, dok):
        self._dok = dok

    def slice_getitem(self, keys):
        """Indexing with slices and scalars only (outer indexing)
        expects with normalize_index() normalized indices
        """
        shape = self._dok.shape
        idxs, res_idxs = [], []
        res_shape = [None] * self._dok.ndim
        for i, key in enumerate(keys):
            if isinstance(key, slice):
                ir = range(*key.indices(shape[i]))
                idxs.append(ir)
                lr = len(ir)
                res_idxs.append(range(lr))
                res_shape[i] = (
                    lr if lr > 0 else 0
                )  # if 0: dimension gets preserved but set to zero
            else:  # should be integral or 0d array of integral
                idxs.append((int(key),))
                res_idxs.append((0,))
                res_shape[i] = -1  # dimension gets removed in result
        if any(rs == 0 for rs in res_shape):
            new_res_shape = tuple(rs for rs in res_shape if rs >= 0)
            return DOK(
                shape=new_res_shape,
                dtype=self._dok.dtype,
                fill_value=self._dok.fill_value,
            )
        data = self._dok.data
        res_data = dict()
        res_shape_idxs = [i for i, s in enumerate(res_shape) if res_shape[i] > 0]
        for res_key, key in zip(product(*res_idxs), product(*idxs)):
            if key in data:
                res_data[tuple(res_key[i] for i in res_shape_idxs)] = data[key]
        new_res_shape = tuple(s for s in res_shape if s > 0)
        if new_res_shape == ():
            if key in data:
                return data[key]
            else:
                return self._dok.fill_value
        else:
            return DOK(
                shape=new_res_shape,
                data=res_data,
                dtype=self._dok.dtype,
                fill_value=self._dok.fill_value,
            )

    def __getitem__(self, keys):
        """Indexing with vectors, slices and scalar integers.
        Example::
            @property
            def vix(self):
                return DOKVectorIndexer(self)
            DOK.vix = vix
            d = DOK((3, 4, 5), dtype=np.int64)
            d0 = d.vix[[1, 2]]
            d1 = d.vix[[1, 2], [1, 2], [1, 2]]
            d2 = d.vix[:, [0, 1], [0, 1]]
            d3 = d.vix[[0, 1], [0, 1], 1]
        """
        shape = self._dok.shape
        # expands keys to full dimension, sequences to arrays, positive indices
        keys = normalize_index(keys, shape)
        vecs = []
        idxs, res_idxs = [], []
        res_shape = [None] * self._dok.ndim
        for i, key in enumerate(keys):
            # normalize_index() may contain arrays of ndim 0 and 1
            if isinstance(key, np.ndarray) and key.ndim == 1:
                vecs.append(key)
                arr_len = len(key)
                res_idxs.append((range(arr_len),))
                res_shape[i] = None  # mark vector indices
                idxs.append((key,))
            elif isinstance(key, slice):
                ir = range(*key.indices(shape[i]))
                il = [repeat(i) for i in ir]
                idxs.append(il)
                res_idxs.append([repeat(i) for i in range(len(il))])
                if len(il) > 0:
                    res_shape[i] = len(il)
                else:
                    res_shape[i] = 0  # dimension gets preserved but set to zero
            else:  # should be integral or 0d array of integral
                idxs.append((repeat(int(key)),))
                res_idxs.append((repeat(0),))
                res_shape[i] = -1  # dimension gets removed in result
        if vecs:
            vec_len = len(vecs[0])
            if not all(vec_len == len(v) for v in vecs[1:]):
                raise IndexError("Unequal length of index sequences!")
        else:
            # TODO: replace with DOC.__getitem__, when it supports such combinations
            return self.slice_getitem(keys)
        if any(
            rs == 0 for rs in res_shape
        ):  # dimension of length zero results in empty array
            new_res_shape = (
                arr_len,
                *[s for s in res_shape if s is not None and s >= 0],
            )
            return DOK(
                shape=new_res_shape,
                dtype=self._dok.dtype,
                fill_value=self._dok.fill_value,
            )
        non_vec_idxs = [
            i for i, s in enumerate(res_shape) if s is not None and res_shape[i] > 0
        ]
        vec_idxs = [i for i, s in enumerate(res_shape) if s is None]
        first_vec_idx = vec_idxs[0]
        data = self._dok.data
        res_data = dict()
        for res_key, key in zip(product(*res_idxs), product(*idxs)):
            for rk, k in zip(zip(*res_key), zip(*key)):
                if k in data:
                    reordered_rk = (rk[first_vec_idx], *[rk[i] for i in non_vec_idxs])
                    res_data[reordered_rk] = data[k]
        new_res_shape = (arr_len, *[s for s in res_shape if s is not None and s > 0])
        return DOK(
            shape=new_res_shape,
            data=res_data,
            dtype=self._dok.dtype,
            fill_value=self._dok.fill_value,
        )

    def __setitem__(self, idxs, values):
        """Indexing supports only vectors as indices.
        Accepted values are vectors and scalars.
        Example::
            @property
            def vix(self):
                return DOKVectorIndexer(self)
            DOK.vix = vix
            d = DOK((2, 2), dtype=np.int64)
            d.vix[[0, 1],[1, 1]] = [1, 2]
            d.vix[[0, 1],[0, 0]] = 2
            d.vix[[0, 1],[1, 1]] = d.vix[[0, 1],[0, 0]]
        """
        if not isinstance(idxs, tuple):  # one dimension, one argument
            idxs = (idxs,)
        if len(idxs) != self._dok.ndim:
            raise NotImplementedError(
                f"Index sequences for all {self._dok.ndim} array dimensions needed!"
            )
        idxs = tuple(np.asanyarray(idxs) for idxs in idxs)
        if not (isinstance(k.dtype, Integral) for k in idxs):
            raise IndexError("Indices must be sequences of integer types!")
        if not all(idxs[0].shape == k.shape for k in idxs[1:]):
            raise IndexError("Unequal length of index sequences!")
        if idxs[0].ndim != 1:
            raise IndexError("Indices are not 1d sequences!")
        values = np.asanyarray(values, self._dok.dtype)
        if values.ndim == 0:
            values = np.full(idxs[0].size, values, self._dok.dtype)
        elif values.ndim > 1:
            raise ValueError(f"Dimension of values ({values.ndim}) must be 0 or 1!")
        if not idxs[0].shape == values.shape:
            raise ValueError(
                f"Shape mismatch of indices ({idxs[0].shape}) and values ({values.shape})!"
            )
        fill_value = self._dok.fill_value
        data = self._dok.data
        for idx, value in zip(zip(*idxs), values):
            # self._dok._setitem(k, value)  # costly equallity checks
            if not value == fill_value:
                data[idx] = value
            elif idx in data:
                del data[idx]


def add_vector_indexer_to_sparse_dok():
    """Add vector indexer to sparse.DOK array"""
    # alternative implementation
    # class VDOK(DOK):
    #     @property
    #     def vidx(self):
    #         return VectorIndexer(self)
    #
    # s2 = VDOK((5, 5), dtype=np.int64)

    @property
    def vix(self):
        return DOKVectorIndexer(self)

    DOK.vix = vix


class DOKArrayIndexer:
    """Indexing with vectors only for sparse.DOK
    This is a fast version for vectorized or "fancy" indexing. Only
    vectors (1d sequences) are accepted for indexing. The getitem
    method always returns dense ndarrays in contrast to normal
    fancy indexing of sparse arrays which create new copies of sparse
    arrays. The setitem method allows only vectors and scalars as
    values. This allows fast calculations over ndarray methods
    without the expensive creation of temporary sparse arrays between
    operations. This is of course only useful for small vectors, which
    fit easily in to memory.
    Example::
        @property
        def aix(self):
            return DOKArrayIndexer(self)
        DOK.aix = aix
        d = DOK((2, 2), dtype=np.int64)
        # setitem
        d.aix[[0, 1],[1, 1]] = [1, 2]
        d.aix[[0, 1],[1, 1]] = 2
        d.aix[[0, 1],[1, 1]] = d.vix[[0, 0],[1, 2]]
    """

    def __init__(self, dok):
        self._dok = dok

    def _normalize_idxs(self, idxs):
        if not isinstance(idxs, tuple):  # one dimension, one argument
            idxs = (idxs,)
        if len(idxs) != self._dok.ndim:
            raise NotImplementedError(
                f"Index sequences for all {self._dok.ndim} array dimensions needed!"
            )
        idxs = tuple(np.asanyarray(idxs) for idxs in idxs)
        if not (isinstance(k.dtype, Integral) for k in idxs):
            raise IndexError("Indices must be sequences of integer types!")
        if not all(idxs[0].shape == k.shape for k in idxs[1:]):
            raise IndexError("Unequal length of index sequences!")
        if idxs[0].ndim != 1:
            raise IndexError("Indices are not 1d sequences!")
        return idxs

    def __getitem__(self, idxs):
        idxs = self._normalize_idxs(idxs)
        fill_value = self._dok.fill_value
        data_keys = self._dok.data.keys()
        data = self._dok.data
        return np.array(
            [data[idx] if idx in data_keys else fill_value for idx in zip(*idxs)],
            self._dok.dtype,
        )

    def __setitem__(self, idxs, values):
        idxs = self._normalize_idxs(idxs)
        values = np.asanyarray(values, self._dok.dtype)
        if values.ndim == 0:
            values = np.full(idxs[0].size, values, self._dok.dtype)
        elif values.ndim > 1:
            raise ValueError(f"Dimension of values ({values.ndim}) must be 0 or 1!")
        if not idxs[0].shape == values.shape:
            raise ValueError(
                f"Shape mismatch of indices ({idxs[0].shape}) and values ({values.shape})!"
            )
        fill_value = self._dok.fill_value
        data = self._dok.data
        for idx, value in zip(zip(*idxs), values):
            # self._dok._setitem(k, value)
            if not value == fill_value:
                data[idx] = value
            elif idx in data:
                del data[idx]


def add_array_indexer_to_sparse_dok():
    """Add array indexer to sparse.DOK array"""

    @property
    def aix(self):
        return DOKArrayIndexer(self)

    DOK.aix = aix


add_vector_indexer_to_sparse_dok()
add_array_indexer_to_sparse_dok()
