# Copyright 2019 Steffen Schroeder
# All the contents of the file have been copied from here
# https://gist.github.com/stschroe/7c4660774835258c06b2be968eb49204#file-test_sparse_dok_indexer-py

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sparse import DOK

from sparse.numpy_vindex import VindexArray
import sparse._dok_indexer


def test_vix():
    assert_array_equal = np.testing.assert_array_equal
    x = np.arange(3 * 4 * 5).reshape((3, 4, 5)).view(VindexArray)
    d = DOK.from_numpy(x)

    # # Test getitem
    # sequences
    assert_array_equal(d.vix[[]].todense(), x[[]])
    assert_array_equal(d.vix[[], []].todense(), x[[], []])
    assert_array_equal(d.vix[[2]].todense(), x[[2]])
    assert_array_equal(d.vix[[1, 2]].todense(), x[[1, 2]])
    assert_array_equal(d.vix[[1, 2], [1, 2]].todense(), x[[1, 2], [1, 2]])
    assert_array_equal(
        d.vix[[1, 2], [1, 2], [1, 2]].todense(), x[[1, 2], [1, 2], [1, 2]]
    )
    # slices
    assert_array_equal(d.vix[[0, 1], [0, 1], :].todense(), x.vindex[[0, 1], [0, 1], :])
    assert_array_equal(d.vix[[0, 1], :, [0, 1]].todense(), x.vindex[[0, 1], :, [0, 1]])
    assert_array_equal(d.vix[:, [0, 1], [0, 1]].todense(), x.vindex[:, [0, 1], [0, 1]])
    assert_array_equal(d.vix[:, :, [0, 1]].todense(), x.vindex[:, :, [0, 1]])
    assert_array_equal(d.vix[:, [0, 1], :].todense(), x.vindex[:, [0, 1], :])
    assert_array_equal(d.vix[[0, 1], :, :].todense(), x.vindex[[0, 1], :, :])
    assert_array_equal(d.vix[[0, 1], [0, 1], 1].todense(), x.vindex[[0, 1], [0, 1], 1])
    assert_array_equal(d.vix[[0, 1], 1, [0, 1]].todense(), x.vindex[[0, 1], 1, [0, 1]])
    assert_array_equal(d.vix[1, [0, 1], [0, 1]].todense(), x.vindex[1, [0, 1], [0, 1]])
    assert_array_equal(d.vix[1, 1, [0, 1]].todense(), x.vindex[1, 1, [0, 1]])
    assert_array_equal(d.vix[1, [0, 1], 1].todense(), x.vindex[1, [0, 1], 1])
    assert_array_equal(d.vix[[0, 1], 1, 1].todense(), x.vindex[[0, 1], 1, 1])
    # different behaviour between scalar indices and slices resulting in one scalar
    assert_array_equal(
        d.vix[[0, 1], [0, 1], 0:1].todense(), x.vindex[[0, 1], [0, 1], 0:1]
    )
    assert_array_equal(
        d.vix[[0, 1], 0:1, [0, 1]].todense(), x.vindex[[0, 1], 0:1, [0, 1]]
    )
    assert_array_equal(
        d.vix[0:1, [0, 1], [0, 1]].todense(), x.vindex[0:1, [0, 1], [0, 1]]
    )
    assert_array_equal(d.vix[0:1, 0:1, [0, 1]].todense(), x.vindex[0:1, 0:1, [0, 1]])
    assert_array_equal(d.vix[0:1, [0, 1], 0:1].todense(), x.vindex[0:1, [0, 1], 0:1])
    assert_array_equal(d.vix[[0, 1], 0:1, 0:1].todense(), x.vindex[[0, 1], 0:1, 0:1])
    # test empty slice
    assert_array_equal(
        d.vix[[0, 1], [0, 1], 1:1].todense(), x.vindex[[0, 1], [0, 1], 1:1]
    )
    # test fallback to slice_getitem
    assert_array_equal(d.vix[1, 1, 1], x.vindex[1, 1, 1])
    assert_array_equal(d.vix[1, 1, :].todense(), x.vindex[1, 1, :])
    assert_array_equal(d.vix[1, :, :].todense(), x.vindex[1, :, :])
    assert_array_equal(d.vix[:, :, :].todense(), x.vindex[:, :, :])

    # # test setitem
    d.vix[[0, 1], [1, 1], [0, 0]] = [1, 2]
    x.vindex[[0, 1], [1, 1], [0, 0]] = [1, 2]
    assert_array_equal(d.todense(), x)
    d.vix[[0, 1], [0, 0], [0, 0]] = 2
    x[[0, 1], [0, 0], [0, 0]] = 2
    assert_array_equal(d.todense(), x)
    d.vix[[0, 1], [1, 1], [0, 0]] = d.vix[[0, 1], [0, 0], [0, 0]].todense()
    x.vindex[[0, 1], [1, 1], [0, 0]] = x.vindex[[0, 1], [0, 0], [0, 0]]
    assert_array_equal(d.todense(), x)


def test_aix():
    assert_array_equal = np.testing.assert_array_equal
    x = np.arange(3 * 4 * 5).reshape((3, 4, 5)).view(VindexArray)
    d = DOK.from_numpy(x)
    # # test getitem and setitem
    d.aix[[0, 1], [1, 1], [0, 0]] = [1, 2]
    x.vindex[[0, 1], [1, 1], [0, 0]] = [1, 2]
    assert_array_equal(d.todense(), x)
    d.aix[[0, 1], [0, 0], [0, 0]] = 2
    x[[0, 1], [0, 0], [0, 0]] = 2
    assert_array_equal(d.todense(), x)
    d.aix[[0, 1], [1, 1], [0, 0]] = d.aix[[0, 1], [0, 0], [0, 0]]
    x.vindex[[0, 1], [1, 1], [0, 0]] = x.vindex[[0, 1], [0, 0], [0, 0]]
    assert_array_equal(d.todense(), x)
