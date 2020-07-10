import numpy as np

from numba import njit, types
from numba import typed

from sparse._meta.compressed_level import Compressed
from sparse._meta.dense_level import Dense

from typing import Tuple, List
import collections


@njit
def mul_sparse_dense_1d(
    c: Compressed,
    c_data: typed.List,
    d: Dense,
    d_data: typed.List,
    out: Compressed,
    data_out: typed.List,
):
    out.append_init(1, 100)
    pos_iter = c.pos_iter(0)
    for p_c1 in pos_iter:
        i_c1, found_c1 = c.pos_access(p_c1, ())
        p_d1, found_d1 = d.locate(0, (i_c1,))
        if found_c1 and found_d1:
            out.append_coord(p_c1, i_c1)
            data_c = c_data[p_c1]
            data_d = d_data[p_d1]
            data_out.append(data_c * data_d)
    out.append_edges(0, pbegin1c, pend1c)
    out.append_finalize(1, 100)


def test_sparse_dense_mul_1d():
    pos = typed.List([0, 3])
    crd = typed.List([0, 7, 39])
    c = Compressed(full=True, ordered=True, unique=True, pos=pos, crd=crd)
    data_c = typed.List([types.int64(e) for e in [7, 13, 29]])

    d = Dense(N=100, unique=True, ordered=True)
    data_d = typed.List([types.int64(e) for e in range(100)])

    out = Compressed(
        full=True,
        ordered=True,
        unique=True,
        pos=typed.List.empty_list(types.int64),
        crd=typed.List.empty_list(types.int64),
    )
    data_out = typed.List.empty_list(types.int64)
    data_expected = typed.List([7 * 0, 13 * 7, 29 * 39])

    mul_sparse_dense_1d(c, data_c, d, data_d, out, data_out)
    assert c.pos == out.pos
    assert c.crd == out.crd
    assert data_expected == data_out


def insert_coords(levels: Tuple[Dense, Compressed], coords: List[Tuple[int, int]]):
    d, c = levels
    coords.sort()
    mapped = map_coords(coords)
    d.insert_init(1, 10)
    c.append_init(10, 70)
    p1 = 0
    for i in mapped.items():
        i0 = i[0]
        p0, f0 = d.coord_access(0, (i0,))
        p1begin = p1
        d.insert_coord(p0, i0)
        for i1 in i[1]:
            c.append_coord(p1, i1)
            p1 += 1
        c.append_edges(p0, p1begin, p1)
    d.insert_finalize(1, 10)
    c.append_finalize(10, 70)


def iter_coords(levels: Tuple[Dense, Compressed]):
    l = []
    d, c = levels
    for i0 in d.coord_iter(()):
        p0, f0 = d.coord_access(0, (i0,))
        if f0:
            l.append((i0, []))
            for p1 in c.pos_iter(p0):
                i1, f1 = c.pos_access(p1, (i0,))
                if f1:
                    l[-1][1].append(i1)

    return l


def make_dense(a, shape):
    out = np.zeros(shape, dtype=np.bool_)
    if not isinstance(a, tuple):
        out[a] = True
        return out

    for tup in a:
        for i in tup[1]:
            a[i] = make_dense(a[tup[1]])

    return out


def coords_from_bool_array(arr):
    l = []
    for i, arr_i in enumerate(arr):
        if np.ndim(arr_i) == 0:
            return i
        if np.any(arr_i):
            l.append((i, coords_from_bool_array(arr_i)))
    return l


def bool_array_from_coords(l, shape, out=None):
    if out is None:
        out = np.zeros(shape, dtype=np.bool_)

    if not isinstance(l, list):
        out[()] = True

    for i, l_i in l:
        bool_array_from_coords(l_i, shape[1:], out=out[i])

    return out


def map_coords(coords):
    d = {}
    for (x, y) in coords:
        if x not in d:
            d[x] = []
        d[x].append(y)
    return d


def test_insert_csr():
    # shape: (10, 7)
    arr = np.zeros((10, 7), dtype=np.bool_)
    coords = [(5, 3), (5, 1), (2, 3), (6, 4), (6, 0)]
    d = Dense(N=10)
    c = Compressed(pos=[], crd=[])
    levels = (d, c)

    insert_coords(levels, coords)
    expected = [
        (0, []),
        (1, []),
        (2, [3]),
        (3, []),
        (4, []),
        (5, [1, 3]),
        (6, [0, 4]),
        (7, []),
        (8, []),
        (9, []),
    ]
    got = iter_coords(levels)
    assert expected == got
