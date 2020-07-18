import numpy as np

from numba import njit, types
from numba import typed

from sparse._meta.compressed_level import Compressed
from sparse._meta.dense_level import Dense
from sparse._meta.format import Format, Tensor, LazyTensor

from typing import Tuple, List
import collections


def test_csr_csr_mul():
    @njit
    def mul_csr_csr():
        pass

    shape = (100, 300)
    d1 = Dense(N=shape[0])
    c1 = Compressed(pos=[], crd=[])
    d2 = Dense(N=shape[0])
    coords = [(5, 3), (5, 1), (2, 3), (6, 4), (6, 0)]
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    # B = Tensor(shape=(10, 5), dims=csr)


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
    pbeginc = 0
    for p_c1 in pos_iter:
        i_c1, found_c1 = c.pos_access(p_c1, ())
        p_d1, found_d1 = d.locate(0, (i_c1,))
        if found_c1 and found_d1:
            out.append_coord(p_c1, i_c1)
            data_c = c_data[p_c1]
            data_d = d_data[p_d1]
            data_out.append(data_c * data_d)
    pendc = p_c1 + 1
    out.append_edges(0, pbeginc, pendc)
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
    mapped = map_coords(coords)
    sz0 = 1
    sz1 = d.size(sz0)  # 10
    d.insert_init(1, sz1)
    p1 = 0
    c.append_init(sz1, p1)
    for i in mapped.items():
        i0 = i[0]
        p0, f0 = d.coord_access(0, (i0,))
        p1begin = p1
        d.insert_coord(p0, i0)
        for i1 in i[1]:
            c.append_coord(p1, i1)
            p1 += 1
        c.append_edges(p0, p1begin, p1)
    d.insert_finalize(1, sz1)
    c.append_finalize(sz1, p1)


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


def map_coords(coords):
    d = {}
    for (x, y) in coords:
        if x not in d:
            d[x] = []
        d[x].append(y)
    return d


def test_insert_csr():
    def flatten(ret):
        got = []
        for (u, l) in ret:
            for v in l:
                got.append((u, v))
        return got

    # shape = (10, 7)
    coords = [(5, 3), (5, 1), (2, 3), (6, 4), (6, 0)]
    coords.sort()
    d = Dense(N=10)
    c = Compressed(pos=[], crd=[])
    levels = (d, c)

    insert_coords(levels, coords)
    got = flatten(iter_coords(levels))
    assert coords == got
