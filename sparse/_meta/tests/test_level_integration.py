from numba import njit, types
from numba import typed

from sparse._meta.compressed_level import Compressed
from sparse._meta.dense_level import Dense
from typing import Tuple


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
