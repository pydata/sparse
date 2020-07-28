from sparse._meta.dense_level import Dense
from sparse._meta.compressed_level import Compressed
from sparse._meta.format import Format, Tensor

# 1d formats
dense = Format(name="dense", levels=(Dense,))
sparse = Format(name="sparse", levels=(Compressed,))

# 2d formats
csr = Format(name="csr", levels=(Dense, Compressed))
dcsr = Format(name="dcsr", levels=(Compressed, Compressed))

# 3d formats
csf = Format(name="csf", levels=(Compressed, Compressed, Compressed))
cdc = Format(name="cdc", levels=(Compressed, Dense, Compressed))

coords_1d = [(0,), (1,), (4,), (6,)]
data_1d = (5, 1, 2, 8)
shape_1d = (8,)

coords_2d = [(0, 0), (0, 1), (1, 0), (1, 1), (3, 0), (3, 3), (3, 4)]
data_2d = data = (5, 1, 7, 3, 8, 4, 9)
shape_2d = (4, 6)

coords_3d = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 2, 1),
    (2, 0, 1),
    (2, 2, 0),
    (2, 2, 1),
    (2, 3, 0),
    (2, 3, 1),
]
data_3d = (1, 7, 5, 2, 4, 8, 3, 9)
shape_3d = (3, 4, 2)


def test_sparse_insert():
    t = Tensor(shape=shape_1d, fmt=sparse)
    t.insert_data(coords=coords_1d, data=data_1d)

    assert t._levels[0].pos == [0, 4]
    assert t._levels[0].crd == [0, 1, 4, 6]


def test_csr_insert():
    t = Tensor(shape=shape_2d, fmt=csr)
    t.insert_data(coords=coords_2d, data=data_2d)

    assert t._levels[1].pos == [0, 2, 4, 4, 7]
    assert t._levels[1].crd == [0, 1, 0, 1, 0, 3, 4]


def test_dcsr_insert():
    t = Tensor(shape=shape_2d, fmt=dcsr)
    t.insert_data(coords=coords_2d, data=data_2d)

    assert t._levels[0].pos == [0, 3]
    assert t._levels[0].crd == [0, 1, 3]

    assert t._levels[1].pos == [0, 2, 4, 7]
    assert t._levels[1].crd == [0, 1, 0, 1, 0, 3, 4]


def test_csf_insert():
    t = Tensor(shape=shape_3d, fmt=csf)
    t.insert_data(coords=coords_3d, data=data_3d)

    assert t._levels[0].pos == [0, 2]
    assert t._levels[0].crd == [0, 2]

    assert t._levels[1].pos == [0, 2, 5]
    assert t._levels[1].crd == [0, 2, 0, 2, 3]

    assert t._levels[2].pos == [0, 2, 3, 4, 6, 8]
    assert t._levels[2].crd == [0, 1, 1, 1, 0, 1, 0, 1]


def test_cdc_insert():
    t = Tensor(shape=shape_3d, fmt=cdc)
    t.insert_data(coords=coords_3d, data=data_3d)

    assert t._levels[0].pos == [0, 2]
    assert t._levels[0].crd == [0, 2]

    assert t._levels[2].pos == [0, 2, 2, 3, 3, 4, 4, 6, 8]
    assert t._levels[2].crd == [0, 1, 1, 1, 0, 1, 0, 1]
