import pytest

import numpy as np

import sparse
from sparse import DOK
from sparse._utils import assert_eq


@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("density", [0.1, 0.3, 0.5, 0.7])
def test_random_shape_nnz(shape, density):
    s = sparse.random(shape, density, format="dok")

    assert isinstance(s, DOK)

    assert s.shape == shape
    expected_nnz = density * np.prod(shape)
    assert np.floor(expected_nnz) <= s.nnz <= np.ceil(expected_nnz)


def test_convert_to_coo():
    s1 = sparse.random((2, 3, 4), 0.5, format="dok")
    s2 = sparse.COO(s1)

    assert_eq(s1, s2)


def test_convert_from_coo():
    s1 = sparse.random((2, 3, 4), 0.5, format="coo")
    s2 = DOK(s1)

    assert_eq(s1, s2)


def test_convert_from_numpy():
    x = np.random.rand(2, 3, 4)
    s = DOK(x)

    assert_eq(x, s)


def test_convert_to_numpy():
    s = sparse.random((2, 3, 4), 0.5, format="dok")
    x = s.todense()

    assert_eq(x, s)


def test_convert_from_scipy_sparse():
    import scipy.sparse

    x = scipy.sparse.rand(6, 3, density=0.2)
    s = DOK(x)

    assert_eq(x, s)


@pytest.mark.parametrize(
    "shape, data",
    [
        (2, {0: 1}),
        ((2, 3), {(0, 1): 3, (1, 2): 4}),
        ((2, 3, 4), {(0, 1): 3, (1, 2, 3): 4, (1, 1): [6, 5, 4, 1]}),
    ],
)
def test_construct(shape, data):
    s = DOK(shape, data)
    x = np.zeros(shape, dtype=s.dtype)

    for c, d in data.items():
        x[c] = d

    assert_eq(x, s)


@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("density", [0.1, 0.3, 0.5, 0.7])
def test_getitem_single(shape, density):
    s = sparse.random(shape, density, format="dok")
    x = s.todense()

    for _ in range(s.nnz):
        idx = np.random.randint(np.prod(shape))
        idx = np.unravel_index(idx, shape)
        print(idx)

        assert np.isclose(s[idx], x[idx])


@pytest.mark.parametrize(
    "shape, density, indices",
    [
        ((2, 3), 0.5, (slice(1),)),
        ((5, 5), 0.2, (slice(0, 4, 2),)),
        ((10, 10), 0.2, (slice(5), slice(0, 10, 3))),
        ((5, 5), 0.5, (slice(0, 4, 4), slice(0, 4, 4))),
        ((5, 5), 0.4, (1, slice(0, 4, 1))),
        ((10, 10), 0.8, ([0, 4, 5], [3, 2, 4])),
        ((10, 10), 0, (slice(10), slice(10))),
    ],
)
def test_getitem(shape, density, indices):
    s = sparse.random(shape, density, format="dok")
    x = s.todense()

    sparse_sliced = s[indices]
    dense_sliced = x[indices]

    assert_eq(sparse_sliced.todense(), dense_sliced)


@pytest.mark.parametrize(
    "shape, density, indices",
    [
        ((10, 10), 0.8, ([0, 4, 5],)),
        ((5, 5, 5), 0.5, ([1, 2, 3], [0, 2, 2])),
    ],
)
def test_getitem_notimplemented_error(shape, density, indices):
    s = sparse.random(shape, density, format="dok")

    with pytest.raises(NotImplementedError):
        s[indices]


@pytest.mark.parametrize(
    "shape, density, indices",
    [
        ((10, 10), 0.8, ([0, 4, 5], [0, 2])),
        ((5, 5, 5), 0.5, ([1, 2, 3], [0], [2, 3, 4])),
        ((10,), 0.5, (5, 6)),
    ],
)
def test_getitem_index_error(shape, density, indices):
    s = sparse.random(shape, density, format="dok")

    with pytest.raises(IndexError):
        s[indices]


@pytest.mark.parametrize(
    "shape, index, value_shape",
    [
        ((2,), slice(None), ()),
        ((2,), slice(1, 2), ()),
        ((2,), slice(0, 2), (2,)),
        ((2,), 1, ()),
        ((2, 3), (0, slice(None)), ()),
        ((2, 3), (0, slice(1, 3)), ()),
        ((2, 3), (1, slice(None)), (3,)),
        ((2, 3), (0, slice(1, 3)), (2,)),
        ((2, 3), (0, slice(2, 0, -1)), (2,)),
        ((2, 3), (slice(None), 1), ()),
        ((2, 3), (slice(None), 1), (2,)),
        ((2, 3), (slice(1, 2), 1), ()),
        ((2, 3), (slice(1, 2), 1), (1,)),
        ((2, 3), (0, 2), ()),
        ((2, 3), ([0, 1], [1, 2]), (2,)),
        ((2, 3), ([0, 1], [1, 2]), ()),
        ((4,), ([1, 3]), ()),
    ],
)
def test_setitem(shape, index, value_shape):
    s = sparse.random(shape, 0.5, format="dok")
    x = s.todense()

    value = np.random.rand(*value_shape)

    s[index] = value
    x[index] = value

    assert_eq(x, s)


def test_setitem_delete():
    shape = (2, 3)
    index = [0, 1], [1, 2]
    value = 0
    s = sparse.random(shape, 1.0, format="dok")
    x = s.todense()

    s[index] = value
    x[index] = value

    assert_eq(x, s)
    assert s.nnz < s.size


@pytest.mark.parametrize(
    "shape, index, value_shape",
    [
        ((2, 3), ([0, 1.5], [1, 2]), ()),
        ((2, 3), ([0, 1], [1]), ()),
        ((2, 3), ([[0], [1]], [1, 2]), ()),
    ],
)
def test_setitem_index_error(shape, index, value_shape):
    s = sparse.random(shape, 0.5, format="dok")
    value = np.random.rand(*value_shape)

    with pytest.raises(IndexError):
        s[index] = value


@pytest.mark.parametrize(
    "shape, index, value_shape",
    [
        ((2, 3), ([0, 1],), ()),
    ],
)
def test_setitem_notimplemented_error(shape, index, value_shape):
    s = sparse.random(shape, 0.5, format="dok")
    value = np.random.rand(*value_shape)
    with pytest.raises(NotImplementedError):
        s[index] = value


@pytest.mark.parametrize(
    "shape, index, value_shape",
    [
        ((2, 3), ([0, 1], [1, 2]), (1, 2)),
        ((2, 3), ([0, 1], [1, 2]), (3,)),
        ((2,), 1, (2,)),
    ],
)
def test_setitem_value_error(shape, index, value_shape):
    s = sparse.random(shape, 0.5, format="dok")
    value = np.random.rand(*value_shape)

    with pytest.raises(ValueError):
        s[index] = value


def test_default_dtype():
    s = DOK((5,))

    assert s.dtype == np.float64


def test_int_dtype():
    data = {1: np.uint8(1), 2: np.uint16(2)}

    s = DOK((5,), data)

    assert s.dtype == np.uint16


def test_float_dtype():
    data = {1: np.uint8(1), 2: np.float32(2)}

    s = DOK((5,), data)

    assert s.dtype == np.float32


def test_set_zero():
    s = DOK((1,), dtype=np.uint8)
    s[0] = 1
    s[0] = 0

    assert s[0] == 0
    assert s.nnz == 0


@pytest.mark.parametrize("format", ["coo", "dok"])
def test_asformat(format):
    s = sparse.random((2, 3, 4), density=0.5, format="dok")
    s2 = s.asformat(format)

    assert_eq(s, s2)


def test_coo_fv_interface():
    s1 = sparse.full((5, 5), fill_value=1 + np.random.rand())
    s2 = sparse.DOK(s1)
    assert_eq(s1, s2)
    s3 = sparse.COO(s2)
    assert_eq(s1, s3)


def test_empty_dok_dtype():
    d = sparse.DOK(5, dtype=np.uint8)
    s = sparse.COO(d)
    assert s.dtype == d.dtype


def test_zeros_like():
    s = sparse.random((2, 3, 4), density=0.5)
    s2 = sparse.zeros_like(s, format="dok")
    assert s.shape == s2.shape
    assert s.dtype == s2.dtype
    assert isinstance(s2, sparse.DOK)


@pytest.mark.parametrize(
    "pad_width",
    [
        2,
        (2, 1),
        ((2), (1)),
        ((1, 2), (4, 5), (7, 8)),
    ],
)
@pytest.mark.parametrize("constant_values", [0, 1, 150, np.nan])
def test_pad_valid(pad_width, constant_values):
    y = sparse.random(
        (50, 50, 3), density=0.15, fill_value=constant_values, format="dok"
    )
    x = y.todense()
    xx = np.pad(x, pad_width=pad_width, constant_values=constant_values)
    yy = np.pad(y, pad_width=pad_width, constant_values=constant_values)
    assert_eq(xx, yy)


@pytest.mark.parametrize(
    "pad_width",
    [
        ((2, 1), (5, 7)),
    ],
)
@pytest.mark.parametrize("constant_values", [150, 2, (1, 2)])
def test_pad_invalid(pad_width, constant_values, fill_value=0):
    y = sparse.random((50, 50, 3), density=0.15, format="dok")
    with pytest.raises(ValueError):
        np.pad(y, pad_width, constant_values=constant_values)


@pytest.mark.parametrize("func", [np.concatenate, np.stack])
def test_dok_concat_stack(func):
    s1 = sparse.random((4, 4), density=0.25, format="dok")
    s2 = sparse.random((4, 4), density=0.25, format="dok")

    x1 = s1.todense()
    x2 = s2.todense()

    assert_eq(func([s1, s2]), func([x1, x2]))


def test_dok_indexing():
    s = sparse.DOK((3, 3))
    s[1, 2] = 0.5
    x = s.todense()
    assert_eq(x[1::-1], s[1::-1])
