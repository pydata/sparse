import sparse
import pytest
from hypothesis import settings, given, strategies as st
from _utils import gen_transpose, gen_reductions
import numpy as np
import scipy

from sparse._compressed import GCXS
from sparse._utils import assert_eq


@pytest.fixture(scope="module", params=["f8", "f4", "i8", "i4"])
def random_sparse(request):
    dtype = request.param
    if np.issubdtype(dtype, np.integer):

        def data_rvs(n):
            return np.random.randint(-1000, 1000, n)

    else:
        data_rvs = None
    return sparse.random(
        (20, 30, 40), density=0.25, format="gcxs", data_rvs=data_rvs
    ).astype(dtype)


@pytest.fixture(scope="module", params=["f8", "f4", "i8", "i4"])
def random_sparse_small(request):
    dtype = request.param
    if np.issubdtype(dtype, np.integer):

        def data_rvs(n):
            return np.random.randint(-10, 10, n)

    else:
        data_rvs = None
    return sparse.random(
        (20, 30, 40), density=0.25, format="gcxs", data_rvs=data_rvs
    ).astype(dtype)


@settings(deadline=None)
@given(p=gen_reductions())
def test_reductions(p, random_sparse):
    reduction, kwargs = p
    x = random_sparse
    y = x.todense()
    xx = getattr(x, reduction)(**kwargs)
    yy = getattr(y, reduction)(**kwargs)
    assert_eq(xx, yy)


@pytest.mark.xfail(
    reason=("Setting output dtype=float16 produces results " "inconsistent with numpy")
)
@pytest.mark.filterwarnings("ignore:overflow")
@given(
    reduction=st.sampled_from(["sum", "mean"]),
    kwargs=st.sampled_from([{"dtype": np.float16}]),
    axis=st.sampled_from([None, 0, 1, 2, (0, 2)]),
)
def test_reductions_float16(random_sparse, reduction, kwargs, axis):
    x = random_sparse
    y = x.todense()
    xx = getattr(x, reduction)(axis=axis, **kwargs)
    yy = getattr(y, reduction)(axis=axis, **kwargs)
    assert_eq(xx, yy, atol=1e-2)


@settings(deadline=None)
@given(
    reduction=st.sampled_from(["any", "all"]),
    kwargs=st.sampled_from([{}]),
    axis=st.sampled_from([None, 0, 1, 2, (0, 2), -3, (1, -1)]),
    keepdims=st.sampled_from([True, False]),
)
def test_reductions_bool(random_sparse, reduction, kwargs, axis, keepdims):
    y = np.zeros((2, 3, 4), dtype=bool)
    y[0] = True
    y[1, 1, 1] = True
    x = sparse.COO.from_numpy(y)
    xx = getattr(x, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    yy = getattr(y, reduction)(axis=axis, keepdims=keepdims, **kwargs)
    assert_eq(xx, yy)


@given(p=gen_reductions(function=True))
def test_ufunc_reductions(random_sparse, p):
    reduction, kwargs = p
    x = random_sparse
    y = x.todense()
    xx = reduction(x, **kwargs)
    yy = reduction(y, **kwargs)
    assert_eq(xx, yy)
    # If not a scalar/1 element array, must be a sparse array
    if xx.size > 1:
        assert isinstance(xx, GCXS)


@settings(deadline=None)
@given(
    reduction=st.sampled_from([np.sum, np.max, np.prod, np.minimum.reduce]),
    kwargs=st.sampled_from([{}, {"axis": 0}, {"keepdims": True}]),
    fill_value=st.sampled_from([0, 1.0, -1, -2.2, 5.0]),
)
def test_ufunc_reductions_kwargs(reduction, kwargs, fill_value):
    x = sparse.random((2, 3, 4), density=0.5, format="gcxs", fill_value=fill_value)
    y = x.todense()
    xx = reduction(x, **kwargs)
    yy = reduction(y, **kwargs)
    assert_eq(xx, yy)
    # If not a scalar/1 element array, must be a sparse array
    if xx.size > 1:
        assert isinstance(xx, GCXS)


@pytest.mark.parametrize(
    "a,b",
    [
        [(3, 4), (3, 4)],
        [(12,), (3, 4)],
        [(12,), (3, -1)],
        [(3, 4), (12,)],
        [(3, 4), (-1, 4)],
        [(3, 4), (3, -1)],
        [(2, 3, 4, 5), (8, 15)],
        [(2, 3, 4, 5), (24, 5)],
        [(2, 3, 4, 5), (20, 6)],
        [(), ()],
    ],
)
def test_reshape(a, b):
    s = sparse.random(a, density=0.5, format="gcxs")
    x = s.todense()

    assert_eq(x.reshape(b), s.reshape(b))


def test_reshape_same():

    s = sparse.random((3, 5), density=0.5, format="gcxs")
    assert s.reshape(s.shape) is s


@settings(deadline=None)
@given(ab=gen_transpose())
def test_tranpose(ab):
    a, b = ab
    s = sparse.random(a, density=0.5, format="gcxs")
    x = s.todense()

    assert_eq(x.transpose(b), s.transpose(b))


def test_to_scipy_sparse():
    s = sparse.random((3, 5), density=0.5, format="gcxs", compressed_axes=(0,))
    a = s.to_scipy_sparse()
    b = scipy.sparse.csr_matrix(s.todense())

    assert_eq(a, b)

    s = sparse.random((3, 5), density=0.5, format="gcxs", compressed_axes=(1,))
    a = s.to_scipy_sparse()
    b = scipy.sparse.csc_matrix(s.todense())

    assert_eq(a, b)


def test_tocoo():
    coo = sparse.random((5, 6), density=0.5)
    b = GCXS.from_coo(coo)
    assert_eq(b.tocoo(), coo)


@given(complex=st.sampled_from([True, False]))
def test_complex_methods(complex):
    if complex:
        x = np.array([1 + 2j, 2 - 1j, 0, 1, 0])
    else:
        x = np.array([1, 2, 0, 0, 0])
    s = GCXS.from_numpy(x)
    assert_eq(s.imag, x.imag)
    assert_eq(s.real, x.real)
    assert_eq(s.conj(), x.conj())


@settings(deadline=None)
@given(
    index=st.sampled_from(
        [
            # Integer
            0,
            1,
            -1,
            (1, 1, 1),
            # Pure slices
            (slice(0, 2),),
            (slice(None, 2), slice(None, 2)),
            (slice(1, None), slice(1, None)),
            (slice(None, None),),
            (slice(None, None, -1),),
            (slice(None, 2, -1), slice(None, 2, -1)),
            (slice(1, None, 2), slice(1, None, 2)),
            (slice(None, None, 2),),
            (slice(None, 2, -1), slice(None, 2, -2)),
            (slice(1, None, 2), slice(1, None, 1)),
            (slice(None, None, -2),),
            # Combinations
            (0, slice(0, 2)),
            (slice(0, 1), 0),
            (None, slice(1, 3), 0),
            (slice(0, 3), None, 0),
            (slice(1, 2), slice(2, 4)),
            (slice(1, 2), slice(None, None)),
            (slice(1, 2), slice(None, None), 2),
            (slice(1, 2, 2), slice(None, None), 2),
            (slice(1, 2, None), slice(None, None, 2), 2),
            (slice(1, 2, -2), slice(None, None), -2),
            (slice(1, 2, None), slice(None, None, -2), 2),
            (slice(1, 2, -1), slice(None, None), -1),
            (slice(1, 2, None), slice(None, None, -1), 2),
            (slice(2, 0, -1), slice(None, None), -1),
            (slice(-2, None, None),),
            (slice(-1, None, None), slice(-2, None, None)),
            # With ellipsis
            (Ellipsis, slice(1, 3)),
            (1, Ellipsis, slice(1, 3)),
            (slice(0, 1), Ellipsis),
            (Ellipsis, None),
            (None, Ellipsis),
            (1, Ellipsis),
            (1, Ellipsis, None),
            (1, 1, 1, Ellipsis),
            (Ellipsis, 1, None),
            # Pathological - Slices larger than array
            (slice(None, 1000)),
            (slice(None), slice(None, 1000)),
            (slice(None), slice(1000, -1000, -1)),
            (slice(None), slice(1000, -1000, -50)),
            # Pathological - Wrong ordering of start/stop
            (slice(5, 0),),
            (slice(0, 5, -1),),
        ]
    ),
    compressed_axes=st.sampled_from([(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]),
)
def test_slicing(index, compressed_axes):
    s = sparse.random(
        (2, 3, 4), density=0.5, format="gcxs", compressed_axes=compressed_axes
    )
    x = s.todense()
    assert_eq(x[index], s[index])


@settings(deadline=None)
@given(
    index=st.sampled_from(
        [
            ([1, 0], 0),
            (1, [0, 2]),
            (0, [1, 0], 0),
            (1, [2, 0], 0),
            ([True, False], slice(1, None), slice(-2, None)),
            (slice(1, None), slice(-2, None), [True, False, True, False]),
            ([1, 0],),
            (Ellipsis, [2, 1, 3]),
            (slice(None), [2, 1, 2]),
            (1, [2, 0, 1]),
        ]
    ),
    compressed_axes=st.sampled_from([(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]),
)
def test_advanced_indexing(index, compressed_axes):
    s = sparse.random(
        (2, 3, 4), density=0.5, format="gcxs", compressed_axes=compressed_axes
    )
    x = s.todense()

    assert_eq(x[index], s[index])


@given(
    index=st.sampled_from(
        [
            (Ellipsis, Ellipsis),
            (1, 1, 1, 1),
            (slice(None),) * 4,
            5,
            -5,
            "foo",
            [True, False, False],
            0.5,
            [0.5],
            {"potato": "kartoffel"},
            ([[0, 1]],),
        ]
    )
)
def test_slicing_errors(index):
    s = sparse.random((2, 3, 4), density=0.5, format="gcxs")

    with pytest.raises(IndexError):
        s[index]


def test_change_compressed_axes():
    coo = sparse.random((3, 4, 5), density=0.5)
    s = GCXS.from_coo(coo, compressed_axes=(0, 1))
    b = GCXS.from_coo(coo, compressed_axes=(1, 2))
    assert_eq(s, b)
    s.change_compressed_axes((1, 2))
    assert_eq(s, b)


def test_concatenate():
    xx = sparse.random((2, 3, 4), density=0.5, format="gcxs")
    x = xx.todense()
    yy = sparse.random((5, 3, 4), density=0.5, format="gcxs")
    y = yy.todense()
    zz = sparse.random((4, 3, 4), density=0.5, format="gcxs")
    z = zz.todense()

    assert_eq(
        np.concatenate([x, y, z], axis=0), sparse.concatenate([xx, yy, zz], axis=0)
    )

    xx = sparse.random((5, 3, 1), density=0.5, format="gcxs")
    x = xx.todense()
    yy = sparse.random((5, 3, 3), density=0.5, format="gcxs")
    y = yy.todense()
    zz = sparse.random((5, 3, 2), density=0.5, format="gcxs")
    z = zz.todense()

    assert_eq(
        np.concatenate([x, y, z], axis=2), sparse.concatenate([xx, yy, zz], axis=2)
    )

    assert_eq(
        np.concatenate([x, y, z], axis=-1), sparse.concatenate([xx, yy, zz], axis=-1)
    )


@given(
    axis=st.sampled_from([0, 1]),
    func=st.sampled_from([sparse.stack, sparse.concatenate]),
)
def test_concatenate_mixed(func, axis):
    s = sparse.random((10, 10), density=0.5, format="gcxs")
    d = s.todense()

    with pytest.raises(ValueError):
        func([d, s, s], axis=axis)


def test_concatenate_noarrays():
    with pytest.raises(ValueError):
        sparse.concatenate([])


@given(
    shape=st.sampled_from([(5,), (2, 3, 4), (5, 2)]), axis=st.sampled_from([0, 1, -1])
)
def test_stack(shape, axis):
    xx = sparse.random(shape, density=0.5, format="gcxs")
    x = xx.todense()
    yy = sparse.random(shape, density=0.5, format="gcxs")
    y = yy.todense()
    zz = sparse.random(shape, density=0.5, format="gcxs")
    z = zz.todense()

    assert_eq(np.stack([x, y, z], axis=axis), sparse.stack([xx, yy, zz], axis=axis))


@given(in_shape=st.sampled_from([(5, 5), 62, (3, 3, 3)]))
def test_flatten(in_shape):
    s = sparse.random(in_shape, format="gcxs", density=0.5)
    x = s.todense()

    a = s.flatten()
    e = x.flatten()

    assert_eq(e, a)


def test_gcxs_valerr():
    a = np.arange(300)
    with pytest.raises(ValueError):
        GCXS.from_numpy(a, idx_dtype=np.int8)


def test_upcast():
    a = sparse.random((50, 50, 50), density=0.1, format="coo", idx_dtype=np.uint8)
    b = a.asformat("gcxs")
    assert b.indices.dtype == np.uint16

    a = sparse.random((8, 7, 6), density=0.5, format="gcxs", idx_dtype=np.uint8)
    b = sparse.random((6, 6, 6), density=0.8, format="gcxs", idx_dtype=np.uint8)
    assert sparse.concatenate((a, a)).indptr.dtype == np.uint16
    assert sparse.stack((b, b)).indptr.dtype == np.uint16


def test_from_coo():
    a = sparse.random((5, 5, 5), density=0.1, format="coo")
    b = GCXS(a)
    assert_eq(a, b)


def test_from_coo_valerr():
    a = sparse.random((25, 25, 25), density=0.01, format="coo")
    with pytest.raises(ValueError):
        GCXS.from_coo(a, idx_dtype=np.int8)


@given(
    pad_width=st.sampled_from(
        [
            2,
            (2, 1),
            ((2), (1)),
            ((1, 2), (4, 5), (7, 8)),
        ]
    ),
    constant_values=st.sampled_from([0, 1, 150, np.nan]),
)
def test_pad_valid(pad_width, constant_values):
    y = sparse.random(
        (50, 50, 3), density=0.15, fill_value=constant_values, format="gcxs"
    )
    x = y.todense()
    xx = np.pad(x, pad_width=pad_width, constant_values=constant_values)
    yy = np.pad(y, pad_width=pad_width, constant_values=constant_values)
    assert_eq(xx, yy)


@given(
    pad_width=st.sampled_from([((2, 1), (5, 7))]),
    constant_values=st.sampled_from([150, 2, (1, 2)]),
    fill_value=st.floats(min_value=0, max_value=10),
)
def test_pad_invalid(pad_width, constant_values, fill_value):
    y = sparse.random((50, 50, 3), density=0.15, format="gcxs", fill_value=fill_value)
    with pytest.raises(ValueError):
        np.pad(y, pad_width, constant_values=constant_values)
