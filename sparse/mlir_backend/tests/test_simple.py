import math
import typing
from collections.abc import Iterable

import sparse

import pytest

import numpy as np
import scipy.sparse as sps

parametrize_dtypes = pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ],
)


def parametrize_scipy_fmt_with_arg(name: str) -> pytest.MarkDecorator:
    return pytest.mark.parametrize(
        name,
        ["csr", "csc", "coo"],
    )


parametrize_scipy_fmt = parametrize_scipy_fmt_with_arg("format")


def assert_sps_equal(
    expected: sps.csr_array | sps.csc_array | sps.coo_array,
    actual: sps.csr_array | sps.csc_array | sps.coo_array,
    /,
    *,
    check_canonical=False,
    check_dtype=True,
) -> None:
    assert expected.shape == actual.shape
    assert expected.format == actual.format

    if check_dtype:
        assert expected.dtype == actual.dtype

    if check_canonical:
        expected.eliminate_zeros()
        expected.sum_duplicates()

        actual.eliminate_zeros()
        actual.sum_duplicates()

    if expected.format != "coo":
        np.testing.assert_array_equal(expected.indptr, actual.indptr)
        np.testing.assert_array_equal(expected.indices, actual.indices)
    else:
        np.testing.assert_array_equal(expected.row, actual.row)
        np.testing.assert_array_equal(expected.col, actual.col)

    np.testing.assert_array_equal(expected.data, actual.data)


def generate_sampler(dtype: np.dtype, rng: np.random.Generator) -> typing.Callable[[tuple[int, ...]], np.ndarray]:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.signedinteger):

        def sampler_signed(size: tuple[int, ...]):
            return rng.integers(-10, 10, dtype=dtype, endpoint=True, size=size)

        return sampler_signed

    if np.issubdtype(dtype, np.unsignedinteger):

        def sampler_unsigned(size: tuple[int, ...]):
            return rng.integers(0, 10, dtype=dtype, endpoint=True, size=size)

        return sampler_unsigned

    if np.issubdtype(dtype, np.floating):

        def sampler_real_floating(size: tuple[int, ...]):
            return -10 + 20 * rng.random(dtype=dtype, size=size)

        return sampler_real_floating

    if np.issubdtype(dtype, np.complexfloating):
        float_dtype = np.array(0, dtype=dtype).real.dtype

        def sampler_complex_floating(size: tuple[int, ...]):
            real_sampler = generate_sampler(float_dtype, rng)
            if not isinstance(size, Iterable):
                size = (size,)
            float_arr = real_sampler(tuple(size) + (2,))
            return float_arr.view(dtype)[..., 0]

        return sampler_complex_floating

    raise NotImplementedError(f"{dtype=} not yet supported.")


def get_example_csf_arrays(dtype: np.dtype) -> tuple:
    pos_1 = np.array([0, 1, 3], dtype=np.int64)
    crd_1 = np.array([1, 0, 1], dtype=np.int64)
    pos_2 = np.array([0, 3, 5, 7], dtype=np.int64)
    crd_2 = np.array([0, 1, 3, 0, 3, 0, 1], dtype=np.int64)
    data = np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype)
    return pos_1, crd_1, pos_2, crd_2, data


@parametrize_dtypes
@pytest.mark.parametrize("shape", [(100,), (10, 200), (5, 10, 20)])
def test_dense_format(dtype, shape):
    data = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
    tensor = sparse.asarray(data)
    actual = sparse.to_numpy(tensor)
    np.testing.assert_equal(actual, data)


def assert_array_equal(
    expected: sparse.Array,
    actual: sparse.Array,
    /,
    *,
    same_format: bool = True,
    same_dtype: bool = True,
    data_test_fn: typing.Callable[[np.ndarray, np.ndarray], None] = np.testing.assert_array_equal,
) -> None:
    if same_format:
        assert expected.format == actual.format

    if same_dtype:
        assert expected.dtype == actual.dtype

    assert expected.shape == actual.shape
    actual = actual.asformat(expected.format)

    carrs_expected = expected.get_constituent_arrays()
    carrs_actual = actual.get_constituent_arrays()

    for e, a in zip(carrs_expected[:-1], carrs_actual[:-1], strict=True):
        assert e.dtype == a.dtype
        np.testing.assert_equal(e, a)

    data_test_fn(carrs_expected[-1], carrs_actual[-1])


@parametrize_dtypes
@parametrize_scipy_fmt
def test_roundtrip(rng, dtype, format):
    SHAPE = (80, 100)
    DENSITY = 0.6
    sampler = generate_sampler(dtype, rng)
    sps_arr = sps.random_array(
        SHAPE, density=DENSITY, format=format, dtype=dtype, random_state=rng, data_sampler=sampler
    )

    sp_arr = sparse.asarray(sps_arr)
    sps_roundtripped = sparse.to_scipy(sp_arr)
    assert_sps_equal(sps_arr, sps_roundtripped)

    sp_arr_roundtripped = sparse.asarray(sps_roundtripped)

    assert_array_equal(sp_arr, sp_arr_roundtripped)


@parametrize_dtypes
@pytest.mark.parametrize("shape", [(80, 100), (200,), (10, 20, 30)])
def test_roundtrip_dense(rng, dtype, shape):
    sampler = generate_sampler(dtype, rng)
    np_arr = sampler(shape)

    sp_arr = sparse.asarray(np_arr)
    np_roundtripped = sparse.to_numpy(sp_arr)
    assert np_arr.dtype == np_roundtripped.dtype
    np.testing.assert_array_equal(np_arr, np_roundtripped)

    sp_arr_roundtripped = sparse.asarray(np_roundtripped)

    assert_array_equal(sp_arr, sp_arr_roundtripped)


@parametrize_dtypes
@parametrize_scipy_fmt_with_arg("format1")
@parametrize_scipy_fmt_with_arg("format2")
def test_add(rng, dtype, format1, format2):
    if format1 == "coo" or format2 == "coo":
        pytest.xfail(reason="https://github.com/llvm/llvm-project/issues/116012")

    SHAPE = (100, 50)
    DENSITY = 0.5
    sampler = generate_sampler(dtype, rng)
    sps_arr1 = sps.random_array(
        SHAPE, density=DENSITY, format=format1, dtype=dtype, random_state=rng, data_sampler=sampler
    )
    sps_arr2 = sps.random_array(
        SHAPE, density=DENSITY, format=format2, dtype=dtype, random_state=rng, data_sampler=sampler
    )

    sp_arr1 = sparse.asarray(sps_arr1)
    sp_arr2 = sparse.asarray(sps_arr2)

    expected = sps_arr1 + sps_arr2
    actual = sparse.add(sp_arr1, sp_arr2)
    actual_sps = sparse.to_scipy(actual.asformat(sparse.asarray(expected).format))

    assert_sps_equal(expected, actual_sps, check_canonical=True)


@parametrize_dtypes
@pytest.mark.parametrize("shape", [(80, 100), (200,), (10, 20, 30)])
def test_add_dense(rng, dtype, shape):
    sampler = generate_sampler(dtype, rng)
    np_arr1 = sampler(shape)
    np_arr2 = sampler(shape)

    sp_arr1 = sparse.asarray(np_arr1)
    sp_arr2 = sparse.asarray(np_arr2)

    expected = np_arr1 + np_arr2
    actual = sparse.add(sp_arr1, sp_arr2)
    actual_np = sparse.to_numpy(actual)

    np.testing.assert_array_equal(expected, actual_np)


@parametrize_dtypes
@parametrize_scipy_fmt
def test_add_dense_sparse(rng, dtype, format):
    if format == "coo":
        pytest.xfail(reason="https://github.com/llvm/llvm-project/issues/116012")
    sampler = generate_sampler(dtype, rng)

    SHAPE = (100, 50)
    DENSITY = 0.5

    np_arr1 = sampler(SHAPE)
    sps_arr2 = sps.random_array(
        SHAPE, density=DENSITY, format=format, dtype=dtype, random_state=rng, data_sampler=sampler
    )

    sp_arr1 = sparse.asarray(np_arr1)
    sp_arr2 = sparse.asarray(sps_arr2)

    expected = np_arr1 + sps_arr2
    actual = sparse.add(sp_arr1, sp_arr2)
    actual_np = sparse.to_numpy(actual.asformat(sp_arr1.format))

    np.testing.assert_array_equal(expected, actual_np)


@parametrize_dtypes
def test_csf_format(dtype):
    format = sparse.formats.Csf().with_ndim(3).with_dtype(dtype).build()

    SHAPE = (2, 2, 4)
    pos_1, crd_1, pos_2, crd_2, data = get_example_csf_arrays(dtype)
    constituent_arrays = (pos_1, crd_1, pos_2, crd_2, data)

    csf_array = sparse.from_constituent_arrays(format=format, arrays=constituent_arrays, shape=SHAPE)
    result_arrays = csf_array.get_constituent_arrays()
    for actual, expected in zip(result_arrays, constituent_arrays, strict=True):
        np.testing.assert_array_equal(actual, expected)

    actual = sparse.add(csf_array, csf_array)
    expected = sparse.from_constituent_arrays(format=format, arrays=(pos_1, crd_1, pos_2, crd_2, data * 2), shape=SHAPE)
    assert_array_equal(expected, actual)


@parametrize_dtypes
def test_coo_3d_format(dtype):
    format = sparse.formats.Coo().with_ndim(3).with_dtype(dtype).build()

    SHAPE = (2, 2, 4)
    pos = np.array([0, 7])
    crd = [np.array([0, 1, 0, 0, 1, 1, 0]), np.array([1, 3, 1, 0, 0, 1, 0]), np.array([3, 1, 1, 0, 1, 1, 1])]
    data = np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype)
    carrs = (pos, *crd, data)

    coo_array = sparse.from_constituent_arrays(format=format, arrays=carrs, shape=SHAPE)
    result = coo_array.get_constituent_arrays()
    for actual, expected in zip(result, carrs, strict=True):
        np.testing.assert_array_equal(actual, expected)

    actual = sparse.add(coo_array, coo_array).asformat(coo_array.format)
    expected = sparse.from_constituent_arrays(format=actual.format, arrays=(pos, *crd, data * 2), shape=SHAPE)
    assert_array_equal(expected, actual)


@parametrize_dtypes
def test_sparse_vector_format(dtype):
    if sparse.asdtype(dtype) in {sparse.complex64, sparse.complex128}:
        pytest.xfail("The sparse_vector format returns incorrect results for complex dtypes.")
    format = sparse.formats.Coo().with_ndim(1).with_dtype(dtype).build()

    SHAPE = (10,)
    pos = np.array([0, 6])
    crd = np.array([0, 1, 2, 6, 8, 9])
    data = np.array([1, 2, 3, 4, 5, 6], dtype=dtype)
    carrs = (pos, crd, data)

    sv_array = sparse.from_constituent_arrays(format=format, arrays=carrs, shape=SHAPE)
    result = sv_array.get_constituent_arrays()
    for actual, expected in zip(result, carrs, strict=True):
        np.testing.assert_array_equal(actual, expected)

    actual = sparse.add(sv_array, sv_array)
    expected = sparse.from_constituent_arrays(format=actual.format, arrays=(pos, crd, data * 2), shape=SHAPE)
    assert_array_equal(expected, actual)

    dense = np.array([1, 2, 3, 0, 0, 0, 4, 0, 5, 6], dtype=dtype)
    dense_array = sparse.asarray(dense)
    res = sparse.to_numpy(sparse.add(dense_array, sv_array))
    np.testing.assert_array_equal(res, dense * 2)


def test_copy():
    arr_np_orig = np.arange(25).reshape((5, 5))
    arr_np_copy = arr_np_orig.copy()

    arr_sp1 = sparse.asarray(arr_np_copy, copy=True)
    arr_sp2 = sparse.asarray(arr_np_copy, copy=False).copy()
    arr_sp3 = sparse.asarray(arr_np_copy, copy=False)
    arr_np_copy[2, 2] = 42

    np.testing.assert_array_equal(sparse.to_numpy(arr_sp1), arr_np_orig)
    np.testing.assert_array_equal(sparse.to_numpy(arr_sp2), arr_np_orig)
    np.testing.assert_array_equal(sparse.to_numpy(arr_sp3), arr_np_copy)


@parametrize_dtypes
@pytest.mark.parametrize(
    "format",
    [
        "csr",
        pytest.param("csc", marks=pytest.mark.xfail(reason="https://github.com/llvm/llvm-project/pull/109641")),
        "coo",
    ],
)
@pytest.mark.parametrize(
    ("shape", "new_shape"),
    [
        ((100, 50), (25, 200)),
        ((100, 50), (10, 500, 1)),
        ((80, 1), (8, 10)),
        ((80, 1), (80,)),
    ],
)
def test_reshape(rng, dtype, format, shape, new_shape):
    DENSITY = 0.5
    sampler = generate_sampler(dtype, rng)

    arr_sps = sps.random_array(
        shape, density=DENSITY, format=format, dtype=dtype, random_state=rng, data_sampler=sampler
    )
    arr_sps.eliminate_zeros()
    arr_sps.sum_duplicates()
    arr = sparse.asarray(arr_sps)

    actual = sparse.reshape(arr, shape=new_shape)
    assert actual.shape == new_shape

    try:
        scipy_format = sparse.to_scipy(actual).format
    except RuntimeError:
        tmp_fmt = sparse.formats.Dense().with_ndim(arr.ndim).with_dtype(dtype).build()
        arr_dense = arr.asformat(tmp_fmt)
        arr_np = sparse.to_numpy(arr_dense)
        expected_np = arr_np.reshape(new_shape)

        out_fmt = sparse.formats.Dense().with_ndim(expected_np.ndim).with_dtype(dtype).build()
        actual_dense = actual.asformat(out_fmt)
        actual_np = sparse.to_numpy(actual_dense)

        np.testing.assert_array_equal(expected_np, actual_np)
        return

    expected = sparse.asarray(arr_sps.reshape(new_shape).asformat(scipy_format))

    for x, y in zip(expected.get_constituent_arrays(), actual.get_constituent_arrays(), strict=True):
        np.testing.assert_array_equal(x, y)


@parametrize_dtypes
def test_reshape_csf(dtype):
    # CSF
    csf_shape = (2, 2, 4)
    csf_format = sparse.formats.Csf().with_ndim(3).with_dtype(dtype).build()
    for shape, new_shape, expected_arrs in [
        (
            csf_shape,
            (4, 4, 1),
            [
                np.array([0, 0, 3, 5, 7]),
                np.array([0, 1, 3, 0, 3, 0, 1]),
                np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                np.array([0, 0, 0, 0, 0, 0, 0]),
                np.array([1, 2, 3, 4, 5, 6, 7]),
            ],
        ),
        (
            csf_shape,
            (2, 1, 8),
            [
                np.array([0, 1, 2]),
                np.array([0, 0]),
                np.array([0, 3, 7]),
                np.array([4, 5, 7, 0, 3, 4, 5]),
                np.array([1, 2, 3, 4, 5, 6, 7]),
            ],
        ),
    ]:
        arrs = get_example_csf_arrays(dtype)
        csf_tensor = sparse.from_constituent_arrays(format=csf_format, arrays=arrs, shape=shape)

        result = sparse.reshape(csf_tensor, shape=new_shape)
        for actual, expected in zip(result.get_constituent_arrays(), expected_arrs, strict=True):
            np.testing.assert_array_equal(actual, expected)


@parametrize_dtypes
def test_reshape_dense(dtype):
    SHAPE = (2, 2, 4)

    np_arr = np.arange(math.prod(SHAPE), dtype=dtype).reshape(SHAPE)
    sp_arr = sparse.asarray(np_arr)

    for new_shape in [
        (4, 4, 1),
        (2, 1, 8),
    ]:
        expected = np_arr.reshape(new_shape)
        actual = sparse.reshape(sp_arr, new_shape)

        actual_np = sparse.to_numpy(actual)

        assert actual_np.dtype == expected.dtype
        np.testing.assert_equal(actual_np, expected)


@pytest.mark.parametrize("src_fmt", ["csr", "csc", "coo"])
@pytest.mark.parametrize("dst_fmt", ["csr", "csc", "coo"])
def test_asformat(rng, src_fmt, dst_fmt):
    if "coo" in {src_fmt, dst_fmt}:
        pytest.xfail(reason="https://github.com/llvm/llvm-project/issues/116012")
    SHAPE = (100, 50)
    DENSITY = 0.5
    sampler = generate_sampler(np.float64, rng)

    sps_arr = sps.random_array(
        SHAPE, density=DENSITY, format=src_fmt, dtype=np.float64, random_state=rng, data_sampler=sampler
    )
    sp_arr = sparse.asarray(sps_arr)

    expected = sps_arr.asformat(dst_fmt)

    actual_fmt = sparse.asarray(expected, copy=False).format
    actual = sp_arr.asformat(actual_fmt)
    actual_sps = sparse.to_scipy(actual)

    assert actual_sps.format == dst_fmt
    assert_sps_equal(expected, actual_sps)
