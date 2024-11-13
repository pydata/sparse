import math
import typing
from collections.abc import Iterable

import sparse

import pytest

import numpy as np
import scipy.sparse as sps

if sparse._BACKEND != sparse._BackendType.MLIR:
    pytest.skip("skipping MLIR tests", allow_module_level=True)

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


def assert_sps_equal(
    expected: sps.csr_array | sps.csc_array | sps.coo_array,
    actual: sps.csr_array | sps.csc_array | sps.coo_array,
) -> None:
    assert expected.format == actual.format
    expected.eliminate_zeros()
    expected.sum_duplicates()

    actual.eliminate_zeros()
    actual.sum_duplicates()

    if expected.format != "coo":
        np.testing.assert_array_equal(expected.indptr, actual.indptr)
        np.testing.assert_array_equal(expected.indices, actual.indices)
    else:
        np.testing.assert_array_equal(expected.row, actual.col)
        np.testing.assert_array_equal(expected.row, actual.col)

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


@parametrize_dtypes
def test_2d_constructors(rng, dtype):
    SHAPE = (80, 100)
    DENSITY = 0.6
    sampler = generate_sampler(dtype, rng)
    csr = sps.random_array(SHAPE, density=DENSITY, format="csr", dtype=dtype, random_state=rng, data_sampler=sampler)
    csc = sps.random_array(SHAPE, density=DENSITY, format="csc", dtype=dtype, random_state=rng, data_sampler=sampler)
    dense = np.arange(math.prod(SHAPE), dtype=dtype).reshape(SHAPE)
    coo = sps.random_array(SHAPE, density=DENSITY, format="coo", dtype=dtype, random_state=rng, data_sampler=sampler)
    coo.sum_duplicates()

    csr_tensor = sparse.asarray(csr)
    csc_tensor = sparse.asarray(csc)
    dense_tensor = sparse.asarray(dense)
    coo_tensor = sparse.asarray(coo)
    dense_2_tensor = sparse.asarray(np.arange(100, dtype=dtype).reshape((25, 4)) + 10)

    csr_retured = sparse.to_scipy(csr_tensor)
    assert_sps_equal(csr_retured, csr)

    csc_retured = sparse.to_scipy(csc_tensor)
    assert_sps_equal(csc_retured, csc)

    dense_returned = sparse.to_numpy(dense_tensor)
    np.testing.assert_equal(dense_returned, dense)

    coo_returned = sparse.to_scipy(coo_tensor)
    np.testing.assert_equal(coo_returned.todense(), coo.todense())

    dense_2_returned = sparse.to_numpy(dense_2_tensor)
    np.testing.assert_equal(dense_2_returned, np.arange(100, dtype=dtype).reshape((25, 4)) + 10)


@parametrize_dtypes
def test_add(rng, dtype):
    SHAPE = (100, 50)
    DENSITY = 0.5
    sampler = generate_sampler(dtype, rng)

    csr = sps.random_array(SHAPE, density=DENSITY, format="csr", dtype=dtype, random_state=rng, data_sampler=sampler)
    csr_2 = sps.random_array(SHAPE, density=DENSITY, format="csr", dtype=dtype, random_state=rng, data_sampler=sampler)
    csc = sps.random_array(SHAPE, density=DENSITY, format="csc", dtype=dtype, random_state=rng, data_sampler=sampler)
    dense = np.arange(math.prod(SHAPE), dtype=dtype).reshape(SHAPE)
    coo = sps.random_array(SHAPE, density=DENSITY, format="coo", dtype=dtype, random_state=rng)
    coo.sum_duplicates()

    csr_tensor = sparse.asarray(csr)
    csr_2_tensor = sparse.asarray(csr_2)
    csc_tensor = sparse.asarray(csc)
    dense_tensor = sparse.asarray(dense)
    coo_tensor = sparse.asarray(coo)

    actual = sparse.to_scipy(sparse.add(csr_tensor, csr_2_tensor))
    expected = csr + csr_2
    assert_sps_equal(expected, actual)

    actual = sparse.to_scipy(sparse.add(csc_tensor, csc_tensor))
    expected = csc + csc
    assert_sps_equal(expected, actual)

    actual = sparse.to_scipy(sparse.add(csc_tensor, csr_tensor))
    expected = (csc + csr).asformat("csr")
    assert_sps_equal(expected, actual)

    actual = sparse.to_numpy(sparse.add(csr_tensor, dense_tensor))
    expected = csr + dense
    np.testing.assert_array_equal(actual, expected)

    actual = sparse.to_numpy(sparse.add(dense_tensor, csr_tensor))
    expected = csr + dense
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected)

    actual = sparse.to_numpy(sparse.add(dense_tensor, dense_tensor))
    expected = dense + dense
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected)

    actual = sparse.to_scipy(sparse.add(csr_2_tensor, coo_tensor))
    expected = csr_2 + coo
    assert_sps_equal(expected, actual)

    # This ends up being DCSR, not COO
    actual_tensor = sparse.add(coo_tensor, coo_tensor)
    actual = sparse.to_scipy(actual_tensor.asformat(coo_tensor.format))
    expected = coo + coo
    np.testing.assert_array_equal(actual.todense(), expected.todense())


@parametrize_dtypes
def test_csf_format(dtype):
    format = sparse.levels.get_storage_format(
        levels=(
            sparse.levels.Level(sparse.levels.LevelFormat.Dense),
            sparse.levels.Level(sparse.levels.LevelFormat.Compressed),
            sparse.levels.Level(sparse.levels.LevelFormat.Compressed),
        ),
        order="C",
        pos_width=64,
        crd_width=64,
        dtype=sparse.asdtype(dtype),
    )

    SHAPE = (2, 2, 4)
    pos_1, crd_1, pos_2, crd_2, data = get_example_csf_arrays(dtype)
    constituent_arrays = (pos_1, crd_1, pos_2, crd_2, data)

    csf_array = sparse.from_constituent_arrays(format=format, arrays=constituent_arrays, shape=SHAPE)
    result_arrays = csf_array.get_constituent_arrays()
    for actual, expected in zip(result_arrays, constituent_arrays, strict=True):
        np.testing.assert_array_equal(actual, expected)

    res_arrays = sparse.add(csf_array, csf_array).get_constituent_arrays()
    expected_arrays = (pos_1, crd_1, pos_2, crd_2, data * 2)
    for actual, expected in zip(res_arrays, expected_arrays, strict=True):
        np.testing.assert_array_equal(actual, expected)


@parametrize_dtypes
def test_coo_3d_format(dtype):
    format = sparse.levels.get_storage_format(
        levels=(
            sparse.levels.Level(sparse.levels.LevelFormat.Compressed, sparse.levels.LevelProperties.NonUnique),
            sparse.levels.Level(
                sparse.levels.LevelFormat.Singleton,
                sparse.levels.LevelProperties.NonUnique | sparse.levels.LevelProperties.SOA,
            ),
            sparse.levels.Level(sparse.levels.LevelFormat.Singleton, sparse.levels.LevelProperties.SOA),
        ),
        order="C",
        pos_width=64,
        crd_width=64,
        dtype=sparse.asdtype(dtype),
    )

    SHAPE = (2, 2, 4)
    pos = np.array([0, 7])
    crd = [np.array([0, 1, 0, 0, 1, 1, 0]), np.array([1, 3, 1, 0, 0, 1, 0]), np.array([3, 1, 1, 0, 1, 1, 1])]
    data = np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype)
    carrs = (pos, *crd, data)

    coo_array = sparse.from_constituent_arrays(format=format, arrays=carrs, shape=SHAPE)
    result = coo_array.get_constituent_arrays()
    for actual, expected in zip(result, carrs, strict=True):
        np.testing.assert_array_equal(actual, expected)

    result_arrays = sparse.add(coo_array, coo_array).asformat(coo_array.format).get_constituent_arrays()
    constituent_arrays = (pos, *crd, data * 2)
    for actual, expected in zip(result_arrays, constituent_arrays, strict=False):
        np.testing.assert_array_equal(actual, expected)


@parametrize_dtypes
def test_sparse_vector_format(dtype):
    format = sparse.levels.get_storage_format(
        levels=(sparse.levels.Level(sparse.levels.LevelFormat.Compressed),),
        order="C",
        pos_width=64,
        crd_width=64,
        dtype=sparse.asdtype(dtype),
    )

    SHAPE = (10,)
    pos = np.array([0, 6])
    crd = np.array([0, 1, 2, 6, 8, 9])
    data = np.array([1, 2, 3, 4, 5, 6], dtype=dtype)
    carrs = (pos, crd, data)

    sv_array = sparse.from_constituent_arrays(format=format, arrays=carrs, shape=SHAPE)
    result = sv_array.get_constituent_arrays()
    for actual, expected in zip(result, carrs, strict=True):
        np.testing.assert_array_equal(actual, expected)

    res_arrs = sparse.add(sv_array, sv_array).get_constituent_arrays()
    sv2_expected = (pos, crd, data * 2)
    for actual, expected in zip(res_arrs, sv2_expected, strict=True):
        np.testing.assert_array_equal(actual, expected)

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
        tmp_levels = (sparse.levels.Level(sparse.levels.LevelFormat.Dense),) * len(shape)
        tmp_fmt = sparse.levels.get_storage_format(
            levels=tmp_levels,
            order="C",
            pos_width=64,
            crd_width=64,
            dtype=dtype,
        )
        arr_dense = arr.asformat(tmp_fmt)
        arr_np = sparse.to_numpy(arr_dense)
        expected_np = arr_np.reshape(new_shape)

        out_levels = (sparse.levels.Level(sparse.levels.LevelFormat.Dense),) * len(new_shape)
        out_fmt = sparse.levels.get_storage_format(
            levels=out_levels,
            order="C",
            pos_width=64,
            crd_width=64,
            dtype=dtype,
        )
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
    csf_format = sparse.levels.get_storage_format(
        levels=(
            sparse.levels.Level(sparse.levels.LevelFormat.Dense),
            sparse.levels.Level(sparse.levels.LevelFormat.Compressed),
            sparse.levels.Level(sparse.levels.LevelFormat.Compressed),
        ),
        order="C",
        pos_width=64,
        crd_width=64,
        dtype=sparse.asdtype(dtype),
    )
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


@pytest.mark.parametrize(
    "src_fmt", ["csr", "csc", pytest.param("coo", marks=pytest.mark.skip(reason="TODO: Report MLIR issue"))]
)
@pytest.mark.parametrize(
    "dst_fmt", ["csr", "csc", pytest.param("coo", marks=pytest.mark.skip(reason="TODO: Report MLIR issue"))]
)
def test_asformat(rng, src_fmt, dst_fmt):
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
