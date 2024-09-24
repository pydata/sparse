import math
import typing

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
    ],
)


def assert_csx_equal(
    expected: sps.csr_array | sps.csc_array,
    actual: sps.csr_array | sps.csc_array,
) -> None:
    np.testing.assert_array_equal(expected.todense(), actual.todense())
    # Broken due to https://github.com/scipy/scipy/issues/21442
    # desired.sort_indices()
    # desired.sum_duplicates()
    # desired.prune()

    # actual.sort_indices()
    # actual.sum_duplicates()
    # actual.prune()

    # np.testing.assert_array_equal(desired.todense(), actual.todense())

    # np.testing.assert_array_equal(desired.indptr, actual.indptr)
    # np.testing.assert_array_equal(desired.indices, actual.indices)
    # np.testing.assert_array_equal(desired.data, actual.data)


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

    raise NotImplementedError(f"{dtype=} not yet supported.")


def get_exampe_csf_arrays(dtype: np.dtype) -> tuple:
    pos_1 = np.array([0, 1, 3], dtype=np.int64)
    crd_1 = np.array([1, 0, 1], dtype=np.int64)
    pos_2 = np.array([0, 3, 5, 7], dtype=np.int64)
    crd_2 = np.array([0, 1, 3, 0, 3, 0, 1], dtype=np.int64)
    data = np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype)
    return pos_1, crd_1, pos_2, crd_2, data


@parametrize_dtypes
@pytest.mark.parametrize("shape", [(100,), (10, 200), (5, 10, 20)])
def test_dense_format(dtype, shape):
    data = np.arange(math.prod(shape), dtype=dtype)
    tensor = sparse.asarray(data)
    actual = tensor.to_scipy_sparse()
    np.testing.assert_equal(actual, data)


@parametrize_dtypes
def test_constructors(rng, dtype):
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

    csr_retured = csr_tensor.to_scipy_sparse()
    assert_csx_equal(csr_retured, csr)

    csc_retured = csc_tensor.to_scipy_sparse()
    assert_csx_equal(csc_retured, csc)

    dense_returned = dense_tensor.to_scipy_sparse()
    np.testing.assert_equal(dense_returned, dense)

    coo_returned = coo_tensor.to_scipy_sparse()
    np.testing.assert_equal(coo_returned.todense(), coo.todense())

    dense_2_returned = dense_2_tensor.to_scipy_sparse()
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

    actual = sparse.add(csr_tensor, csr_2_tensor).to_scipy_sparse()
    expected = csr + csr_2
    assert_csx_equal(expected, actual)

    actual = sparse.add(csc_tensor, csc_tensor).to_scipy_sparse()
    expected = csc + csc
    assert_csx_equal(expected, actual)

    actual = sparse.add(csc_tensor, csr_tensor).to_scipy_sparse()
    expected = csc + csr
    assert_csx_equal(expected, actual)

    actual = sparse.add(csr_tensor, dense_tensor).to_scipy_sparse()
    expected = sps.csr_matrix(csr + dense)
    assert_csx_equal(expected, actual)

    actual = sparse.add(dense_tensor, csr_tensor).to_scipy_sparse()
    expected = csr + dense
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected)

    # NOTE: Fixed in https://github.com/llvm/llvm-project/pull/108615
    # actual = sparse.add(c_tensor, c_tensor).to_scipy_sparse()
    # expected = c + c
    # assert isinstance(actual, np.ndarray)
    # np.testing.assert_array_equal(actual, expected)

    actual = sparse.add(csr_2_tensor, coo_tensor).to_scipy_sparse()
    expected = csr_2 + coo
    np.testing.assert_array_equal(actual.todense(), expected.todense())

    # NOTE: https://discourse.llvm.org/t/passmanager-fails-on-simple-coo-addition-example/81247
    # actual = sparse.add(d_tensor, d_tensor).to_scipy_sparse()
    # expected = d + d
    # np.testing.assert_array_equal(actual.todense(), expected.todense())


@parametrize_dtypes
def test_csf_format(dtype):
    SHAPE = (2, 2, 4)
    pos_1, crd_1, pos_2, crd_2, data = get_exampe_csf_arrays(dtype)
    csf = [pos_1, crd_1, pos_2, crd_2, data]

    csf_tensor = sparse.asarray(csf, shape=SHAPE, dtype=sparse.asdtype(dtype), format="csf")
    result = csf_tensor.to_scipy_sparse()
    for actual, expected in zip(result, csf, strict=False):
        np.testing.assert_array_equal(actual, expected)

    res_tensor = sparse.add(csf_tensor, csf_tensor).to_scipy_sparse()
    csf_2 = [pos_1, crd_1, pos_2, crd_2, data * 2]
    for actual, expected in zip(res_tensor, csf_2, strict=False):
        np.testing.assert_array_equal(actual, expected)


@parametrize_dtypes
def test_coo_3d_format(dtype):
    SHAPE = (2, 2, 4)
    pos = np.array([0, 7])
    crd = np.array([[0, 1, 0, 0, 1, 1, 0], [1, 3, 1, 0, 0, 1, 0], [3, 1, 1, 0, 1, 1, 1]])
    data = np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype)
    coo = [pos, crd, data]

    coo_tensor = sparse.asarray(coo, shape=SHAPE, dtype=sparse.asdtype(dtype), format="coo")
    result = coo_tensor.to_scipy_sparse()
    for actual, expected in zip(result, coo, strict=False):
        np.testing.assert_array_equal(actual, expected)

    # NOTE: Blocked by https://github.com/llvm/llvm-project/pull/109135
    # res_tensor = sparse.add(coo_tensor, coo_tensor).to_scipy_sparse()
    # coo_2 = [pos, crd, data * 2]
    # for actual, expected in zip(res_tensor, coo_2, strict=False):
    #     np.testing.assert_array_equal(actual, expected)


@parametrize_dtypes
def test_reshape(rng, dtype):
    DENSITY = 0.5
    sampler = generate_sampler(dtype, rng)

    # CSR, CSC, COO
    for shape, new_shape in [
        ((100, 50), (25, 200)),
        ((100, 50), (10, 500, 1)),
        ((80, 1), (8, 10)),
        ((80, 1), (80,)),
    ]:
        for format in ["csr", "csc", "coo"]:
            if format == "coo":
                # NOTE: Blocked by https://github.com/llvm/llvm-project/pull/109135
                continue
            if format == "csc":
                # NOTE: Blocked by https://github.com/llvm/llvm-project/issues/109641
                continue

            arr = sps.random_array(
                shape, density=DENSITY, format=format, dtype=dtype, random_state=rng, data_sampler=sampler
            )
            arr.sum_duplicates()
            tensor = sparse.asarray(arr)

            actual = sparse.reshape(tensor, shape=new_shape).to_scipy_sparse()
            if isinstance(actual, sparse.PackedArgumentTuple):
                continue  # skip checking CSF output
            if not isinstance(actual, np.ndarray):
                actual = actual.todense()
            expected = arr.todense().reshape(new_shape)

            np.testing.assert_array_equal(actual, expected)

    # CSF
    csf_shape = (2, 2, 4)
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
        csf = get_exampe_csf_arrays(dtype)
        csf_tensor = sparse.asarray(csf, shape=shape, dtype=sparse.asdtype(dtype), format="csf")

        result = sparse.reshape(csf_tensor, shape=new_shape).to_scipy_sparse()

        for actual, expected in zip(result, expected_arrs, strict=False):
            np.testing.assert_array_equal(actual, expected)

    # DENSE
    # NOTE: dense reshape is probably broken in MLIR in 19.x branch
    # dense = np.arange(math.prod(SHAPE), dtype=dtype).reshape(SHAPE)


@parametrize_dtypes
def test_broadcast_to(dtype):
    # CSR, CSC, COO
    for shape, new_shape, dimensions, input_arr, expected_arrs in [
        (
            (3, 4),
            (2, 3, 4),
            [0],
            np.array([[0, 1, 0, 3], [0, 0, 4, 5], [6, 7, 0, 0]]),
            [
                np.array([0, 3, 6]),
                np.array([0, 1, 2, 0, 1, 2]),
                np.array([0, 2, 4, 6, 8, 10, 12]),
                np.array([1, 3, 2, 3, 0, 1, 1, 3, 2, 3, 0, 1]),
                np.array([1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            ],
        ),
        (
            (4, 2),
            (4, 2, 2),
            [1],
            np.array([[0, 1], [0, 0], [2, 3], [4, 0]]),
            [
                np.array([0, 2, 2, 4, 6]),
                np.array([0, 1, 0, 1, 0, 1]),
                np.array([0, 1, 2, 4, 6, 7, 8]),
                np.array([1, 1, 0, 1, 0, 1, 0, 0]),
                np.array([1.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 4.0]),
            ],
        ),
    ]:
        for fn_format in [sps.csr_array, sps.csc_array, sps.coo_array]:
            arr = fn_format(input_arr, shape=shape, dtype=dtype)
            arr.sum_duplicates()
            tensor = sparse.asarray(arr)
            result = sparse.broadcast_to(tensor, new_shape, dimensions=dimensions).to_scipy_sparse()

            for actual, expected in zip(result, expected_arrs, strict=False):
                np.testing.assert_allclose(actual, expected)

    # DENSE
    np_arr = np.array([0, 0, 2, 3, 0, 1])
    arr = np.asarray(np_arr, dtype=dtype)
    tensor = sparse.asarray(arr)
    result = sparse.broadcast_to(tensor, (3, 6), dimensions=[0]).to_scipy_sparse()

    assert result.format == "csr"
    np.testing.assert_allclose(result.todense(), np.repeat(np_arr[np.newaxis], 3, axis=0))


@pytest.mark.skip(reason="https://discourse.llvm.org/t/illegal-operation-when-slicing-csr-csc-coo-tensor/81404")
@parametrize_dtypes
@pytest.mark.parametrize(
    "index",
    [
        0,
        (2,),
        (2, 3),
        (..., slice(0, 4, 2)),
        (1, slice(1, None, 1)),
        # TODO: For below cases we need an update to ownership mechanism.
        #       `tensor[:, :]` returns the same memref that was passed.
        #       The mechanism sees the result as MLIR-allocated and frees
        #       it, while it still can be owned by SciPy/NumPy causing a
        #       segfault when it frees SciPy/NumPy managed memory.
        # ...,
        # slice(None),
        # (slice(None), slice(None)),
    ],
)
def test_indexing_2d(rng, dtype, index):
    SHAPE = (20, 30)
    DENSITY = 0.5

    for format in ["csr", "csc", "coo"]:
        arr = sps.random_array(SHAPE, density=DENSITY, format=format, dtype=dtype, random_state=rng)
        arr.sum_duplicates()

        tensor = sparse.asarray(arr)

        actual = tensor[index].to_scipy_sparse()
        expected = arr.todense()[index]

        np.testing.assert_array_equal(actual.todense(), expected)
