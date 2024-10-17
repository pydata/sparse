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
    expected.eliminate_zeros()
    expected.sum_duplicates()

    actual.eliminate_zeros()
    actual.sum_duplicates()

    np.testing.assert_array_equal(expected.indptr, actual.indptr)
    np.testing.assert_array_equal(expected.indices, actual.indices)
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
    assert_csx_equal(csr_retured, csr)

    csc_retured = sparse.to_scipy(csc_tensor)
    assert_csx_equal(csc_retured, csc)

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
    assert_csx_equal(expected, actual)

    actual = sparse.to_scipy(sparse.add(csc_tensor, csc_tensor))
    expected = csc + csc
    assert_csx_equal(expected, actual)

    actual = sparse.to_scipy(sparse.add(csc_tensor, csr_tensor))
    expected = csc + csr
    assert_csx_equal(expected, actual)

    actual = sparse.to_scipy(sparse.add(csr_tensor, dense_tensor))
    expected = sps.csr_matrix(csr + dense)
    assert_csx_equal(expected, actual)

    actual = sparse.to_numpy(sparse.add(dense_tensor, csr_tensor))
    expected = csr + dense
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected)

    # NOTE: Fixed in https://github.com/llvm/llvm-project/pull/108615
    # actual = sparse.add(c_tensor, c_tensor).to_scipy_sparse()
    # expected = c + c
    # assert isinstance(actual, np.ndarray)
    # np.testing.assert_array_equal(actual, expected)

    actual = sparse.to_scipy(sparse.add(csr_2_tensor, coo_tensor))
    expected = csr_2 + coo
    assert_csx_equal(expected, actual)

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
def test_sparse_vector_format(dtype):
    SHAPE = (10,)
    pos = np.array([0, 6])
    crd = np.array([0, 1, 2, 6, 8, 9])
    data = np.array([1, 2, 3, 4, 5, 6], dtype=dtype)
    sparse_vector = [pos, crd, data]

    sv_tensor = sparse.asarray(
        sparse_vector,
        shape=SHAPE,
        dtype=sparse.asdtype(dtype),
        format="sparse_vector",
    )
    result = sv_tensor.to_scipy_sparse()
    for actual, expected in zip(result, sparse_vector, strict=False):
        np.testing.assert_array_equal(actual, expected)

    res_tensor = sparse.add(sv_tensor, sv_tensor).to_scipy_sparse()
    sparse_vector_2 = [pos, crd, data * 2]
    for actual, expected in zip(res_tensor, sparse_vector_2, strict=False):
        np.testing.assert_array_equal(actual, expected)

    dense = np.array([1, 2, 3, 0, 0, 0, 4, 0, 5, 6], dtype=dtype)
    dense_tensor = sparse.asarray(dense)
    res_tensor = sparse.add(dense_tensor, sv_tensor).to_scipy_sparse()
    np.testing.assert_array_equal(res_tensor, dense * 2)
