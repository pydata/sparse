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


def assert_csr_equal(expected: sps.csr_array, actual: sps.csr_array) -> None:
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
    a = sps.random_array(SHAPE, density=DENSITY, format="csr", dtype=dtype, random_state=rng, data_sampler=sampler)
    c = np.arange(math.prod(SHAPE), dtype=dtype).reshape(SHAPE)
    d = sps.random_array(SHAPE, density=DENSITY, format="coo", dtype=dtype, random_state=rng, data_sampler=sampler)
    d.sum_duplicates()

    a_tensor = sparse.asarray(a)
    c_tensor = sparse.asarray(c)
    d_tensor = sparse.asarray(d)
    e_tensor = sparse.asarray(np.arange(100, dtype=dtype).reshape((25, 4)) + 10)

    a_retured = a_tensor.to_scipy_sparse()
    assert_csr_equal(a, a_retured)

    c_returned = c_tensor.to_scipy_sparse()
    np.testing.assert_equal(c, c_returned)

    d_returned = d_tensor.to_scipy_sparse()
    np.testing.assert_equal(d.todense(), d_returned.todense())

    e_returned = e_tensor.to_scipy_sparse()
    np.testing.assert_equal(e_returned, np.arange(100, dtype=dtype).reshape((25, 4)) + 10)


@parametrize_dtypes
def test_add(rng, dtype):
    SHAPE = (100, 50)
    DENSITY = 0.5
    sampler = generate_sampler(dtype, rng)

    a = sps.random_array(SHAPE, density=DENSITY, format="csr", dtype=dtype, random_state=rng, data_sampler=sampler)
    b = sps.random_array(SHAPE, density=DENSITY, format="csr", dtype=dtype, random_state=rng, data_sampler=sampler)
    c = np.arange(math.prod(SHAPE), dtype=dtype).reshape(SHAPE)
    d = sps.random_array(SHAPE, density=DENSITY, format="coo", dtype=dtype, random_state=rng)
    d.sum_duplicates()

    a_tensor = sparse.asarray(a)
    b_tensor = sparse.asarray(b)
    c_tensor = sparse.asarray(c)
    d_tensor = sparse.asarray(d)

    actual = sparse.add(a_tensor, b_tensor).to_scipy_sparse()
    expected = a + b
    assert_csr_equal(expected, actual)

    actual = sparse.add(a_tensor, c_tensor).to_scipy_sparse()
    expected = sps.csr_matrix(a + c)
    assert_csr_equal(expected, actual)

    actual = sparse.add(c_tensor, a_tensor).to_scipy_sparse()
    expected = a + c
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected)

    # NOTE: Fixed in https://github.com/llvm/llvm-project/pull/108615
    # actual = sparse.add(c_tensor, c_tensor).to_scipy_sparse()
    # expected = c + c
    # assert isinstance(actual, np.ndarray)
    # np.testing.assert_array_equal(actual, expected)

    actual = sparse.add(b_tensor, d_tensor).to_scipy_sparse()
    expected = b + d
    np.testing.assert_array_equal(actual.todense(), expected.todense())

    # NOTE: https://discourse.llvm.org/t/passmanager-fails-on-simple-coo-addition-example/81247
    # actual = sparse.add(d_tensor, d_tensor).to_scipy_sparse()
    # expected = d + d
    # np.testing.assert_array_equal(actual.todense(), expected.todense())
