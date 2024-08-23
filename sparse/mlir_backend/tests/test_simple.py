import sparse

import pytest

import numpy as np
import scipy.sparse as sps

if sparse._BACKEND != sparse._BackendType.MLIR:
    pytest.skip("skipping MLIR tests", allow_module_level=True)


def test_constructors(rng):
    SHAPE = (10, 5)
    DENSITY = 0.5

    a = sps.random_array(SHAPE, density=DENSITY, format="csr", dtype=np.float64, random_state=rng)
    c = np.arange(50, dtype=np.float64).reshape((10, 5))

    a_tensor = sparse.asarray(a)
    c_tensor = sparse.asarray(c)

    a_retured = a_tensor.to_scipy_sparse()
    np.testing.assert_array_equal(a_retured.indices, a.indices)
    np.testing.assert_array_equal(a_retured.indptr, a.indptr)
    np.testing.assert_array_equal(a_retured.data, a.data)

    c_returned = c_tensor.to_scipy_sparse()
    np.testing.assert_array_equal(c_returned, c)


def test_add(rng):
    SHAPE = (10, 5)
    DENSITY = 0.5

    a = sps.random_array(SHAPE, density=DENSITY, format="csr", dtype=np.float64, random_state=rng)
    b = sps.random_array(SHAPE, density=DENSITY, format="csr", dtype=np.float64, random_state=rng)
    c = np.arange(50, dtype=np.float64).reshape((10, 5))

    a_tensor = sparse.asarray(a)
    b_tensor = sparse.asarray(b)
    c_tensor = sparse.asarray(c)

    actual = sparse.add(a_tensor, b_tensor).to_scipy_sparse()
    expected = a + b
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.data, expected.data)

    actual = sparse.add(a_tensor, c_tensor).to_scipy_sparse()
    expected = sps.csr_matrix(a + c)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.data, expected.data)

    actual = sparse.add(c_tensor, a_tensor).to_scipy_sparse()
    expected = a + c
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected)
