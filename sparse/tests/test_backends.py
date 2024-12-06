import warnings

import sparse

import pytest

import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.csgraph as spgraph
import scipy.sparse.linalg as splin
from numpy.testing import assert_almost_equal, assert_equal


def test_backends(backend):
    rng = np.random.default_rng(0)
    x = sparse.random((100, 10, 100), density=0.01, random_state=rng)
    y = sparse.random((100, 10, 100), density=0.01, random_state=rng)

    if backend == sparse._BackendType.Finch:
        import finch

        def storage():
            return finch.Storage(finch.Dense(finch.SparseList(finch.SparseList(finch.Element(0.0)))), order="C")

        x = x.to_storage(storage())
        y = y.to_storage(storage())
    else:
        x.asformat("gcxs")
        y.asformat("gcxs")

    z = x + y
    result = sparse.sum(z)
    assert result.shape == ()


def test_finch_lazy_backend(backend):
    if backend != sparse._BackendType.Finch:
        pytest.skip("Tested only for Finch backend")

    import finch

    np_eye = np.eye(5)
    sp_arr = sps.csr_matrix(np_eye)
    finch_dense = finch.Tensor(np_eye)

    assert np.shares_memory(finch_dense.todense(), np_eye)

    finch_arr = finch.Tensor(sp_arr)

    assert_equal(finch_arr.todense(), np_eye)

    transposed = sparse.permute_dims(finch_arr, (1, 0))

    assert_equal(transposed.todense(), np_eye.T)

    @sparse.compiled()
    def my_fun(tns1, tns2):
        tmp = sparse.add(tns1, tns2)
        return sparse.sum(tmp, axis=0)

    result = my_fun(finch_dense, finch_arr)

    assert_equal(result.todense(), np.sum(2 * np_eye, axis=0))


@pytest.mark.parametrize("format, order", [("csc", "F"), ("csr", "C"), ("coo", "F"), ("coo", "C")])
def test_asarray(backend, format, order):
    arr = np.eye(5, order=order)

    result = sparse.asarray(arr, format=format)

    assert_equal(result.todense(), arr)


@pytest.mark.parametrize("format, order", [("csc", "F"), ("csr", "C"), ("coo", "F"), ("coo", "C")])
def test_scipy_spsolve(backend, format, order):
    x = np.eye(10, order=order) * 2
    y = np.ones((10, 1), order=order)
    x_pydata = sparse.asarray(x, format=format)
    y_pydata = sparse.asarray(y, format="coo")

    actual = splin.spsolve(x_pydata, y_pydata)
    expected = np.linalg.solve(x, y.ravel())
    assert_almost_equal(actual, expected)


@pytest.mark.parametrize("format, order", [("csc", "F"), ("csr", "C"), ("coo", "F"), ("coo", "C")])
def test_scipy_inv(backend, format, order):
    x = np.eye(10, order=order) * 2
    x_pydata = sparse.asarray(x, format=format)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=sps.SparseEfficiencyWarning)
        actual = splin.inv(x_pydata)
    expected = np.linalg.inv(x)
    assert_almost_equal(actual.todense(), expected)


@pytest.mark.skip(reason="https://github.com/scipy/scipy/pull/20759")
@pytest.mark.parametrize("format, order", [("csc", "F"), ("csr", "C"), ("coo", "F"), ("coo", "C")])
def test_scipy_norm(backend, format, order):
    x = np.eye(10, order=order) * 2
    x_pydata = sparse.asarray(x, format=format)

    actual = splin.norm(x_pydata)
    expected = sp.linalg.norm(x)
    assert_almost_equal(actual, expected)


@pytest.mark.skip(reason="https://github.com/scipy/scipy/pull/20759")
@pytest.mark.parametrize("format, order", [("csc", "F"), ("csr", "C"), ("coo", "F"), ("coo", "C")])
def test_scipy_lsqr(backend, format, order):
    x = np.eye(10, order=order) * 2
    y = np.ones((10, 1), order=order)
    x_pydata = sparse.asarray(x, format=format)

    actual_x, _ = splin.lsqr(x_pydata, y)[:2]
    expected_x, _ = sp.linalg.lstsq(x, y)[:2]
    assert_almost_equal(actual_x, expected_x.ravel())


@pytest.mark.skip(reason="https://github.com/scipy/scipy/pull/20759")
@pytest.mark.parametrize("format, order", [("csc", "F"), ("csr", "C"), ("coo", "F"), ("coo", "C")])
def test_scipy_eigs(backend, format, order):
    x = np.eye(10, order=order) * 2
    x_pydata = sparse.asarray(x, format=format)
    x_sp = sps.coo_matrix(x)

    actual_vals, _ = splin.eigs(x_pydata, k=3)
    expected_vals, _ = splin.eigs(x_sp, k=3)
    assert_almost_equal(actual_vals, expected_vals)


@pytest.mark.parametrize(
    "matrix_fn, format, order",
    [(sps.csc_matrix, "csc", "F"), (sps.csr_matrix, "csr", "C"), (sps.coo_matrix, "coo", "F")],
)
def test_scipy_connected_components(backend, graph, matrix_fn, format, order):
    graph = matrix_fn(np.array(graph, order=order))
    sp_graph = sparse.asarray(graph, format=format)

    actual_n_components, actual_labels = spgraph.connected_components(sp_graph)
    expected_n_components, expected_labels = spgraph.connected_components(graph)
    assert actual_n_components == expected_n_components
    assert_equal(actual_labels, expected_labels)


@pytest.mark.parametrize(
    "matrix_fn, format, order",
    [(sps.csc_matrix, "csc", "F"), (sps.csr_matrix, "csr", "C"), (sps.coo_matrix, "coo", "F")],
)
def test_scipy_laplacian(backend, graph, matrix_fn, format, order):
    graph = matrix_fn(np.array(graph, order=order))
    sp_graph = sparse.asarray(graph, format=format)

    actual_lap = spgraph.laplacian(sp_graph)
    expected_lap = spgraph.laplacian(graph)
    assert_equal(actual_lap.todense(), expected_lap.toarray())


@pytest.mark.parametrize("matrix_fn, format, order", [(sps.csc_matrix, "csc", "F"), (sps.csr_matrix, "csr", "C")])
def test_scipy_shortest_path(backend, graph, matrix_fn, format, order):
    graph = matrix_fn(np.array(graph, order=order))
    sp_graph = sparse.asarray(graph, format=format)

    actual_dist_matrix, actual_predecessors = spgraph.shortest_path(sp_graph, return_predecessors=True)
    expected_dist_matrix, expected_predecessors = spgraph.shortest_path(graph, return_predecessors=True)
    assert_equal(actual_dist_matrix, expected_dist_matrix)
    assert_equal(actual_predecessors, expected_predecessors)


@pytest.mark.parametrize(
    "matrix_fn, format, order",
    [(sps.csc_matrix, "csc", "F"), (sps.csr_matrix, "csr", "C"), (sps.coo_matrix, "coo", "F")],
)
def test_scipy_breadth_first_tree(backend, graph, matrix_fn, format, order):
    graph = matrix_fn(np.array(graph, order=order))
    sp_graph = sparse.asarray(graph, format=format)

    actual_bft = spgraph.breadth_first_tree(sp_graph, 0, directed=False)
    expected_bft = spgraph.breadth_first_tree(graph, 0, directed=False)
    assert_equal(actual_bft.todense(), expected_bft.toarray())


@pytest.mark.parametrize(
    "matrix_fn, format, order",
    [(sps.csc_matrix, "csc", "F"), (sps.csr_matrix, "csr", "C"), (sps.coo_matrix, "coo", "F")],
)
def test_scipy_dijkstra(backend, graph, matrix_fn, format, order):
    graph = matrix_fn(np.array(graph, order=order))
    sp_graph = sparse.asarray(graph, format=format)

    actual_dist_matrix = spgraph.dijkstra(sp_graph, directed=False)
    expected_dist_matrix = spgraph.dijkstra(graph, directed=False)
    assert_equal(actual_dist_matrix, expected_dist_matrix)


@pytest.mark.parametrize(
    "matrix_fn, format, order",
    [(sps.csc_matrix, "csc", "F"), (sps.csr_matrix, "csr", "C"), (sps.coo_matrix, "coo", "F")],
)
def test_scipy_minimum_spanning_tree(backend, graph, matrix_fn, format, order):
    graph = matrix_fn(np.array(graph, order=order))
    sp_graph = sparse.asarray(graph, format=format)

    actual_span_tree = spgraph.minimum_spanning_tree(sp_graph)
    expected_span_tree = spgraph.minimum_spanning_tree(graph)
    assert_equal(actual_span_tree.todense(), expected_span_tree.toarray())


@pytest.mark.skip(reason="https://github.com/scikit-learn/scikit-learn/pull/29031")
@pytest.mark.parametrize("matrix_fn, format, order", [(sps.csc_matrix, "csc", "F")])
def test_scikit_learn_dispatch(backend, graph, matrix_fn, format, order):
    from sklearn.cluster import KMeans

    graph = matrix_fn(np.array(graph, order=order))

    sp_graph = sparse.asarray(graph, format=format)

    neigh = KMeans(n_clusters=2)
    actual_labels = neigh.fit_predict(sp_graph)

    neigh = KMeans(n_clusters=2)
    expected_labels = neigh.fit_predict(graph)

    assert_equal(actual_labels, expected_labels)
