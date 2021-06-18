import sparse
import pytest
import pytaco as pt
import numpy as np
import scipy


def test_todense():
    shape = (1, 2, 3)
    s1 = sparse.tensor(shape)
    s2 = pt.tensor(shape)
    s1 = s1.todense()
    s2 = s2.to_dense()
    np.testing.assert_array_equal(s1.to_array(), s2.to_array())


def test_from_scipy_sparse():
    s = scipy.sparse.csc_matrix((3, 4))
    t = sparse.ndsparse.from_scipy_sparse(s)
    np.testing.assert_array_equal(t.to_array(), s.toarray())


def test_from_ndarray():
    shape = (2, 2, 2)
    s = np.random.rand(*shape)
    t = sparse.ndsparse.from_array(s)
    np.testing.assert_array_equal(t.to_array(), s)


def test_to_scipy_csc():
    shape = (3, 4)
    s = scipy.sparse.csc_matrix(shape)
    t = sparse.ndsparse.from_scipy_sparse(s)
    t = t.to_sp_csc()
    np.testing.assert_array_equal(t.to_array(), s.toarray())


def test_to_scipy_csr():
    shape = (3, 4)
    s = scipy.sparse.csr_matrix(shape)
    t = sparse.ndsparse.from_scipy_sparse(s)
    t = t.to_sp_csr()
    np.testing.assert_array_equal(t.to_array(), s.toarray())


def test_order():
    shape = (1, 2, 3)
    s = sparse.tensor(shape)
    assert s._tensor.order == 3


def test_insert():
    shape = (1, 2, 3)
    s = sparse.tensor(shape)
    s.insert([0, 1, 2], 3)
    assert s[0, 1, 2] == 3


def test_add():
    a = np.random.rand(2, 2)
    b = np.random.rand(2, 2)
    sa = sparse.ndsparse.from_array(a)
    sb = sparse.ndsparse.from_array(b)
    c = sa + sb
    np.testing.assert_array_equal(a + b, c.to_dense())


def test_mul():
    a = np.random.rand(2, 2)
    b = np.random.rand(2, 2)
    sa = sparse.ndsparse.from_array(a)
    sb = sparse.ndsparse.from_array(b)
    c = sa * sb
    np.testing.assert_array_equal(a * b, c.to_dense())


def test_sub():
    a = np.random.rand(2, 2)
    b = np.random.rand(2, 2)
    sa = sparse.ndsparse.from_array(a)
    sb = sparse.ndsparse.from_array(b)
    c = sa - sb
    np.testing.assert_array_equal(a - b, c.to_dense())


def test_div():
    a = np.random.rand(2, 2)
    b = np.random.rand(2, 2)
    sa = sparse.ndsparse.from_array(a)
    sb = sparse.ndsparse.from_array(b)
    c = sa / sb
    np.testing.assert_array_equal(a / b, c.to_dense())
