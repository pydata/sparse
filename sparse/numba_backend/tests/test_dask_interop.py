import sparse

from dask.base import tokenize


def test_deterministic_token():
    a = sparse.COO(data=[1, 2, 3], coords=[10, 20, 30], shape=(40,))
    b = sparse.COO(data=[1, 2, 3], coords=[10, 20, 30], shape=(40,))
    assert tokenize(a) == tokenize(b)
    # One of these things is not like the other....
    c = sparse.COO(data=[1, 2, 4], coords=[10, 20, 30], shape=(40,))
    assert tokenize(a) != tokenize(c)
