import itertools

from numba import njit
from sparse._meta.dense_level import Dense

import pytest


def apply_decorators(decorators):
    def inner(f):
        for d in reversed(decorators):
            f = d(f)

        return f

    return inner


parametrize_dense = apply_decorators(
    [
        pytest.mark.parametrize("N", [3, 4, 5]),
        pytest.mark.parametrize("unique", [True, False]),
        pytest.mark.parametrize("ordered", [True, False]),
    ]
)


def round_trip(x: Dense) -> Dense:
    return x


def locate(d, pkm1, i):
    return d.locate(pkm1, i)


def attr_access(x: Dense):
    return (x.N, x.full, x.ordered, x.unique, x.branchless, x.compact)


def coord_bounds(d: Dense):
    return d.coord_bounds(0)


@parametrize_dense
def test_roundtrip(N, ordered, unique):
    """
    This tests if boxing/unboxing works as expected
    """
    d = Dense(N=N, unique=unique, ordered=ordered)

    pyfunc = round_trip
    cfunc = njit(pyfunc)

    d2 = round_trip(d)
    assert d2 == d
    assert d2.N == N


@parametrize_dense
def test_attribute_access(N, ordered, unique):
    d = Dense(N=N, unique=unique, ordered=ordered)

    pyfunc = attr_access
    cfunc = njit(attr_access)
    assert pyfunc(d) == cfunc(d), d


def coord_bounds(d: Dense):
    return d.coord_bounds(0)


@parametrize_dense
def test_locate(N, ordered, unique):
    d = Dense(N=N, unique=unique, ordered=ordered)

    pyfunc = locate
    cfunc = njit(locate)
    for pkm1, i in itertools.product(range(2), repeat=2):
        assert pyfunc(d, pkm1, (i,)) == cfunc(d, pkm1, (i,))


def size(d: Dense, szkm1: int):
    return d.size(szkm1)

njit_size = njit(size)

@parametrize_dense
def test_size(N, ordered, unique):
    d = Dense(N=N, unique=unique, ordered=ordered)

    pyfunc = size
    cfunc = njit_size
    for i in range(2):
        assert pyfunc(d, i) == cfunc(d, i)


def insert_init(d, szkm1: int, szk: int):
    return d.insert_init(szkm1, szk)


def insert_coord(d, pk: int, ik: int):
    return d.insert_coord(pk, ik)


def insert_finalize(d, szkm1: int, szk: int):
    return d.insert_finalize(szkm1, szk)


@pytest.mark.parametrize("func", [insert_init, insert_coord, insert_finalize])
@parametrize_dense
def test_func(func, N, ordered, unique):
    d = Dense(N=N, unique=unique, ordered=ordered)
    pyfunc = func
    cfunc = njit(func)

    for i, j in itertools.product(range(2), repeat=2):
        assert pyfunc(d, i, j) == cfunc(d, i, j)


@pytest.mark.parametrize("func", [coord_bounds])
@parametrize_dense
def test_func2(func, N, ordered, unique):
    d = Dense(N=N, unique=unique, ordered=ordered)
    pyfunc = func
    cfunc = njit(func)

    assert pyfunc(d) == cfunc(d)
