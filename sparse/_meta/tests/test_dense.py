import itertools

import pytest
import hypothesis
from hypothesis import strategies

from numba import njit
from sparse._meta.dense_level import Dense
import functools

from .utils import njit_cached


size_strategy = strategies.integers(min_value=0, max_value=10)

dense_strategy = strategies.builds(
    Dense, N=size_strategy, unique=strategies.booleans(), ordered=strategies.booleans(),
)


def round_trip(x: Dense) -> Dense:
    return x


def locate(d, pkm1, i):
    return d.locate(pkm1, i)


def attr_access(x: Dense):
    return (x.N, x.full, x.ordered, x.unique, x.branchless, x.compact)


def coord_bounds(d: Dense):
    return d.coord_bounds(0)


@hypothesis.given(d=dense_strategy)
@hypothesis.settings(deadline=None)
def test_roundtrip(d):
    """
    This tests if boxing/unboxing works as expected
    """

    pyfunc = round_trip
    cfunc = njit_cached(pyfunc)

    d2 = round_trip(d)
    assert d2 == d
    assert d2.N == d.N


@hypothesis.given(d=dense_strategy)
@hypothesis.settings(deadline=None)
def test_attribute_access(d):
    pyfunc = attr_access
    cfunc = njit_cached(attr_access)
    assert pyfunc(d) == cfunc(d)


def coord_bounds(d: Dense):
    return d.coord_bounds(0)


@hypothesis.given(d=dense_strategy)
@hypothesis.settings(deadline=None)
def test_locate(d):
    pyfunc = locate
    cfunc = njit_cached(locate)
    for pkm1, i in itertools.product(range(2), repeat=2):
        assert pyfunc(d, pkm1, (i,)) == cfunc(d, pkm1, (i,))


def size(d: Dense, szkm1: int):
    return d.size(szkm1)


@hypothesis.given(d=dense_strategy, i=size_strategy)
@hypothesis.settings(deadline=None)
def test_size(d, i):
    pyfunc = size
    cfunc = njit_cached(size)
    assert pyfunc(d, i) == cfunc(d, i)


def insert_init(d, szkm1: int, szk: int):
    return d.insert_init(szkm1, szk)


def insert_coord(d, pk: int, ik: int):
    return d.insert_coord(pk, ik)


def insert_finalize(d, szkm1: int, szk: int):
    return d.insert_finalize(szkm1, szk)


@pytest.mark.parametrize("func", [insert_init, insert_coord, insert_finalize])
@hypothesis.given(d=dense_strategy)
@hypothesis.settings(deadline=None)
def test_func(d, func):
    pyfunc = func
    cfunc = njit_cached(func)

    for i, j in itertools.product(range(2), repeat=2):
        assert pyfunc(d, i, j) == cfunc(d, i, j)


@pytest.mark.parametrize("func", [coord_bounds])
@hypothesis.given(d=dense_strategy)
@hypothesis.settings(deadline=None)
def test_func2(func, d):
    pyfunc = func
    cfunc = njit_cached(func)

    assert pyfunc(d) == cfunc(d)
