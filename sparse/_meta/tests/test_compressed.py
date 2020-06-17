import itertools

import pytest
import hypothesis
from hypothesis import strategies

from numba import njit
from numba import typed
from numba.core import types
from sparse._meta.compressed_level import Compressed
import functools

from .utils import njit_cached


size_strategy = strategies.integers(min_value=0, max_value=10)


def typed_int64_list(l, sort=False):
    pos = typed.List.empty_list(types.int64)
    for elem in l:
        pos.append(elem)
    if sort:
        pos.sort()
    return pos


@strategies.composite
def compressed_strategy(draw):
    pos = typed_int64_list(
        draw(strategies.lists(size_strategy, min_size=1, max_size=10)), sort=True
    )

    crd = typed_int64_list(
        draw(
            strategies.lists(
                strategies.integers(min_value=0, max_value=500),
                min_size=pos[-1],
                max_size=pos[-1],
            )
        ),
        sort=True,
    )

    full = draw(strategies.booleans())
    ordered = draw(strategies.booleans())
    unique = draw(strategies.booleans())

    return Compressed(pos=pos, crd=crd, full=full, ordered=ordered, unique=unique)


def round_trip(x: Compressed) -> Compressed:
    return x


def locate(d, pkm1, i):
    return d.locate(pkm1, i)


def attr_access(x: Compressed):
    return (x.pos, x.crd, x.full, x.ordered, x.unique, x.branchless, x.compact)


def coord_bounds(d: Compressed):
    return d.coord_bounds(0)


@hypothesis.given(c=compressed_strategy())
@hypothesis.settings(deadline=None)
def test_roundtrip(c):
    """
    This tests if boxing/unboxing works as expected
    """

    pyfunc = round_trip
    cfunc = njit_cached(pyfunc)

    c2 = round_trip(c)
    assert c2 == c
    assert c2.pos == c.pos
    assert c2.crd == c.crd


@hypothesis.given(c=compressed_strategy())
@hypothesis.settings(deadline=None)
def test_attribute_access(c):
    pyfunc = attr_access
    cfunc = njit_cached(attr_access)
    assert pyfunc(c) == cfunc(c)


def pos_bounds(c: Compressed, pkm1: int):
    return c.pos_bounds(pkm1)


@hypothesis.given(c=compressed_strategy())
@hypothesis.settings(deadline=None)
def test_pos_bounds(c):
    pyfunc = pos_bounds
    cfunc = njit_cached(pos_bounds)
    for i in range(len(c.pos) - 1):
        assert pyfunc(c, i) == cfunc(c, i)


def pos_access(c: Compressed, pk: int):
    return c.pos_access(pk, 0)


@hypothesis.given(c=compressed_strategy())
@hypothesis.settings(deadline=None)
def test_pos_access(c):
    pyfunc = pos_access
    cfunc = njit_cached(pos_access)
    for i in range(len(c.crd)):
        assert pyfunc(c, i) == cfunc(c, i)


# @hypothesis.given(d=dense_strategy, i=size_strategy)
# @hypothesis.settings(deadline=None)
# def test_size(d, i):
#     pyfunc = size
#     cfunc = njit_cached(size)
#     assert pyfunc(d, i) == cfunc(d, i)


# def insert_init(d, szkm1: int, szk: int):
#     return d.insert_init(szkm1, szk)


# def insert_coord(d, pk: int, ik: int):
#     return d.insert_coord(pk, ik)


# def insert_finalize(d, szkm1: int, szk: int):
#     return d.insert_finalize(szkm1, szk)


# @pytest.mark.parametrize("func", [insert_init, insert_coord, insert_finalize])
# @hypothesis.given(d=dense_strategy)
# @hypothesis.settings(deadline=None)
# def test_func(d, func):
#     pyfunc = func
#     cfunc = njit_cached(func)

#     for i, j in itertools.product(range(2), repeat=2):
#         assert pyfunc(d, i, j) == cfunc(d, i, j)


# @pytest.mark.parametrize("func", [coord_bounds])
# @hypothesis.given(d=dense_strategy)
# @hypothesis.settings(deadline=None)
# def test_func2(func, d):
#     pyfunc = func
#     cfunc = njit_cached(func)

#     assert pyfunc(d) == cfunc(d)
