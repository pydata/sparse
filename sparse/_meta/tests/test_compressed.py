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
@hypothesis.settings(deadline=None, max_examples=100)
def test_roundtrip(c):
    """
    This tests if boxing/unboxing works as expected
    """
    
    pyfunc = round_trip
    cfunc = njit_cached(pyfunc)

    c2 = cfunc(c)
    assert c == c2
    assert c.pos == c2.pos
    assert c.crd == c2.crd


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


def make_copy(c: Compressed) -> Compressed:
    return Compressed(
        pos=c.pos.copy(),
        crd=c.crd.copy(),
        full=c.full,
        ordered=c.ordered,
        unique=c.unique,
    )


def append(c: Compressed) -> None:
    pkbegin = 0
    c.append_init(3, 9)
    for pkm1 in range(3):
        pkend = pkbegin
        for jj in range(3):
            pkend += 1
            c.append_coord(pkend, jj)
        c.append_edges(pkm1, pkbegin, pkend)
        pkbegin = pkend
    c.append_finalize(3, 9)


@hypothesis.given(c=compressed_strategy())
@hypothesis.settings(deadline=None)
def test_append(c):
    c2 = make_copy(c)
    pyfunc = append
    cfunc = njit_cached(append)

    assert pyfunc(c) == cfunc(c2)
    assert c.crd == c2.crd
