import functools

from numba import njit


@functools.lru_cache(None)
def njit_cached(f):
    return njit(f)


def apply_decorators(decorators):
    def inner(f):
        for d in reversed(decorators):
            f = d(f)

        return f

    return inner
