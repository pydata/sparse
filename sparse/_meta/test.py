from numba import njit
from .sparsedim import Dense
from .numba_bindings import DenseType


@njit
def foo(N, ordered, unique):
    d = Dense(N=N, ordered=ordered, unique=unique)


foo(3, True, True)
