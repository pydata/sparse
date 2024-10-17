import ctypes
import functools
import weakref
from dataclasses import dataclass

import mlir.runtime as rt

import numpy as np

from ._core import libc
from ._dtypes import DType, asdtype


def fn_cache(f, maxsize: int | None = None):
    return functools.wraps(f)(functools.lru_cache(maxsize=maxsize)(f))


def get_nd_memref_descr(rank: int, dtype: type[DType]) -> ctypes.Structure:
    return _get_nd_memref_descr(int(rank), asdtype(dtype))


@fn_cache
def _get_nd_memref_descr(rank: int, dtype: type[DType]) -> ctypes.Structure:
    return rt.make_nd_memref_descriptor(rank, dtype.to_ctype())


def numpy_to_ranked_memref(arr: np.ndarray) -> ctypes.Structure:
    memref = rt.get_ranked_memref_descriptor(arr)
    memref_descr = get_nd_memref_descr(arr.ndim, asdtype(arr.dtype))
    # Required due to ctypes type checks
    return memref_descr(
        allocated=memref.allocated,
        aligned=memref.aligned,
        offset=memref.offset,
        shape=memref.shape,
        strides=memref.strides,
    )


def ranked_memref_to_numpy(ref: ctypes.Structure) -> np.ndarray:
    return rt.ranked_memref_to_numpy([ref])


def free_memref(obj: ctypes.Structure) -> None:
    libc.free(ctypes.cast(obj.allocated, ctypes.c_void_p))


@dataclass
class PackedArgumentTuple:
    contents: tuple

    def __getitem__(self, index):
        return self.contents[index]

    def __iter__(self):
        yield from self.contents

    def __len__(self):
        return len(self.contents)


def _hold_self_ref_in_ret(fn):
    @functools.wraps(fn)
    def wrapped(self, *a, **kw):
        ret = fn(self, *a, **kw)
        _hold_ref(ret, self)
        return ret

    return wrapped


def _hold_ref(owner, obj):
    ptr = ctypes.py_object(obj)
    ctypes.pythonapi.Py_IncRef(ptr)

    def finalizer(ptr):
        ctypes.pythonapi.Py_DecRef(ptr)

    weakref.finalize(owner, finalizer, ptr)
