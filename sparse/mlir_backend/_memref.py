import ctypes

import numpy as np

from ._common import fn_cache
from ._dtypes import DType, asdtype


def make_memref_ctype(dtype: type[DType], rank: int) -> type[ctypes.Structure]:
    dtype = np.dtype(asdtype(dtype).np_dtype)
    return _make_memref_ctype(dtype, rank)


@fn_cache
def _make_memref_ctype(dtype: np.dtype, rank: int) -> type[ctypes.Structure]:
    ctype = np.ctypeslib.as_ctypes_type(dtype)
    ptr_t = ctypes.POINTER(ctype)

    class MemrefType(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ptr_t),
            ("offset", np.ctypeslib.c_intp),
            ("shape", np.ctypeslib.c_intp * rank),
            ("strides", np.ctypeslib.c_intp * rank),
        ]

        @classmethod
        def from_numpy(cls, arr: np.ndarray) -> "MemrefType":
            if not arr.dtype == dtype:
                raise TypeError(f"Expected {dtype=}, found {arr.dtype=}.")
            if not all(s % arr.itemsize == 0 for s in arr.strides):
                raise ValueError(f"Strides not item aligned: {arr.strides=}, {arr.itemsize=}")

            ptr = ctypes.cast(arr.ctypes.data, ctypes.c_void_p)
            ptr_typed = ctypes.cast(ptr, ptr_t)
            return cls(
                allocated=ptr,
                aligned=ptr_typed,
                offset=0,
                shape=arr.shape,
                strides=tuple(s // arr.itemsize for s in arr.strides),
            )

        def to_numpy(self) -> np.ndarray:
            if ctypes.cast(self.aligned, ctypes.c_void_p).value != ctypes.cast(self.allocated, ctypes.c_void_p).value:
                raise RuntimeError("Encountered different values for `aligned` and `allocated`.")
            shape = tuple(self.shape)
            ptr = self.aligned
            ret = np.ctypeslib.as_array(ptr, shape)
            strides = tuple(s * ret.itemsize for s in self.strides)
            if ret.strides != strides:
                raise RuntimeError(f"Expected {ret.strides=} for {shape=}, got {strides=}.")
            return ret

        def __hash__(self) -> int:
            return hash(id(self))

    return MemrefType


def ranked_memref_from_np(arr: np.ndarray) -> ctypes.Structure:
    memref_type = _make_memref_ctype(arr.dtype, arr.ndim)
    return memref_type.from_numpy(arr)
