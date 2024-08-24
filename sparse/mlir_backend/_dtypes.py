import inspect
import math
import sys

from mlir import ir

import numpy as np

from ._common import MlirType


def _get_pointer_width() -> int:
    return round(math.log2(sys.maxsize + 1.0)) + 1


_PTR_WIDTH = _get_pointer_width()


def _signless_is_signed() -> bool:
    with ir.Context(), ir.Location.unknown():
        return ir.IntegerType.get_signless(_PTR_WIDTH).is_signed


_SIGNLESS_IS_SIGNED = _signless_is_signed()


class DType(MlirType):
    np_dtype: np.dtype


class FloatingDType(DType): ...


class Float64(FloatingDType):
    np_dtype = np.float64

    @classmethod
    def get_mlir_type(cls):
        return ir.F64Type.get()


class Float32(FloatingDType):
    np_dtype = np.float32

    @classmethod
    def get_mlir_type(cls):
        return ir.F32Type.get()


class Float16(FloatingDType):
    np_dtype = np.float16

    @classmethod
    def get_mlir_type(cls):
        return ir.F16Type.get()


class IntegerDType(DType): ...


class UnsignedIntegerDType(IntegerDType): ...


class SignedIntegerDType(IntegerDType): ...


class Int64(SignedIntegerDType):
    np_dtype = np.int64

    @classmethod
    def get_mlir_type(cls):
        return ir.IntegerType.get_signed(64)


class UInt64(UnsignedIntegerDType):
    np_dtype = np.uint64

    @classmethod
    def get_mlir_type(cls):
        return ir.IntegerType.get_unsigned(64)


class Int32(SignedIntegerDType):
    np_dtype = np.int32

    @classmethod
    def get_mlir_type(cls):
        return ir.IntegerType.get_signed(32)


class UInt32(UnsignedIntegerDType):
    np_dtype = np.uint32

    @classmethod
    def get_mlir_type(cls):
        return ir.IntegerType.get_unsigned(32)


class Index(DType):
    np_dtype = np.intp if _SIGNLESS_IS_SIGNED else np.uintp

    @classmethod
    def get_mlir_type(cls):
        return ir.IndexType.get()


IntP: type[SignedIntegerDType] = locals()[f"Int{_PTR_WIDTH}"]
UIntP: type[UnsignedIntegerDType] = locals()[f"UInt{_PTR_WIDTH}"]
SignlessIntP: type[IntegerDType] = IntP if _SIGNLESS_IS_SIGNED else UIntP


def isdtype(dt, /) -> bool:
    return isinstance(dt, type) and issubclass(dt, DType) and not inspect.isabstract(dt)


NUMPY_DTYPE_MAP = {np.dtype(dt.np_dtype): dt for dt in locals().values() if isdtype(dt)}


def asdtype(dt, /) -> type[DType]:
    if isdtype(dt):
        return dt

    return NUMPY_DTYPE_MAP[np.dtype(dt)]
