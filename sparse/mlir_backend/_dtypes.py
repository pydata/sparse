import inspect
import math
import sys
import typing

from mlir import ir

import numpy as np

from ._common import MlirType


def _get_pointer_width() -> int:
    return round(math.log2(sys.maxsize + 1.0)) + 1


_PTR_WIDTH = _get_pointer_width()


def _make_int_classes(namespace: dict[str, object], bit_widths: typing.Iterable[int]) -> None:
    for bw in bit_widths:

        class SignedBW(SignedIntegerDType):
            np_dtype = getattr(np, f"int{bw}")
            bit_width = bw

            @classmethod
            def get_mlir_type(cls):
                return ir.IntegerType.get_signless(cls.bit_width)

        SignedBW.__name__ = f"Int{bw}"
        SignedBW.__module__ = __name__

        class UnsignedBW(UnsignedIntegerDType):
            np_dtype = getattr(np, f"uint{bw}")
            bit_width = bw

            @classmethod
            def get_mlir_type(cls):
                return ir.IntegerType.get_signless(cls.bit_width)

        UnsignedBW.__name__ = f"UInt{bw}"
        UnsignedBW.__module__ = __name__

        namespace[SignedBW.__name__] = SignedBW
        namespace[UnsignedBW.__name__] = UnsignedBW


class DType(MlirType):
    np_dtype: np.dtype
    bit_width: int


class FloatingDType(DType): ...


class Float64(FloatingDType):
    np_dtype = np.float64
    bit_width = 64

    @classmethod
    def get_mlir_type(cls):
        return ir.F64Type.get()


class Float32(FloatingDType):
    np_dtype = np.float32
    bit_width = 32

    @classmethod
    def get_mlir_type(cls):
        return ir.F32Type.get()


class Float16(FloatingDType):
    np_dtype = np.float16
    bit_width = 16

    @classmethod
    def get_mlir_type(cls):
        return ir.F16Type.get()


class IntegerDType(DType): ...


class UnsignedIntegerDType(IntegerDType): ...


class SignedIntegerDType(IntegerDType): ...


_make_int_classes(locals(), [8, 16, 32, 64])


class Index(DType):
    np_dtype = np.intp

    @classmethod
    def get_mlir_type(cls):
        return ir.IndexType.get()


IntP: type[SignedIntegerDType] = locals()[f"Int{_PTR_WIDTH}"]
UIntP: type[UnsignedIntegerDType] = locals()[f"UInt{_PTR_WIDTH}"]


def isdtype(dt, /) -> bool:
    return isinstance(dt, type) and issubclass(dt, DType) and not inspect.isabstract(dt)


NUMPY_DTYPE_MAP = {np.dtype(dt.np_dtype): dt for dt in locals().values() if isdtype(dt)}


def asdtype(dt, /) -> type[DType]:
    if isdtype(dt):
        return dt

    return NUMPY_DTYPE_MAP[np.dtype(dt)]
