import abc
import dataclasses
import math
import sys

import mlir_finch.runtime as rt
from mlir_finch import ir

import numpy as np


class MlirType(abc.ABC):
    @abc.abstractmethod
    def _get_mlir_type(self) -> ir.Type: ...


def _get_pointer_width() -> int:
    return round(math.log2(sys.maxsize + 1.0)) + 1


_PTR_WIDTH = _get_pointer_width()


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class DType(MlirType):
    bit_width: int

    @property
    @abc.abstractmethod
    def np_dtype(self) -> np.dtype:
        raise NotImplementedError

    def to_ctype(self):
        return rt.as_ctype(self.np_dtype)

    def __eq__(self, value):
        if np.isdtype(value) or isinstance(value, str):
            value = asdtype(value)
        return super().__eq__(value)


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class IeeeRealFloatingDType(DType):
    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(getattr(np, f"float{self.bit_width}"))

    def _get_mlir_type(self) -> ir.Type:
        return getattr(ir, f"F{self.bit_width}Type").get()


float64 = IeeeRealFloatingDType(bit_width=64)
float32 = IeeeRealFloatingDType(bit_width=32)
float16 = IeeeRealFloatingDType(bit_width=16)


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class IeeeComplexFloatingDType(DType):
    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(getattr(np, f"complex{self.bit_width}"))

    def _get_mlir_type(self) -> ir.Type:
        return ir.ComplexType.get(getattr(ir, f"F{self.bit_width // 2}Type").get())


complex64 = IeeeComplexFloatingDType(bit_width=64)
complex128 = IeeeComplexFloatingDType(bit_width=128)


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class IntegerDType(DType):
    def _get_mlir_type(self) -> ir.Type:
        return ir.IntegerType.get_signless(self.bit_width)


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class UnsignedIntegerDType(IntegerDType):
    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(getattr(np, f"uint{self.bit_width}"))


uint8 = UnsignedIntegerDType(bit_width=8)
uint16 = UnsignedIntegerDType(bit_width=16)
uint32 = UnsignedIntegerDType(bit_width=32)
uint64 = UnsignedIntegerDType(bit_width=64)


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class SignedIntegerDType(IntegerDType):
    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(getattr(np, f"int{self.bit_width}"))


int8 = SignedIntegerDType(bit_width=8)
int16 = SignedIntegerDType(bit_width=16)
int32 = SignedIntegerDType(bit_width=32)
int64 = SignedIntegerDType(bit_width=64)


intp: SignedIntegerDType = locals()[f"int{_PTR_WIDTH}"]
uintp: UnsignedIntegerDType = locals()[f"uint{_PTR_WIDTH}"]


def isdtype(dt, /) -> bool:
    return isinstance(dt, DType)


NUMPY_DTYPE_MAP = {np.dtype(dt.np_dtype): dt for dt in locals().values() if isdtype(dt)}


def asdtype(dt, /) -> DType:
    if isdtype(dt):
        return dt

    return NUMPY_DTYPE_MAP[np.dtype(dt)]
