from mlir import ir

import numpy as np


class DType:
    pass


class Float64(DType):
    np_dtype = np.float64

    @classmethod
    def get(cls):
        return ir.F64Type.get()


class Float32(DType):
    np_dtype = np.float32

    @classmethod
    def get(cls):
        return ir.F32Type.get()


class Int64(DType):
    np_dtype = np.int64

    @classmethod
    def get(cls):
        return ir.IntegerType.get_signed(64)


class UInt64(DType):
    np_dtype = np.uint64

    @classmethod
    def get(cls):
        return ir.IntegerType.get_unsigned(64)


class Int32(DType):
    np_dtype = np.int32

    @classmethod
    def get(cls):
        return ir.IntegerType.get_signed(32)


class UInt32(DType):
    np_dtype = np.uint32

    @classmethod
    def get(cls):
        return ir.IntegerType.get_unsigned(32)


class Index(DType):
    np_dtype = np.intp

    @classmethod
    def get(cls):
        return ir.IndexType.get()


class SignlessInt64(DType):
    np_dtype = np.int64

    @classmethod
    def get(cls):
        return ir.IntegerType.get_signless(64)
