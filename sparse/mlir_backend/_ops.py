import ctypes
import math

import mlir_finch.execution_engine
import mlir_finch.passmanager
from mlir_finch import ir
from mlir_finch.dialects import arith, complex, func, linalg, sparse_tensor, tensor

import numpy as np

from ._array import Array
from ._common import as_shape, fn_cache
from ._core import CWD, DEBUG, OPT_LEVEL, SHARED_LIBS, ctx, pm
from ._dtypes import DType, IeeeComplexFloatingDType, IeeeRealFloatingDType, IntegerDType
from .formats import ConcreteFormat, _determine_format


@fn_cache
def get_add_module(
    a_tensor_type: ir.RankedTensorType,
    b_tensor_type: ir.RankedTensorType,
    out_tensor_type: ir.RankedTensorType,
    dtype: DType,
) -> ir.Module:
    with ir.Location.unknown(ctx):
        module = ir.Module.create()
        if isinstance(dtype, IeeeRealFloatingDType):
            arith_op = arith.AddFOp
        elif isinstance(dtype, IeeeComplexFloatingDType):
            arith_op = complex.AddOp
        elif isinstance(dtype, IntegerDType):
            arith_op = arith.AddIOp
        else:
            raise RuntimeError(f"Can not add {dtype=}.")

        dtype = dtype._get_mlir_type()
        max_rank = out_tensor_type.rank

        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(a_tensor_type, b_tensor_type)
            def add(a, b):
                out = tensor.empty(out_tensor_type.shape, dtype, encoding=out_tensor_type.encoding)
                generic_op = linalg.GenericOp(
                    [out_tensor_type],
                    [a, b],
                    [out],
                    ir.ArrayAttr.get(
                        [
                            ir.AffineMapAttr.get(ir.AffineMap.get_minor_identity(max_rank, t.rank))
                            for t in (a_tensor_type, b_tensor_type, out_tensor_type)
                        ]
                    ),
                    ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * max_rank),
                )
                block = generic_op.regions[0].blocks.append(dtype, dtype, dtype)
                with ir.InsertionPoint(block):
                    a, b, o = block.arguments
                    res = sparse_tensor.BinaryOp(dtype, a, b)
                    overlap = res.regions[0].blocks.append(dtype, dtype)
                    with ir.InsertionPoint(overlap):
                        arg0, arg1 = overlap.arguments
                        overlap_res = arith_op(arg0, arg1)
                        sparse_tensor.YieldOp([overlap_res])
                    left_region = res.regions[1].blocks.append(dtype)
                    with ir.InsertionPoint(left_region):
                        (arg0,) = left_region.arguments
                        sparse_tensor.YieldOp([arg0])
                    right_region = res.regions[2].blocks.append(dtype)
                    with ir.InsertionPoint(right_region):
                        (arg0,) = right_region.arguments
                        sparse_tensor.YieldOp([arg0])
                    linalg.YieldOp([res])
                return generic_op.result

        add.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        if DEBUG:
            (CWD / "add_module.mlir").write_text(str(module))
        pm.run(module.operation)
        if DEBUG:
            (CWD / "add_module_opt.mlir").write_text(str(module))

    return mlir_finch.execution_engine.ExecutionEngine(module, opt_level=OPT_LEVEL, shared_libs=SHARED_LIBS)


@fn_cache
def get_reshape_module(
    a_tensor_type: ir.RankedTensorType,
    shape_tensor_type: ir.RankedTensorType,
    out_tensor_type: ir.RankedTensorType,
) -> ir.Module:
    with ir.Location.unknown(ctx):
        module = ir.Module.create()

        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(a_tensor_type, shape_tensor_type)
            def reshape(a, shape):
                return tensor.reshape(out_tensor_type, a, shape)

            reshape.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            if DEBUG:
                (CWD / "reshape_module.mlir").write_text(str(module))
            pm.run(module.operation)
            if DEBUG:
                (CWD / "reshape_module_opt.mlir").write_text(str(module))

    return mlir_finch.execution_engine.ExecutionEngine(module, opt_level=OPT_LEVEL, shared_libs=SHARED_LIBS)


@fn_cache
def get_broadcast_to_module(
    in_tensor_type: ir.RankedTensorType,
    out_tensor_type: ir.RankedTensorType,
    dimensions: tuple[int, ...],
) -> ir.Module:
    with ir.Location.unknown(ctx):
        module = ir.Module.create()

        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(in_tensor_type)
            def broadcast_to(in_tensor):
                out = tensor.empty(
                    out_tensor_type.shape, out_tensor_type.element_type, encoding=out_tensor_type.encoding
                )
                return linalg.broadcast(in_tensor, outs=[out], dimensions=dimensions)

            broadcast_to.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            if DEBUG:
                (CWD / "broadcast_to_module.mlir").write_text(str(module))
            pm.run(module.operation)
            if DEBUG:
                (CWD / "broadcast_to_module_opt.mlir").write_text(str(module))

    return mlir_finch.execution_engine.ExecutionEngine(module, opt_level=OPT_LEVEL, shared_libs=SHARED_LIBS)


@fn_cache
def get_convert_module(
    in_tensor_type: ir.RankedTensorType,
    out_tensor_type: ir.RankedTensorType,
):
    with ir.Location.unknown(ctx):
        module = ir.Module.create()

        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(in_tensor_type)
            def convert(in_tensor):
                return sparse_tensor.convert(out_tensor_type, in_tensor)

            convert.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            if DEBUG:
                (CWD / "convert_module.mlir").write_text(str(module))
            pm.run(module.operation)
            if DEBUG:
                (CWD / "convert_module.mlir").write_text(str(module))

    return mlir_finch.execution_engine.ExecutionEngine(module, opt_level=OPT_LEVEL, shared_libs=SHARED_LIBS)


def add(x1: Array, x2: Array, /) -> Array:
    # TODO: Determine output format via autoscheduler
    ret_storage_format = _determine_format(x1.format, x2.format, dtype=x1.dtype, union=True)
    ret_storage = ret_storage_format._get_ctypes_type(owns_memory=True)()
    out_tensor_type = ret_storage_format._get_mlir_type(shape=np.broadcast_shapes(x1.shape, x2.shape))

    add_module = get_add_module(
        x1._get_mlir_type(),
        x2._get_mlir_type(),
        out_tensor_type=out_tensor_type,
        dtype=x1.dtype,
    )
    add_module.invoke(
        "add",
        ctypes.pointer(ctypes.pointer(ret_storage)),
        *x1._to_module_arg(),
        *x2._to_module_arg(),
    )
    return Array(storage=ret_storage, shape=tuple(out_tensor_type.shape))


def asformat(x: Array, /, format: ConcreteFormat) -> Array:
    if format.rank != x.ndim:
        raise ValueError(f"`format.rank != `self.ndim`, {format.rank=}, {x.ndim=}")

    if format == x.format:
        return x

    out_tensor_type = format._get_mlir_type(shape=x.shape)
    ret_storage = format._get_ctypes_type(owns_memory=True)()

    convert_module = get_convert_module(
        x._get_mlir_type(),
        out_tensor_type,
    )

    convert_module.invoke(
        "convert",
        ctypes.pointer(ctypes.pointer(ret_storage)),
        *x._to_module_arg(),
    )

    return Array(storage=ret_storage, shape=x.shape)


def reshape(x: Array, /, shape: tuple[int, ...]) -> Array:
    from ._conversions import _from_numpy

    shape = as_shape(shape)
    if math.prod(x.shape) != math.prod(shape):
        raise ValueError(f"`math.prod(x.shape) != math.prod(shape)`, {x.shape=}, {shape=}")

    ret_storage_format = _determine_format(x.format, dtype=x.dtype, union=len(shape) > x.ndim, out_ndim=len(shape))
    shape_array = _from_numpy(np.asarray(shape, dtype=np.uint64))
    out_tensor_type = ret_storage_format._get_mlir_type(shape=shape)
    ret_storage = ret_storage_format._get_ctypes_type(owns_memory=True)()

    reshape_module = get_reshape_module(x._get_mlir_type(), shape_array._get_mlir_type(), out_tensor_type)

    reshape_module.invoke(
        "reshape",
        ctypes.pointer(ctypes.pointer(ret_storage)),
        *x._to_module_arg(),
        *shape_array._to_module_arg(),
    )

    return Array(storage=ret_storage, shape=shape)
