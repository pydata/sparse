import ctypes
import dataclasses

import mlir.execution_engine
import mlir.passmanager
from mlir import ir
from mlir.dialects import arith, func, linalg, sparse_tensor, tensor

from ._array import Array
from ._common import fn_cache
from ._core import CWD, DEBUG, MLIR_C_RUNNER_UTILS, ctx, pm
from ._dtypes import DType, FloatingDType


@fn_cache
def get_add_module(
    a_tensor_type: ir.RankedTensorType,
    b_tensor_type: ir.RankedTensorType,
    out_tensor_type: ir.RankedTensorType,
    dtype: type[DType],
    rank: int,
) -> ir.Module:
    with ir.Location.unknown(ctx):
        module = ir.Module.create()
        # TODO: add support for complex dialect/dtypes
        arith_op = arith.AddFOp if issubclass(dtype, FloatingDType) else arith.AddIOp
        dtype = dtype._get_mlir_type()
        ordering = ir.AffineMap.get_permutation(range(rank))

        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(a_tensor_type, b_tensor_type)
            def add(a, b):
                out = tensor.empty(out_tensor_type, [])
                generic_op = linalg.GenericOp(
                    [out_tensor_type],
                    [a, b],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (ordering,) * 3]),
                    ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * rank),
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

    return mlir.execution_engine.ExecutionEngine(module, opt_level=2, shared_libs=[MLIR_C_RUNNER_UTILS])


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

    return mlir.execution_engine.ExecutionEngine(module, opt_level=2, shared_libs=[MLIR_C_RUNNER_UTILS])


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
                out = tensor.empty(out_tensor_type, [])
                return linalg.broadcast(in_tensor, outs=[out], dimensions=dimensions)

            broadcast_to.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            if DEBUG:
                (CWD / "broadcast_to_module.mlir").write_text(str(module))
            pm.run(module.operation)
            if DEBUG:
                (CWD / "broadcast_to_module_opt.mlir").write_text(str(module))

    return mlir.execution_engine.ExecutionEngine(module, opt_level=2, shared_libs=[MLIR_C_RUNNER_UTILS])


def add(x1: Array, x2: Array) -> Array:
    ret_storage_format = dataclasses.replace(x1._get_storage_format(), owns_memory=True)
    ret_storage = ret_storage_format._get_ctypes_type()()
    out_tensor_type = ret_storage_format._get_mlir_type(shape=x1.shape)

    # TODO: Decide what will be the output tensor_type
    add_module = get_add_module(
        x1._get_mlir_type(),
        x2._get_mlir_type(),
        out_tensor_type=out_tensor_type,
        dtype=x1.dtype,
        rank=x1.ndim,
    )
    add_module.invoke(
        "add",
        ctypes.pointer(ctypes.pointer(ret_storage)),
        *x1._to_module_arg(),
        *x2._to_module_arg(),
    )
    return Array(storage=ret_storage, shape=out_tensor_type.shape)
