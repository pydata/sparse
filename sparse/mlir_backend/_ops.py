import ctypes

import mlir.execution_engine
import mlir.passmanager
from mlir import ir
from mlir.dialects import arith, func, linalg, sparse_tensor, tensor

from ._constructors import Tensor
from ._core import CWD, DEBUG, MLIR_C_RUNNER_UTILS, ctx, pm
from ._dtypes import DType, FloatingDType


def get_add_module(
    a_tensor_type: ir.RankedTensorType,
    b_tensor_type: ir.RankedTensorType,
    out_tensor_type: ir.RankedTensorType,
    dtype: type[DType],
) -> ir.Module:
    with ir.Location.unknown(ctx):
        module = ir.Module.create()
        # TODO: add support for complex dialect/dtypes
        arith_op = arith.AddFOp if issubclass(dtype, FloatingDType) else arith.AddIOp
        dtype = dtype.get_mlir_type()
        ordering = ir.AffineMap.get_permutation([0, 1])

        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(a_tensor_type, b_tensor_type)
            def add(a, b):
                out = tensor.empty(out_tensor_type, [])
                generic_op = linalg.GenericOp(
                    [out_tensor_type],
                    [a, b],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (ordering,) * 3]),
                    ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 2),
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


def add(x1: Tensor, x2: Tensor) -> Tensor:
    ret_ptr = ctypes.c_void_p()

    # TODO: Add proper caching
    # TODO: Decide what will be the output tensor_type
    add_module = get_add_module(
        x1.tensor_type,
        x2.tensor_type,
        out_tensor_type=x1.tensor_type,
        dtype=x1.values_dtype,
    )
    add_module.invoke(
        "add",
        ctypes.pointer(x1.obj),
        ctypes.pointer(x2.obj),
        ctypes.pointer(ret_ptr),
    )
    return Tensor(ret_ptr, x1.module, x1.tensor_type, x1.disassemble_fn, x1.values_dtype, x1.index_dtype)
