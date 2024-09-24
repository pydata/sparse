import ctypes
from types import EllipsisType

import mlir.execution_engine
import mlir.passmanager
from mlir import ir
from mlir.dialects import arith, func, linalg, sparse_tensor, tensor

import numpy as np

from ._common import fn_cache
from ._constructors import Tensor, numpy_to_ranked_memref
from ._core import CWD, DEBUG, MLIR_C_RUNNER_UTILS, ctx, pm
from ._dtypes import DType, FloatingDType, Index


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
        dtype = dtype.get_mlir_type()
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
def get_slice_module(
    in_tensor_type: ir.RankedTensorType,
    out_tensor_type: ir.RankedTensorType,
    offsets: tuple[int, ...],
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
) -> ir.Module:
    with ir.Location.unknown(ctx):
        module = ir.Module.create()

        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(in_tensor_type)
            def getitem(a):
                return tensor.extract_slice(out_tensor_type, a, [], [], [], offsets, sizes, strides)

        getitem.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        if DEBUG:
            (CWD / "getitem_module.mlir").write_text(str(module))
        pm.run(module.operation)
        if DEBUG:
            (CWD / "getitem_module_opt.mlir").write_text(str(module))

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


def add(x1: Tensor, x2: Tensor) -> Tensor:
    ret_obj = x1._format_class()
    out_tensor_type = x1._obj.get_tensor_definition(x1.shape)

    # TODO: Decide what will be the output tensor_type
    add_module = get_add_module(
        x1._obj.get_tensor_definition(x1.shape),
        x2._obj.get_tensor_definition(x2.shape),
        out_tensor_type=out_tensor_type,
        dtype=x1._values_dtype,
        rank=x1.ndim,
    )
    add_module.invoke(
        "add",
        ctypes.pointer(ctypes.pointer(ret_obj)),
        *x1._obj.to_module_arg(),
        *x2._obj.to_module_arg(),
    )
    return Tensor(ret_obj, shape=out_tensor_type.shape)


def _infer_format_class(rank: int, values_dtype: type[DType], index_dtype: type[DType]) -> type[ctypes.Structure]:
    from ._constructors import get_csf_class, get_csx_class, get_dense_class

    if rank == 1:
        return get_dense_class(values_dtype, index_dtype)
    if rank == 2:
        return get_csx_class(values_dtype, index_dtype, order="r")
    if rank == 3:
        return get_csf_class(values_dtype, index_dtype)
    raise Exception(f"Rank not supported to infer format: {rank}")


def reshape(x: Tensor, /, shape: tuple[int, ...]) -> Tensor:
    x_tensor_type = x._obj.get_tensor_definition(x.shape)
    if len(x.shape) == len(shape):
        out_tensor_type = x._obj.get_tensor_definition(shape)
        ret_obj = x._format_class()
    else:
        format_class = _infer_format_class(len(shape), x._values_dtype, x._index_dtype)
        out_tensor_type = format_class.get_tensor_definition(shape)
        ret_obj = format_class()

    with ir.Location.unknown(ctx):
        shape_tensor_type = ir.RankedTensorType.get([len(shape)], Index.get_mlir_type())

    reshape_module = get_reshape_module(x_tensor_type, shape_tensor_type, out_tensor_type)

    shape = np.array(shape)
    reshape_module.invoke(
        "reshape",
        ctypes.pointer(ctypes.pointer(ret_obj)),
        *x._obj.to_module_arg(),
        ctypes.pointer(ctypes.pointer(numpy_to_ranked_memref(shape))),
    )

    return Tensor(ret_obj, shape=out_tensor_type.shape)


def broadcast_to(x: Tensor, /, shape: tuple[int, ...], dimensions: list[int]) -> Tensor:
    x_tensor_type = x._obj.get_tensor_definition(x.shape)
    format_class = _infer_format_class(len(shape), x._values_dtype, x._index_dtype)
    out_tensor_type = format_class.get_tensor_definition(shape)
    ret_obj = format_class()

    broadcast_to_module = get_broadcast_to_module(x_tensor_type, out_tensor_type, tuple(dimensions))

    broadcast_to_module.invoke(
        "broadcast_to",
        ctypes.pointer(ctypes.pointer(ret_obj)),
        *x._obj.to_module_arg(),
    )

    return Tensor(ret_obj, shape=shape)


def _add_missing_dims(key: tuple, ndim: int) -> tuple:
    if len(key) < ndim and Ellipsis not in key:
        return key + (...,)
    return key


def _expand_ellipsis(key: tuple, ndim: int) -> tuple:
    if Ellipsis in key:
        if len([e for e in key if e is Ellipsis]) > 1:
            raise Exception(f"Ellipsis should be used once: {key}")
        to_expand = ndim - len(key) + 1
        if to_expand <= 0:
            raise Exception(f"Invalid use of Ellipsis in {key}")
        idx = key.index(Ellipsis)
        return key[:idx] + tuple(slice(None) for _ in range(to_expand)) + key[idx + 1 :]
    return key


def _decompose_slices(
    key: tuple,
    shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    offsets = []
    sizes = []
    strides = []

    for key_elem, size in zip(key, shape, strict=False):
        if isinstance(key_elem, slice):
            offset = key_elem.start if key_elem.start is not None else 0
            size = key_elem.stop - offset if key_elem.stop is not None else size - offset
            stride = key_elem.step if key_elem.step is not None else 1
        elif isinstance(key_elem, int):
            offset = key_elem
            size = key_elem + 1
            stride = 1
        offsets.append(offset)
        sizes.append(size)
        strides.append(stride)

    return tuple(offsets), tuple(sizes), tuple(strides)


def _get_new_shape(sizes, strides) -> tuple[int, ...]:
    return tuple(size // stride for size, stride in zip(sizes, strides, strict=False))


def getitem(
    x: Tensor,
    key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...],
) -> Tensor:
    if not isinstance(key, tuple):
        key = (key,)
    if None in key:
        raise Exception(f"Lazy indexing isn't supported: {key}")

    ret_obj = x._format_class()

    key = _add_missing_dims(key, x.ndim)
    key = _expand_ellipsis(key, x.ndim)
    offsets, sizes, strides = _decompose_slices(key, x.shape)

    new_shape = _get_new_shape(sizes, strides)
    out_tensor_type = x._obj.get_tensor_definition(new_shape)

    slice_module = get_slice_module(
        x._obj.get_tensor_definition(x.shape),
        out_tensor_type,
        offsets,
        sizes,
        strides,
    )

    slice_module.invoke("getitem", ctypes.pointer(ctypes.pointer(ret_obj)), *x._obj.to_module_arg())

    return Tensor(ret_obj, shape=out_tensor_type.shape)
