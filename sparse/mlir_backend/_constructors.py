import ctypes
import functools
import weakref

import mlir.execution_engine
import mlir.passmanager
from mlir import ir
from mlir import runtime as rt
from mlir.dialects import arith, bufferization, func, sparse_tensor, tensor

import numpy as np
import scipy.sparse as sps

from ._common import fn_cache
from ._core import CWD, DEBUG, MLIR_C_RUNNER_UTILS, ctx
from ._dtypes import DType, Index, asdtype


def _hold_self_ref_in_ret(fn):
    @functools.wraps(fn)
    def wrapped(self, *a, **kw):
        ptr = ctypes.py_object(self)
        ctypes.pythonapi.Py_IncRef(ptr)
        ret = fn(self, *a, **kw)

        def finalizer(ptr):
            ctypes.pythonapi.Py_DecRef(ptr)

        weakref.finalize(ret, finalizer, ptr)
        return ret

    return wrapped


class Tensor:
    def __init__(self, obj, module, tensor_type, disassemble_fn, values_dtype, index_dtype):
        self.obj = obj
        self.module = module
        self.tensor_type = tensor_type
        self.disassemble_fn = disassemble_fn
        self.values_dtype = values_dtype
        self.index_dtype = index_dtype

    def __del__(self):
        self.module.invoke("free_tensor", ctypes.pointer(self.obj))

    @_hold_self_ref_in_ret
    def to_scipy_sparse(self):
        """
        Returns scipy.sparse or ndarray
        """
        return self.disassemble_fn(self.module, self.obj, self.values_dtype)


class DenseFormat:
    @fn_cache
    def get_module(shape: tuple[int], values_dtype: DType, index_dtype: DType):
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            values_dtype = values_dtype.get_mlir_type()
            index_dtype = index_dtype.get_mlir_type()
            index_width = getattr(index_dtype, "width", 0)
            levels = (sparse_tensor.LevelFormat.dense, sparse_tensor.LevelFormat.dense)
            ordering = ir.AffineMap.get_permutation([0, 1])
            encoding = sparse_tensor.EncodingAttr.get(levels, ordering, ordering, index_width, index_width)
            dense_shaped = ir.RankedTensorType.get(list(shape), values_dtype, encoding)
            tensor_1d = tensor.RankedTensorType.get([ir.ShapedType.get_dynamic_size()], values_dtype)

            with ir.InsertionPoint(module.body):

                @func.FuncOp.from_py_func(tensor_1d)
                def assemble(data):
                    return sparse_tensor.assemble(dense_shaped, [], data)

                @func.FuncOp.from_py_func(dense_shaped)
                def disassemble(tensor_shaped):
                    data = tensor.EmptyOp([arith.constant(ir.IndexType.get(), 0)], values_dtype)
                    data, data_len = sparse_tensor.disassemble(
                        [],
                        tensor_1d,
                        [],
                        index_dtype,
                        tensor_shaped,
                        [],
                        data,
                    )
                    shape_x = arith.constant(index_dtype, shape[0])
                    shape_y = arith.constant(index_dtype, shape[1])
                    return data, data_len, shape_x, shape_y

                @func.FuncOp.from_py_func(dense_shaped)
                def free_tensor(tensor_shaped):
                    bufferization.dealloc_tensor(tensor_shaped)

            assemble.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            disassemble.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            free_tensor.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            if DEBUG:
                (CWD / "dense_module.mlir").write_text(str(module))
            pm = mlir.passmanager.PassManager.parse("builtin.module(sparsifier{create-sparse-deallocs=1})")
            pm.run(module.operation)
            if DEBUG:
                (CWD / "dense_module_opt.mlir").write_text(str(module))

        module = mlir.execution_engine.ExecutionEngine(module, opt_level=2, shared_libs=[MLIR_C_RUNNER_UTILS])
        return (module, dense_shaped)

    @classmethod
    def assemble(cls, module, arr: np.ndarray) -> ctypes.c_void_p:
        assert arr.ndim == 2
        data = rt.get_ranked_memref_descriptor(arr.flatten())
        out = ctypes.c_void_p()
        module.invoke(
            "assemble",
            ctypes.pointer(ctypes.pointer(data)),
            ctypes.pointer(out),
        )
        return out

    @classmethod
    def disassemble(cls, module: ir.Module, ptr: ctypes.c_void_p, dtype: type[DType]) -> np.ndarray:
        class Dense(ctypes.Structure):
            _fields_ = [
                ("data", rt.make_nd_memref_descriptor(1, dtype.to_ctype())),
                ("data_len", np.ctypeslib.c_intp),
                ("shape_x", np.ctypeslib.c_intp),
                ("shape_y", np.ctypeslib.c_intp),
            ]

            def to_np(self) -> np.ndarray:
                data = rt.ranked_memref_to_numpy([self.data])[: self.data_len]
                return data.reshape((self.shape_x, self.shape_y))

        arr = Dense()
        module.invoke(
            "disassemble",
            ctypes.pointer(ctypes.pointer(arr)),
            ctypes.pointer(ptr),
        )
        return arr.to_np()


class COOFormat:
    @fn_cache
    def get_module(shape: tuple[int], values_dtype: type[DType], index_dtype: type[DType]):
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            values_dtype = values_dtype.get_mlir_type()
            index_dtype = index_dtype.get_mlir_type()
            index_width = getattr(index_dtype, "width", 0)
            compressed_lvl = sparse_tensor.EncodingAttr.build_level_type(
                sparse_tensor.LevelFormat.compressed, [sparse_tensor.LevelProperty.non_unique]
            )
            levels = (compressed_lvl, sparse_tensor.LevelFormat.singleton)
            ordering = ir.AffineMap.get_permutation([0, 1])
            encoding = sparse_tensor.EncodingAttr.get(levels, ordering, ordering, index_width, index_width)
            coo_shaped = ir.RankedTensorType.get(list(shape), values_dtype, encoding)

            tensor_1d_index = tensor.RankedTensorType.get([ir.ShapedType.get_dynamic_size()], index_dtype)
            tensor_2d_index = tensor.RankedTensorType.get([ir.ShapedType.get_dynamic_size(), len(shape)], index_dtype)
            tensor_1d_values = tensor.RankedTensorType.get([ir.ShapedType.get_dynamic_size()], values_dtype)

            with ir.InsertionPoint(module.body):

                @func.FuncOp.from_py_func(tensor_1d_index, tensor_2d_index, tensor_1d_values)
                def assemble(pos, index, values):
                    return sparse_tensor.assemble(coo_shaped, (pos, index), values)

                @func.FuncOp.from_py_func(coo_shaped)
                def disassemble(tensor_shaped):
                    nse = sparse_tensor.number_of_entries(tensor_shaped)
                    pos = tensor.EmptyOp([arith.constant(ir.IndexType.get(), 2)], index_dtype)
                    index = tensor.EmptyOp([nse, 2], index_dtype)
                    values = tensor.EmptyOp([nse], values_dtype)
                    pos, index, values, pos_len, index_len, values_len = sparse_tensor.disassemble(
                        (tensor_1d_index, tensor_2d_index),
                        tensor_1d_values,
                        (index_dtype, index_dtype),
                        index_dtype,
                        tensor_shaped,
                        (pos, index),
                        values,
                    )
                    shape_consts = [arith.constant(index_dtype, s) for s in shape]
                    return pos, index, values, pos_len, index_len, values_len, *shape_consts

                @func.FuncOp.from_py_func(coo_shaped)
                def free_tensor(tensor_shaped):
                    bufferization.dealloc_tensor(tensor_shaped)

            assemble.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            disassemble.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            free_tensor.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            if DEBUG:
                (CWD / "coo_module.mlir").write_text(str(module))
            pm = mlir.passmanager.PassManager.parse("builtin.module(sparsifier{create-sparse-deallocs=1})")
            pm.run(module.operation)
            if DEBUG:
                (CWD / "coo_module_opt.mlir").write_text(str(module))

        module = mlir.execution_engine.ExecutionEngine(module, opt_level=2, shared_libs=[MLIR_C_RUNNER_UTILS])
        return (module, coo_shaped)

    @classmethod
    def assemble(cls, module: ir.Module, arr: sps.coo_array) -> ctypes.c_void_p:
        out = ctypes.c_void_p()
        module.invoke(
            "assemble",
            ctypes.pointer(
                ctypes.pointer(rt.get_ranked_memref_descriptor(np.array([0, arr.size], dtype=arr.coords[0].dtype)))
            ),
            ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(np.stack(arr.coords, axis=1)))),
            ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(arr.data))),
            ctypes.pointer(out),
        )
        return out

    @classmethod
    def disassemble(cls, module: ir.Module, ptr: ctypes.c_void_p, dtype: type[DType]) -> sps.coo_array:
        class Coo(ctypes.Structure):
            _fields_ = [
                ("pos", rt.make_nd_memref_descriptor(1, Index.to_ctype())),
                ("index", rt.make_nd_memref_descriptor(2, Index.to_ctype())),
                ("values", rt.make_nd_memref_descriptor(1, dtype.to_ctype())),
                ("pos_len", np.ctypeslib.c_intp),
                ("index_len", np.ctypeslib.c_intp),
                ("values_len", np.ctypeslib.c_intp),
                ("shape_x", np.ctypeslib.c_intp),
                ("shape_y", np.ctypeslib.c_intp),
            ]

            def to_sps(self) -> sps.coo_array:
                pos = rt.ranked_memref_to_numpy([self.pos])[: self.pos_len]
                index = rt.ranked_memref_to_numpy([self.index])[pos[0] : pos[1]]
                values = rt.ranked_memref_to_numpy([self.values])[: self.values_len]
                return sps.coo_array((values, index.T), shape=(self.shape_x, self.shape_y))

        arr = Coo()
        module.invoke(
            "disassemble",
            ctypes.pointer(ctypes.pointer(arr)),
            ctypes.pointer(ptr),
        )
        return arr.to_sps()


class CSRFormat:
    @fn_cache
    def get_module(shape: tuple[int], values_dtype: type[DType], index_dtype: type[DType]):
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            values_dtype = values_dtype.get_mlir_type()
            index_dtype = index_dtype.get_mlir_type()
            index_width = getattr(index_dtype, "width", 0)
            levels = (sparse_tensor.LevelFormat.dense, sparse_tensor.LevelFormat.compressed)
            ordering = ir.AffineMap.get_permutation([0, 1])
            encoding = sparse_tensor.EncodingAttr.get(levels, ordering, ordering, index_width, index_width)
            csr_shaped = ir.RankedTensorType.get(list(shape), values_dtype, encoding)

            tensor_1d_index = tensor.RankedTensorType.get([ir.ShapedType.get_dynamic_size()], index_dtype)
            tensor_1d_values = tensor.RankedTensorType.get([ir.ShapedType.get_dynamic_size()], values_dtype)

            with ir.InsertionPoint(module.body):

                @func.FuncOp.from_py_func(tensor_1d_index, tensor_1d_index, tensor_1d_values)
                def assemble(pos, crd, data):
                    return sparse_tensor.assemble(csr_shaped, (pos, crd), data)

                @func.FuncOp.from_py_func(csr_shaped)
                def disassemble(tensor_shaped):
                    pos = tensor.EmptyOp([arith.constant(ir.IndexType.get(), 0)], index_dtype)
                    crd = tensor.EmptyOp([arith.constant(ir.IndexType.get(), 0)], index_dtype)
                    data = tensor.EmptyOp([arith.constant(ir.IndexType.get(), 0)], values_dtype)
                    pos, crd, data, pos_len, crd_len, data_len = sparse_tensor.disassemble(
                        (tensor_1d_index, tensor_1d_index),
                        tensor_1d_values,
                        (index_dtype, index_dtype),
                        index_dtype,
                        tensor_shaped,
                        (pos, crd),
                        data,
                    )
                    shape_x = arith.constant(index_dtype, shape[0])
                    shape_y = arith.constant(index_dtype, shape[1])
                    return pos, crd, data, pos_len, crd_len, data_len, shape_x, shape_y

                @func.FuncOp.from_py_func(csr_shaped)
                def free_tensor(tensor_shaped):
                    bufferization.dealloc_tensor(tensor_shaped)

            assemble.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            disassemble.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            free_tensor.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            if DEBUG:
                (CWD / "csr_module.mlir").write_text(str(module))
            pm = mlir.passmanager.PassManager.parse("builtin.module(sparsifier{create-sparse-deallocs=1})")
            pm.run(module.operation)
            if DEBUG:
                (CWD / "csr_module_opt.mlir").write_text(str(module))

        module = mlir.execution_engine.ExecutionEngine(module, opt_level=2, shared_libs=[MLIR_C_RUNNER_UTILS])
        return (module, csr_shaped)

    @classmethod
    def assemble(cls, module: ir.Module, arr: sps.csr_array) -> ctypes.c_void_p:
        out = ctypes.c_void_p()
        module.invoke(
            "assemble",
            ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(arr.indptr))),
            ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(arr.indices))),
            ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(arr.data))),
            ctypes.pointer(out),
        )
        return out

    @classmethod
    def disassemble(cls, module: ir.Module, ptr: ctypes.c_void_p, dtype: type[DType]) -> sps.csr_array:
        class Csr(ctypes.Structure):
            _fields_ = [
                ("pos", rt.make_nd_memref_descriptor(1, Index.to_ctype())),
                ("crd", rt.make_nd_memref_descriptor(1, Index.to_ctype())),
                ("data", rt.make_nd_memref_descriptor(1, dtype.to_ctype())),
                ("pos_len", np.ctypeslib.c_intp),
                ("crd_len", np.ctypeslib.c_intp),
                ("data_len", np.ctypeslib.c_intp),
                ("shape_x", np.ctypeslib.c_intp),
                ("shape_y", np.ctypeslib.c_intp),
            ]

            def to_sps(self) -> sps.csr_array:
                pos = rt.ranked_memref_to_numpy([self.pos])[: self.pos_len]
                crd = rt.ranked_memref_to_numpy([self.crd])[: self.crd_len]
                data = rt.ranked_memref_to_numpy([self.data])[: self.data_len]
                return sps.csr_array((data, crd, pos), shape=(self.shape_x, self.shape_y))

        arr = Csr()
        module.invoke(
            "disassemble",
            ctypes.pointer(ctypes.pointer(arr)),
            ctypes.pointer(ptr),
        )
        return arr.to_sps()


def _is_scipy_sparse_obj(x) -> bool:
    return hasattr(x, "__module__") and x.__module__.startswith("scipy.sparse")


def _is_numpy_obj(x) -> bool:
    return isinstance(x, np.ndarray)


def asarray(obj) -> Tensor:
    # TODO: discover obj's dtype
    values_dtype = asdtype(obj.dtype)

    # TODO: support other scipy formats
    if _is_scipy_sparse_obj(obj):
        if obj.format == "csr":
            format_class = CSRFormat
            # This can be int32 or int64
            index_dtype = asdtype(obj.indptr.dtype)
        elif obj.format == "coo":
            format_class = COOFormat
            # This can be int32 or int64
            index_dtype = asdtype(obj.coords[0].dtype)
        else:
            raise Exception(f"{obj.format} SciPy format not supported.")
    elif _is_numpy_obj(obj):
        format_class = DenseFormat
        index_dtype = Index
    else:
        raise Exception(f"{type(obj)} not supported.")

    # TODO: support proper caching
    module, tensor_type = format_class.get_module(obj.shape, values_dtype, index_dtype)

    assembled_obj = format_class.assemble(module, obj)
    return Tensor(assembled_obj, module, tensor_type, format_class.disassemble, values_dtype, index_dtype)
