import abc
import ctypes
import functools
import typing
import weakref

import mlir.execution_engine
import mlir.passmanager
from mlir import ir
from mlir import runtime as rt
from mlir.dialects import arith, bufferization, func, sparse_tensor, tensor

import numpy as np
import scipy.sparse as sps

from ._common import fn_cache
from ._core import CWD, DEBUG, MLIR_C_RUNNER_UTILS, ctx, pm
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
        return self.disassemble_fn(
            self.module,
            self.obj,
            self.tensor_type.shape,
            self.values_dtype,
            self.index_dtype,
        )


class BaseFormat(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_format_str(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_levels(cls) -> tuple[sparse_tensor.LevelFormat, ...]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_ordering(cls) -> ir.AffineMap:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_assemble_functions(
        cls,
        module: ir.Module,
        tensor_shaped: ir.RankedTensorType,
        values_dtype: type[DType],
        index_dtype: type[DType],
    ) -> tuple[typing.Callable, ...]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def assemble(cls, module: ir.Module, arr: np.ndarray | sps.sparray) -> ctypes.c_void_p:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def disassemble(
        cls,
        module: ir.Module,
        ptr: ctypes.c_void_p,
        shape: list[int],
        values_dtype: type[DType],
        index_dtype: type[DType],
    ) -> np.ndarray | sps.sparray:
        raise NotImplementedError

    @classmethod
    @fn_cache
    def get_module(
        cls,
        shape: tuple[int],
        values_dtype: type[DType],
        index_dtype: type[DType],
    ) -> tuple[ir.Module, ir.RankedTensorType]:
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            values_dtype = values_dtype.get_mlir_type()
            index_dtype = index_dtype.get_mlir_type()
            index_width = getattr(index_dtype, "width", 0)
            levels = cls.get_levels()
            ordering = cls.get_ordering()
            encoding = sparse_tensor.EncodingAttr.get(levels, ordering, ordering, index_width, index_width)
            tensor_shaped = ir.RankedTensorType.get(list(shape), values_dtype, encoding)

            assemble, disassemble, free_tensor = cls.get_assemble_functions(
                module, tensor_shaped, values_dtype, index_dtype
            )

            assemble.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            disassemble.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            free_tensor.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            if DEBUG:
                (CWD / f"{cls.get_format_str()}.mlir").write_text(str(module))
            pm.run(module.operation)
            if DEBUG:
                (CWD / f"{cls.get_format_str()}_opt.mlir").write_text(str(module))

        module = mlir.execution_engine.ExecutionEngine(module, opt_level=2, shared_libs=[MLIR_C_RUNNER_UTILS])
        return (module, tensor_shaped)


class AbstractDenseFormat(BaseFormat):
    @classmethod
    def get_assemble_functions(
        cls,
        module: ir.Module,
        tensor_shaped: ir.RankedTensorType,
        values_dtype: type[DType],
        index_dtype: type[DType],
    ) -> tuple[typing.Callable, ...]:
        tensor_1d = tensor.RankedTensorType.get([ir.ShapedType.get_dynamic_size()], values_dtype)
        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(tensor_1d)
            def assemble(data):
                return sparse_tensor.assemble(tensor_shaped, [], data)

            @func.FuncOp.from_py_func(tensor_shaped)
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
                return data, data_len

            @func.FuncOp.from_py_func(tensor_shaped)
            def free_tensor(tensor_shaped):
                bufferization.dealloc_tensor(tensor_shaped)

        return assemble, disassemble, free_tensor

    @classmethod
    def assemble(cls, module: ir.Module, arr: np.ndarray) -> ctypes.c_void_p:
        data = rt.get_ranked_memref_descriptor(arr)
        out = ctypes.c_void_p()
        module.invoke(
            "assemble",
            ctypes.pointer(ctypes.pointer(data)),
            ctypes.pointer(out),
        )
        return out

    @classmethod
    def disassemble(
        cls,
        module: ir.Module,
        ptr: ctypes.c_void_p,
        shape: list[int],
        values_dtype: type[DType],
        index_dtype: type[DType],
    ) -> np.ndarray:
        class Dense(ctypes.Structure):
            _fields_ = [
                ("data", rt.make_nd_memref_descriptor(1, values_dtype.to_ctype())),
                ("data_len", np.ctypeslib.c_intp),
            ]

            def to_np(self) -> np.ndarray:
                data = rt.ranked_memref_to_numpy([self.data])[: self.data_len]
                return data.reshape(shape)

        arr = Dense()
        module.invoke(
            "disassemble",
            ctypes.pointer(ctypes.pointer(arr)),
            ctypes.pointer(ptr),
        )
        return arr.to_np()


class VectorFormat(AbstractDenseFormat):
    @classmethod
    def get_format_str(cls) -> str:
        return "sparse_vector_format"

    @classmethod
    def get_levels(cls) -> tuple[sparse_tensor.LevelFormat, ...]:
        return (sparse_tensor.LevelFormat.dense,)

    @classmethod
    def get_ordering(cls) -> ir.AffineMap:
        return ir.AffineMap.get_permutation([0])


class Dense2DFormat(AbstractDenseFormat):
    @classmethod
    def get_format_str(cls) -> str:
        return "dense_2d_format"

    @classmethod
    def get_levels(cls) -> tuple[sparse_tensor.LevelFormat, ...]:
        return (sparse_tensor.LevelFormat.dense,) * 2

    @classmethod
    def get_ordering(cls) -> ir.AffineMap:
        return ir.AffineMap.get_permutation([0, 1])


class Dense3DFormat(AbstractDenseFormat):
    @classmethod
    def get_format_str(cls) -> str:
        return "dense_3d_format"

    @classmethod
    def get_levels(cls) -> tuple[sparse_tensor.LevelFormat, ...]:
        return (sparse_tensor.LevelFormat.dense,) * 3

    @classmethod
    def get_ordering(cls) -> ir.AffineMap:
        return ir.AffineMap.get_permutation([0, 2, 3])


class COOFormat(BaseFormat):
    @classmethod
    def get_format_str(cls) -> str:
        return "coo_format"

    @classmethod
    def get_levels(cls) -> tuple[sparse_tensor.LevelFormat, ...]:
        compressed_lvl = sparse_tensor.EncodingAttr.build_level_type(
            sparse_tensor.LevelFormat.compressed, [sparse_tensor.LevelProperty.non_unique]
        )
        return (compressed_lvl, sparse_tensor.LevelFormat.singleton)

    @classmethod
    def get_ordering(cls) -> ir.AffineMap:
        return ir.AffineMap.get_permutation([0, 1])

    @classmethod
    def get_assemble_functions(
        cls,
        module: ir.Module,
        tensor_shaped: ir.RankedTensorType,
        values_dtype: type[DType],
        index_dtype: type[DType],
    ) -> tuple[typing.Callable, ...]:
        tensor_1d_index = tensor.RankedTensorType.get([ir.ShapedType.get_dynamic_size()], index_dtype)
        tensor_2d_index = tensor.RankedTensorType.get(
            [ir.ShapedType.get_dynamic_size(), tensor_shaped.rank], index_dtype
        )
        tensor_1d_values = tensor.RankedTensorType.get([ir.ShapedType.get_dynamic_size()], values_dtype)
        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(tensor_1d_index, tensor_2d_index, tensor_1d_values)
            def assemble(pos, index, values):
                return sparse_tensor.assemble(tensor_shaped, (pos, index), values)

            @func.FuncOp.from_py_func(tensor_shaped)
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
                return pos, index, values, pos_len, index_len, values_len

            @func.FuncOp.from_py_func(tensor_shaped)
            def free_tensor(tensor_shaped):
                bufferization.dealloc_tensor(tensor_shaped)

        return assemble, disassemble, free_tensor

    @classmethod
    def assemble(cls, module: ir.Module, arr: sps.coo_array) -> ctypes.c_void_p:
        out = ctypes.c_void_p()
        index_dtype = arr.coords[0].dtype
        module.invoke(
            "assemble",
            ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(np.array([0, arr.size], dtype=index_dtype)))),
            ctypes.pointer(
                ctypes.pointer(rt.get_ranked_memref_descriptor(np.stack(arr.coords, axis=1, dtype=index_dtype)))
            ),
            ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(arr.data))),
            ctypes.pointer(out),
        )
        return out

    @classmethod
    def disassemble(
        cls,
        module: ir.Module,
        ptr: ctypes.c_void_p,
        shape: list[int],
        values_dtype: type[DType],
        index_dtype: type[DType],
    ) -> sps.coo_array:
        class Coo(ctypes.Structure):
            _fields_ = [
                ("pos", rt.make_nd_memref_descriptor(1, index_dtype.to_ctype())),
                ("index", rt.make_nd_memref_descriptor(2, index_dtype.to_ctype())),
                ("values", rt.make_nd_memref_descriptor(1, values_dtype.to_ctype())),
                ("pos_len", np.ctypeslib.c_intp),
                ("index_len", np.ctypeslib.c_intp),
                ("values_len", np.ctypeslib.c_intp),
            ]

            def to_sps(self) -> sps.coo_array:
                pos = rt.ranked_memref_to_numpy([self.pos])[: self.pos_len]
                index = rt.ranked_memref_to_numpy([self.index])[pos[0] : pos[1]]
                values = rt.ranked_memref_to_numpy([self.values])[: self.values_len]
                return sps.coo_array((values, index.T), shape=shape)

        arr = Coo()
        module.invoke(
            "disassemble",
            ctypes.pointer(ctypes.pointer(arr)),
            ctypes.pointer(ptr),
        )
        return arr.to_sps()


class CSRFormat(BaseFormat):
    @classmethod
    def get_format_str(cls) -> str:
        return "csr_format"

    @classmethod
    def get_levels(cls) -> tuple[sparse_tensor.LevelFormat, ...]:
        return (sparse_tensor.LevelFormat.dense, sparse_tensor.LevelFormat.compressed)

    @classmethod
    def get_ordering(cls) -> ir.AffineMap:
        return ir.AffineMap.get_permutation([0, 1])

    @classmethod
    def get_assemble_functions(
        cls,
        module: ir.Module,
        tensor_shaped: ir.RankedTensorType,
        values_dtype: type[DType],
        index_dtype: type[DType],
    ) -> tuple[typing.Callable, ...]:
        tensor_1d_index = tensor.RankedTensorType.get([ir.ShapedType.get_dynamic_size()], index_dtype)
        tensor_1d_values = tensor.RankedTensorType.get([ir.ShapedType.get_dynamic_size()], values_dtype)
        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(tensor_1d_index, tensor_1d_index, tensor_1d_values)
            def assemble(pos, crd, data):
                return sparse_tensor.assemble(tensor_shaped, (pos, crd), data)

            @func.FuncOp.from_py_func(tensor_shaped)
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
                return pos, crd, data, pos_len, crd_len, data_len

            @func.FuncOp.from_py_func(tensor_shaped)
            def free_tensor(tensor_shaped):
                bufferization.dealloc_tensor(tensor_shaped)

        return assemble, disassemble, free_tensor

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
    def disassemble(
        cls,
        module: ir.Module,
        ptr: ctypes.c_void_p,
        shape: list[int],
        values_dtype: type[DType],
        index_dtype: type[DType],
    ) -> sps.csr_array:
        class Csr(ctypes.Structure):
            _fields_ = [
                ("pos", rt.make_nd_memref_descriptor(1, index_dtype.to_ctype())),
                ("crd", rt.make_nd_memref_descriptor(1, index_dtype.to_ctype())),
                ("data", rt.make_nd_memref_descriptor(1, values_dtype.to_ctype())),
                ("pos_len", np.ctypeslib.c_intp),
                ("crd_len", np.ctypeslib.c_intp),
                ("data_len", np.ctypeslib.c_intp),
            ]

            def to_sps(self) -> sps.csr_array:
                pos = rt.ranked_memref_to_numpy([self.pos])[: self.pos_len]
                crd = rt.ranked_memref_to_numpy([self.crd])[: self.crd_len]
                data = rt.ranked_memref_to_numpy([self.data])[: self.data_len]
                return sps.csr_array((data, crd, pos), shape=shape)

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
            index_dtype = asdtype(obj.indptr.dtype)
        elif obj.format == "coo":
            format_class = COOFormat
            index_dtype = asdtype(obj.coords[0].dtype)
        else:
            raise Exception(f"{obj.format} SciPy format not supported.")
    elif _is_numpy_obj(obj):
        if obj.ndim == 1:
            format_class = VectorFormat
        elif obj.ndim == 2:
            format_class = Dense2DFormat
        elif obj.ndim == 3:
            format_class = Dense3DFormat
        else:
            raise Exception(f"Rank {obj.ndim} of dense tensor not supported.")
        index_dtype = Index
    else:
        raise Exception(f"{type(obj)} not supported.")

    # TODO: support proper caching
    module, tensor_type = format_class.get_module(obj.shape, values_dtype, index_dtype)

    assembled_obj = format_class.assemble(module, obj)
    return Tensor(assembled_obj, module, tensor_type, format_class.disassemble, values_dtype, index_dtype)
