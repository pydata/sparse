import ctypes
import weakref

import mlir.runtime as rt
from mlir import ir
from mlir.dialects import sparse_tensor

import numpy as np
import scipy.sparse as sps

from ._common import fn_cache
from ._core import ctx, libc
from ._dtypes import DType, asdtype


def _take_owneship(owner, obj):
    ptr = ctypes.py_object(obj)
    ctypes.pythonapi.Py_IncRef(ptr)

    def finalizer(ptr):
        ctypes.pythonapi.Py_DecRef(ptr)

    weakref.finalize(owner, finalizer, ptr)


###########
# Memrefs #
###########


@fn_cache
def get_nd_memref_descr(rank: int, dtype: type[DType]) -> type:
    return rt.make_nd_memref_descriptor(rank, dtype.to_ctype())


def numpy_to_ranked_memref(arr: np.ndarray) -> ctypes.Structure:
    memref = rt.get_ranked_memref_descriptor(arr)
    memref_descr = get_nd_memref_descr(arr.ndim, asdtype(arr.dtype))
    # Required due to ctypes type checks
    return memref_descr(
        allocated=memref.allocated,
        aligned=memref.aligned,
        offset=memref.offset,
        shape=memref.shape,
        strides=memref.strides,
    )


def ranked_memref_to_numpy(ref: ctypes.Structure) -> np.ndarray:
    return rt.ranked_memref_to_numpy([ref])


def freeme(obj: ctypes.Structure) -> None:
    # TODO: I think there's still a memory leak
    libc.free(ctypes.cast(obj.allocated, ctypes.c_void_p))


###########
# Formats #
###########


@fn_cache
def get_csr_class(values_dtype: type[DType], index_dtype: type[DType]) -> type:
    class Csr(ctypes.Structure):
        _fields_ = [
            ("indptr", get_nd_memref_descr(1, index_dtype)),
            ("indices", get_nd_memref_descr(1, index_dtype)),
            ("data", get_nd_memref_descr(1, values_dtype)),
        ]
        dtype = values_dtype
        _index_dtype = index_dtype

        @classmethod
        def from_sps(cls, arr: sps.csr_array) -> "Csr":
            indptr = numpy_to_ranked_memref(arr.indptr)
            indices = numpy_to_ranked_memref(arr.indices)
            data = numpy_to_ranked_memref(arr.data)
            return cls(indptr=indptr, indices=indices, data=data)

        def to_sps(self, shape: tuple[int, ...]) -> sps.csr_array:
            pos = ranked_memref_to_numpy(self.indptr)
            crd = ranked_memref_to_numpy(self.indices)
            data = ranked_memref_to_numpy(self.data)
            return sps.csr_array((data, crd, pos), shape=shape)

        def to_module_arg(self) -> list:
            return [
                ctypes.pointer(ctypes.pointer(self.indptr)),
                ctypes.pointer(ctypes.pointer(self.indices)),
                ctypes.pointer(ctypes.pointer(self.data)),
            ]

        @classmethod
        @fn_cache
        def get_tensor_definition(cls, shape: tuple[int, ...]) -> ir.RankedTensorType:
            with ir.Location.unknown(ctx):
                values_dtype = cls.dtype.get_mlir_type()
                index_dtype = cls._index_dtype.get_mlir_type()
                index_width = getattr(index_dtype, "width", 0)
                levels = (sparse_tensor.LevelFormat.dense, sparse_tensor.LevelFormat.compressed)
                ordering = ir.AffineMap.get_permutation([0, 1])
                encoding = sparse_tensor.EncodingAttr.get(levels, ordering, ordering, index_width, index_width)
                return ir.RankedTensorType.get(list(shape), values_dtype, encoding)

    return Csr


@fn_cache
def get_coo_class(values_dtype: type[DType], index_dtype: type[DType]) -> type:
    class Coo(ctypes.Structure):
        _fields_ = [
            ("pos", get_nd_memref_descr(1, index_dtype)),
            ("coords", get_nd_memref_descr(2, index_dtype)),
            ("data", get_nd_memref_descr(1, values_dtype)),
        ]
        dtype = values_dtype
        _index_dtype = index_dtype

        @classmethod
        def from_sps(cls, arr: sps.csr_array) -> "Coo":
            np_pos = np.array([0, arr.size], dtype=index_dtype.np_dtype)
            np_coords = np.stack(arr.coords, axis=1, dtype=index_dtype.np_dtype)
            pos = numpy_to_ranked_memref(np_pos)
            coords = numpy_to_ranked_memref(np_coords)
            data = numpy_to_ranked_memref(arr.data)

            coo_instance = cls(pos=pos, coords=coords, data=data)
            _take_owneship(coo_instance, np_pos)
            _take_owneship(coo_instance, np_coords)

            return coo_instance

        def to_sps(self, shape: tuple[int, ...]) -> sps.csr_array:
            pos = ranked_memref_to_numpy(self.pos)
            coords = ranked_memref_to_numpy(self.coords)[pos[0] : pos[1]]
            data = ranked_memref_to_numpy(self.data)
            return sps.coo_array((data, coords.T), shape=shape)

        def to_module_arg(self) -> list:
            return [
                ctypes.pointer(ctypes.pointer(self.pos)),
                ctypes.pointer(ctypes.pointer(self.coords)),
                ctypes.pointer(ctypes.pointer(self.data)),
            ]

        @classmethod
        @fn_cache
        def get_tensor_definition(cls, shape: tuple[int, ...]) -> ir.RankedTensorType:
            with ir.Location.unknown(ctx):
                values_dtype = cls.dtype.get_mlir_type()
                index_dtype = cls._index_dtype.get_mlir_type()
                index_width = getattr(index_dtype, "width", 0)
                compressed_lvl = sparse_tensor.EncodingAttr.build_level_type(
                    sparse_tensor.LevelFormat.compressed, [sparse_tensor.LevelProperty.non_unique]
                )
                levels = (compressed_lvl, sparse_tensor.LevelFormat.singleton)
                ordering = ir.AffineMap.get_permutation([0, 1])
                encoding = sparse_tensor.EncodingAttr.get(levels, ordering, ordering, index_width, index_width)
                return ir.RankedTensorType.get(list(shape), values_dtype, encoding)

    return Coo


@fn_cache
def get_csf_class(values_dtype: type[DType], index_dtype: type[DType]) -> type:
    raise NotImplementedError


@fn_cache
def get_dense_class(values_dtype: type[DType], rank: int) -> type:
    class Dense(ctypes.Structure):
        _fields_ = [
            ("data", get_nd_memref_descr(rank, values_dtype)),
        ]
        dtype = values_dtype

        @classmethod
        def from_sps(cls, arr: np.ndarray) -> "Dense":
            data = numpy_to_ranked_memref(arr)
            return cls(data=data)

        def to_sps(self, shape: tuple[int, ...]) -> sps.csr_array:
            return ranked_memref_to_numpy(self.data)

        def to_module_arg(self) -> list:
            return [ctypes.pointer(ctypes.pointer(self.data))]

        @classmethod
        @fn_cache
        def get_tensor_definition(cls, shape: tuple[int, ...]) -> ir.RankedTensorType:
            with ir.Location.unknown(ctx):
                values_dtype = cls.dtype.get_mlir_type()
                return ir.RankedTensorType.get(list(shape), values_dtype)

    return Dense


def _is_scipy_sparse_obj(x) -> bool:
    return hasattr(x, "__module__") and x.__module__.startswith("scipy.sparse")


def _is_numpy_obj(x) -> bool:
    return isinstance(x, np.ndarray)


def _is_mlir_obj(x) -> bool:
    return isinstance(x, ctypes.Structure)


################
# Tensor class #
################


class Tensor:
    def __init__(self, obj, shape=None) -> None:
        self.shape = shape if shape is not None else obj.shape
        self.values_dtype = asdtype(obj.dtype)

        if _is_scipy_sparse_obj(obj):
            self.owns_memory = True

            if obj.format == "csr":
                index_dtype = asdtype(obj.indptr.dtype)
                self.format_class = get_csr_class(self.values_dtype, index_dtype)
                self.obj = self.format_class.from_sps(obj)
            elif obj.format == "coo":
                index_dtype = asdtype(obj.coords[0].dtype)
                self.format_class = get_coo_class(self.values_dtype, index_dtype)
                self.obj = self.format_class.from_sps(obj)
            else:
                raise Exception(f"{obj.format} SciPy format not supported.")

        elif _is_numpy_obj(obj):
            self.owns_memory = True
            self.format_class = get_dense_class(self.values_dtype, obj.ndim)
            self.obj = self.format_class.from_sps(obj)

        elif _is_mlir_obj(obj):
            self.owns_memory = False
            self.format_class = type(obj)
            self.obj = obj

        else:
            raise Exception(f"{type(obj)} not supported.")

    def to_scipy_sparse(self) -> sps.sparray | np.ndarray:
        return self.obj.to_sps(self.shape)


def asarray(obj) -> Tensor:
    return Tensor(obj)
