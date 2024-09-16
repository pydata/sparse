import ctypes

import mlir.runtime as rt
from mlir import ir
from mlir.dialects import sparse_tensor

import numpy as np
import scipy.sparse as sps

from ._common import _hold_self_ref_in_ret, _take_owneship, fn_cache
from ._core import ctx, libc
from ._dtypes import DType, asdtype

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


def free_memref(obj: ctypes.Structure) -> None:
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

            csr_instance = cls(indptr=indptr, indices=indices, data=data)
            _take_owneship(csr_instance, arr)

            return csr_instance

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

        def get__fields_(self) -> list:
            return [self.indptr, self.indices, self.data]

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
        def from_sps(cls, arr: sps.coo_array) -> "Coo":
            assert arr.has_canonical_format, "COO must have canonical format"
            np_pos = np.array([0, arr.size], dtype=index_dtype.np_dtype)
            np_coords = np.stack(arr.coords, axis=1, dtype=index_dtype.np_dtype)
            pos = numpy_to_ranked_memref(np_pos)
            coords = numpy_to_ranked_memref(np_coords)
            data = numpy_to_ranked_memref(arr.data)

            coo_instance = cls(pos=pos, coords=coords, data=data)
            _take_owneship(coo_instance, np_pos)
            _take_owneship(coo_instance, np_coords)
            _take_owneship(coo_instance, arr)

            return coo_instance

        def to_sps(self, shape: tuple[int, ...]) -> sps.coo_array:
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

        def get__fields_(self) -> list:
            return [self.pos, self.coords, self.data]

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
def get_dense_class(values_dtype: type[DType], index_dtype: type[DType]) -> type:
    class Dense(ctypes.Structure):
        _fields_ = [
            ("data", get_nd_memref_descr(1, values_dtype)),
        ]
        dtype = values_dtype
        _index_dtype = index_dtype

        @classmethod
        def from_sps(cls, arr: np.ndarray) -> "Dense":
            data = numpy_to_ranked_memref(arr.ravel())

            dense_instance = cls(data=data)
            _take_owneship(dense_instance, arr)

            return dense_instance

        def to_sps(self, shape: tuple[int, ...]) -> sps.csr_array:
            data = ranked_memref_to_numpy(self.data)
            return data.reshape(shape)

        def to_module_arg(self) -> list:
            return [ctypes.pointer(ctypes.pointer(self.data))]

        def get__fields_(self) -> list:
            return [self.data]

        @classmethod
        @fn_cache
        def get_tensor_definition(cls, shape: tuple[int, ...]) -> ir.RankedTensorType:
            with ir.Location.unknown(ctx):
                values_dtype = cls.dtype.get_mlir_type()
                index_dtype = cls._index_dtype.get_mlir_type()
                index_width = getattr(index_dtype, "width", 0)
                levels = (sparse_tensor.LevelFormat.dense,) * len(shape)
                ordering = ir.AffineMap.get_permutation([*range(len(shape))])
                encoding = sparse_tensor.EncodingAttr.get(levels, ordering, ordering, index_width, index_width)
                return ir.RankedTensorType.get(list(shape), values_dtype, encoding)

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
        self._values_dtype = asdtype(obj.dtype)

        if _is_scipy_sparse_obj(obj):
            self._owns_memory = False

            if obj.format == "csr":
                index_dtype = asdtype(obj.indptr.dtype)
                self._format_class = get_csr_class(self._values_dtype, index_dtype)
                self._obj = self._format_class.from_sps(obj)
            elif obj.format == "coo":
                index_dtype = asdtype(obj.coords[0].dtype)
                self._format_class = get_coo_class(self._values_dtype, index_dtype)
                self._obj = self._format_class.from_sps(obj)
            else:
                raise Exception(f"{obj.format} SciPy format not supported.")

        elif _is_numpy_obj(obj):
            self._owns_memory = False
            index_dtype = asdtype(np.intp)
            self._format_class = get_dense_class(self._values_dtype, index_dtype)
            self._obj = self._format_class.from_sps(obj)

        elif _is_mlir_obj(obj):
            self._owns_memory = True
            self._format_class = type(obj)
            self._obj = obj

        else:
            raise Exception(f"{type(obj)} not supported.")

    def __del__(self):
        if self._owns_memory:
            for field in self._obj.get__fields_():
                free_memref(field)

    @_hold_self_ref_in_ret
    def to_scipy_sparse(self) -> sps.sparray | np.ndarray:
        return self._obj.to_sps(self.shape)


def asarray(obj) -> Tensor:
    return Tensor(obj)
