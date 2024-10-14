import ctypes
import dataclasses
import enum
import itertools
import re
import typing

import mlir.runtime as rt
from mlir import ir
from mlir.dialects import sparse_tensor

import numpy as np

from ._common import (
    PackedArgumentTuple,
    _take_owneship,
    fn_cache,
    numpy_to_ranked_memref,
    ranked_memref_to_numpy,
)
from ._dtypes import DType, asdtype

_CAMEL_TO_SNAKE = [re.compile("(.)([A-Z][a-z]+)"), re.compile("([a-z0-9])([A-Z])")]


def _camel_to_snake(name: str) -> str:
    for exp in _CAMEL_TO_SNAKE:
        name = exp.sub(r"\1_\2", name)

    return name.lower()


@fn_cache
def get_nd_memref_descr(rank: int, dtype: type[DType]) -> type:
    return rt.make_nd_memref_descriptor(rank, dtype.to_ctype())


class LevelProperties(enum.Flag):
    NonOrdered = enum.auto()
    NonUnique = enum.auto()

    def build(self) -> list[sparse_tensor.LevelProperty]:
        return [getattr(sparse_tensor.LevelProperty, _camel_to_snake(p.name)) for p in type(self) if p in self]


class LevelFormat(enum.Enum):
    Dense = "dense"
    Compressed = "compressed"
    Singleton = "singleton"

    def build(self) -> sparse_tensor.LevelFormat:
        return getattr(sparse_tensor.LevelFormat, self.value)


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class Level:
    format: LevelFormat
    properties: LevelProperties = LevelProperties(0)

    def build(self):
        sparse_tensor.EncodingAttr.build_level_type(self.format.build(), self.properties.build())


@dataclasses.dataclass(kw_only=True)
class StorageFormat:
    levels: tuple[Level, ...]
    order: typing.Literal["C", "F"] | tuple[int, ...]
    pos_width: int
    crd_width: int
    dtype: type[DType]

    @property
    def storage_rank(self) -> int:
        return len(self.levels)

    @property
    def rank(self) -> int:
        return self.storage_rank

    def __post_init__(self):
        rank = self.storage_rank
        self.dtype = asdtype(self.dtype)
        if self.order == "C":
            self.order = tuple(range(rank))
            return

        if self.order == "F":
            self.order = tuple(reversed(range(rank)))
            return

        if sorted(self.order) != list(range(rank)):
            raise ValueError(f"`sorted(self.order) != list(range(rank))`, {self.order=}, {rank=}.")

        self.order = tuple(self.order)

    @fn_cache
    def get_mlir_type(self, *, shape: tuple[int, ...]) -> ir.RankedTensorType:
        if len(shape) != self.rank:
            raise ValueError(f"`len(shape) != self.rank`, {shape=}, {self.rank=}")
        mlir_levels = [level.build() for level in self.levels]
        mlir_order = list(self.order)
        mlir_reverse_order = [0] * self.rank
        for i, r in enumerate(mlir_order):
            mlir_reverse_order[r] = i

        dtype = self.dtype.get_mlir_type()
        encoding = sparse_tensor.EncodingAttr.get(
            mlir_levels, mlir_order, mlir_reverse_order, self.pos_width, self.crd_width
        )
        return ir.RankedTensorType.get(list(shape), dtype, encoding)

    @fn_cache
    def get_ctypes_type(self):
        ptr_dtype = asdtype(getattr(np, f"uint{self.pos_width}"))
        idx_dtype = asdtype(getattr(np, f"uint{self.crd_width}"))

        def get_fields():
            fields = []
            compressed_counter = 0
            for level, next_level in itertools.zip_longest(self.levels, self.levels[1:]):
                if LevelFormat.Compressed == level.format:
                    compressed_counter += 1
                    fields.append((f"pointers_to_{compressed_counter}", get_nd_memref_descr(1, ptr_dtype)))
                    if next_level is not None and LevelFormat.Singleton == next_level.format:
                        fields.append((f"indices_{compressed_counter}", get_nd_memref_descr(2, idx_dtype)))
                    else:
                        fields.append((f"indices_{compressed_counter}", get_nd_memref_descr(1, idx_dtype)))

            fields.append(("values", get_nd_memref_descr(1, self.dtype.np_dtype)))
            return fields

        storage_format = self

        class Format(ctypes.Structure):
            _fields_ = get_fields()

            def get_mlir_type(self, *, shape: tuple[int, ...]):
                return self.get_storage_format().get_mlir_type(shape=shape)

            def to_module_arg(self) -> list:
                return [ctypes.pointer(ctypes.pointer(f) for f in self.get__fields_())]

            def get__fields_(self) -> list:
                return [getattr(self, field[0]) for field in self._fields_]

            def to_constituent_arrays(self) -> PackedArgumentTuple:
                return PackedArgumentTuple(tuple(ranked_memref_to_numpy(field) for field in self.get__fields_()))

            def get_storage_format(self) -> StorageFormat:
                return storage_format

            @classmethod
            def from_constituent_arrays(cls, arrs: list[np.ndarray]) -> "Format":
                inst = cls(*(numpy_to_ranked_memref(arr) for arr in arrs))
                for arr in arrs:
                    _take_owneship(inst, arr)
                return inst

        return Format

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, value):
        return self is value
