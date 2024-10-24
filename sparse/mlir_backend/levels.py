import ctypes
import dataclasses
import enum
import itertools
import re
import typing

from mlir_finch import ir
from mlir_finch.dialects import sparse_tensor

import numpy as np

from ._common import (
    _hold_ref,
    fn_cache,
    free_memref,
    get_nd_memref_descr,
    numpy_to_ranked_memref,
    ranked_memref_to_numpy,
)
from ._core import ctx
from ._dtypes import DType, asdtype

_CAMEL_TO_SNAKE = [re.compile("(.)([A-Z][a-z]+)"), re.compile("([a-z0-9])([A-Z])")]

__all__ = ["LevelProperties", "LevelFormat", "StorageFormat", "Level", "get_storage_format"]


def _camel_to_snake(name: str) -> str:
    for exp in _CAMEL_TO_SNAKE:
        name = exp.sub(r"\1_\2", name)

    return name.lower()


class LevelProperties(enum.Flag):
    NonOrdered = enum.auto()
    NonUnique = enum.auto()
    SOA = enum.auto()

    def build(self) -> list[sparse_tensor.LevelProperty]:
        return [getattr(sparse_tensor.LevelProperty, _camel_to_snake(p.name)) for p in type(self) if p in self]


class LevelFormat(enum.Enum):
    Dense = "dense"
    Compressed = "compressed"
    Singleton = "singleton"

    def build(self) -> sparse_tensor.LevelFormat:
        return getattr(sparse_tensor.LevelFormat, self.value)


@dataclasses.dataclass(eq=True, frozen=True)
class Level:
    format: LevelFormat
    properties: LevelProperties = LevelProperties(0)

    def build(self):
        return sparse_tensor.EncodingAttr.build_level_type(self.format.build(), self.properties.build())


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class StorageFormat:
    levels: tuple[Level, ...]
    order: tuple[int, ...]
    pos_width: int
    crd_width: int
    dtype: DType

    @property
    def storage_rank(self) -> int:
        return len(self.levels)

    @property
    def rank(self) -> int:
        return self.storage_rank

    def __post_init__(self):
        if sorted(self.order) != list(range(self.rank)):
            raise ValueError(f"`sorted(self.order) != list(range(self.rank))`, `{self.order=}`, `{self.rank=}`.")

    @fn_cache
    def _get_mlir_type(self, *, shape: tuple[int, ...]) -> ir.RankedTensorType:
        if len(shape) != self.rank:
            raise ValueError(f"`len(shape) != self.rank`, {shape=}, {self.rank=}")
        with ir.Location.unknown(ctx):
            mlir_levels = [level.build() for level in self.levels]
            mlir_order = list(self.order)
            mlir_reverse_order = [0] * self.rank
            for i, r in enumerate(mlir_order):
                mlir_reverse_order[r] = i

            dtype = self.dtype._get_mlir_type()
            encoding = sparse_tensor.EncodingAttr.get(
                mlir_levels,
                ir.AffineMap.get_permutation(mlir_order),
                ir.AffineMap.get_permutation(mlir_reverse_order),
                self.pos_width,
                self.crd_width,
            )
            return ir.RankedTensorType.get(list(shape), dtype, encoding)

    @fn_cache
    def _get_ctypes_type(self, *, owns_memory=False):
        ptr_dtype = asdtype(getattr(np, f"int{self.pos_width}"))
        idx_dtype = asdtype(getattr(np, f"int{self.crd_width}"))

        def get_fields():
            fields = []
            compressed_counter = 0
            singleton_counter = 0
            for level, next_level in itertools.zip_longest(self.levels, self.levels[1:]):
                if LevelFormat.Compressed == level.format:
                    compressed_counter += 1
                    fields.append((f"pointers_to_{compressed_counter}", get_nd_memref_descr(1, ptr_dtype)))
                    if next_level is not None and LevelFormat.Singleton == next_level.format:
                        singleton_counter += 1
                        fields.append(
                            (
                                f"indices_{compressed_counter}_coords_{singleton_counter}",
                                get_nd_memref_descr(1, idx_dtype),
                            )
                        )
                    else:
                        fields.append((f"indices_{compressed_counter}", get_nd_memref_descr(1, idx_dtype)))

                if LevelFormat.Singleton == level.format:
                    singleton_counter += 1
                    fields.append(
                        (f"indices_{compressed_counter}_coords_{singleton_counter}", get_nd_memref_descr(1, idx_dtype))
                    )

            fields.append(("values", get_nd_memref_descr(1, self.dtype)))
            return fields

        storage_format = self

        class Storage(ctypes.Structure):
            _fields_ = get_fields()

            def to_module_arg(self) -> list:
                return [ctypes.pointer(ctypes.pointer(f)) for f in self.get__fields_()]

            def get__fields_(self) -> list:
                return [getattr(self, field[0]) for field in self._fields_]

            def get_constituent_arrays(self) -> tuple[np.ndarray, ...]:
                arrays = tuple(ranked_memref_to_numpy(field) for field in self.get__fields_())
                for arr in arrays:
                    _hold_ref(arr, self)
                return arrays

            def get_storage_format(self) -> StorageFormat:
                return storage_format

            @classmethod
            def from_constituent_arrays(cls, arrs: list[np.ndarray]) -> "Storage":
                storage = cls(*(numpy_to_ranked_memref(arr) for arr in arrs))
                for arr in arrs:
                    _hold_ref(storage, arr)
                return storage

            if owns_memory:

                def __del__(self) -> None:
                    for field in self.get__fields_():
                        free_memref(field)

        return Storage


def get_storage_format(
    *,
    levels: tuple[Level, ...],
    order: typing.Literal["C", "F"] | tuple[int, ...],
    pos_width: int,
    crd_width: int,
    dtype: DType,
) -> StorageFormat:
    levels = tuple(levels)
    if isinstance(order, str):
        if order == "C":
            order = tuple(range(len(levels)))
        if order == "F":
            order = tuple(reversed(range(len(levels))))
    return _get_storage_format(
        levels=levels,
        order=order,
        pos_width=int(pos_width),
        crd_width=int(crd_width),
        dtype=asdtype(dtype),
    )


@fn_cache
def _get_storage_format(
    *,
    levels: tuple[Level, ...],
    order: tuple[int, ...],
    pos_width: int,
    crd_width: int,
    dtype: DType,
) -> StorageFormat:
    return StorageFormat(
        levels=levels,
        order=order,
        pos_width=pos_width,
        crd_width=crd_width,
        dtype=dtype,
    )


def _is_sparse_level(lvl: Level | LevelFormat, /) -> bool:
    if isinstance(lvl, Level):
        lvl = lvl.format
    return LevelFormat.Dense != lvl


def _count_sparse_levels(format: StorageFormat) -> int:
    return sum(_is_sparse_level(lvl) for lvl in format.levels)


def _determine_levels(*formats: StorageFormat, dtype: DType, union: bool, out_ndim: int | None = None) -> StorageFormat:
    if len(formats) == 0:
        if out_ndim is None:
            out_ndim = 0
        return get_storage_format(
            levels=(Level(LevelFormat.Dense),) * out_ndim,
            order="C",
            pos_width=64,
            crd_width=64,
            dtype=dtype,
        )

    if out_ndim is None:
        out_ndim = max(fmt.rank for fmt in formats)

    n_sparse = 0
    pos_width = 0
    crd_width = 0
    op = max if union else min
    order = ()
    for fmt in formats:
        n_sparse = op(n_sparse, _count_sparse_levels(fmt))
        pos_width = max(pos_width, fmt.pos_width)
        crd_width = max(crd_width, fmt.crd_width)
        if order != "C":
            if fmt.order[: len(order)] == order:
                order = fmt.order
            elif order[: len(fmt.order)] != fmt.order:
                order = "C"

    if out_ndim < n_sparse:
        n_sparse = out_ndim

    levels = (Level(LevelFormat.Dense),) * (out_ndim - n_sparse) + (Level(LevelFormat.Compressed),) * n_sparse
    return get_storage_format(
        levels=levels,
        order=order,
        pos_width=pos_width,
        crd_width=crd_width,
        dtype=dtype,
    )
