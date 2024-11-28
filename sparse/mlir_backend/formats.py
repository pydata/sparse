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

__all__ = ["LevelProperties", "LevelFormat", "ConcreteFormat", "Level", "get_concrete_format"]


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
class ConcreteFormat:
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

            def get_storage_format(self) -> ConcreteFormat:
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


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class FormatFactory:
    levels: tuple[Level, ...] | None = None
    order: typing.Literal["C", "F"] | tuple[int, ...] = "C"
    pos_width: int = 64
    crd_width: int = 64
    dtype: DType | None = None

    def is_ready(self) -> bool:
        fields = dataclasses.fields(self)
        return all(getattr(self, f.name) is not None for f in fields)

    def build(self) -> ConcreteFormat:
        if not self.is_ready():
            raise RuntimeError("This factory is not ready. All fields must be non-None.")

        return get_concrete_format(
            levels=self.levels,
            order=self.order,
            pos_width=self.pos_width,
            crd_width=self.crd_width,
            dtype=self.dtype,
        )

    @classmethod
    def _get_levels_from_ndim(cls, ndim: int, /) -> tuple[Level, ...]:
        raise TypeError(f"`{cls.__name__}` doesn't implement this method.")

    def with_ndim(self, ndim: int, /, *, canonical: bool = True) -> "FormatFactory":
        if ndim < 0:
            raise ValueError(f"`ndim < 0`, `{ndim=}`.")

        levels = self._get_levels_from_ndim(ndim)
        if not canonical:
            levels = tuple(
                dataclasses.replace(
                    level, properties=level.properties | LevelProperties.NonOrdered | LevelProperties.NonUnique
                )
                for level in levels
            )

        assert len(levels) == ndim
        return self.with_levels(levels)

    def with_levels(self, levels: tuple[Level, ...], /) -> "FormatFactory":
        out = dataclasses.replace(self, levels=levels)
        out._check_consistency()
        return out

    def _check_consistency(self) -> None:
        order = self.order
        if isinstance(order, str):
            if order in {"C", "F"}:
                return

            raise ValueError(f"Invalid order, `{order=}`.")

        if sorted(order) != list(range(len(order))):
            raise ValueError(f"`sorted(order) != list(range(len(order)))`, `{order=}`.")

        levels = self.levels
        if levels is not None and len(levels) != len(order):
            raise ValueError(f"`levels is not None and len(levels) != len(order)`, `{order=}`, `{levels=}`.")

    def with_order(self, order: typing.Literal["C", "F"] | tuple[int, ...], /):
        out = dataclasses.replace(self, order=order)
        out._check_consistency()
        return out

    def with_ptr_width(self, width: int, /) -> "FormatFactory":
        return dataclasses.replace(self, pos_width=width, crd_width=width)

    def with_pos_width(self, width: int, /) -> "FormatFactory":
        return dataclasses.replace(self, pos_width=width)

    def with_crd_width(self, width: int, /) -> "FormatFactory":
        return dataclasses.replace(self, crd_width=width)

    def with_dtype(self, dtype: DType, /) -> "FormatFactory":
        return dataclasses.replace(self, dtype=dtype)

    @classmethod
    def is_this_format(cls, format: ConcreteFormat) -> bool:
        levels_self = cls._get_levels_from_ndim(format.storage_rank)
        levels_other = format.levels

        return all(
            dataclasses.replace(l1, properties=l1.properties | LevelProperties.NonOrdered | LevelProperties.NonUnique)
            == dataclasses.replace(
                l2, properties=l2.properties | LevelProperties.NonOrdered | LevelProperties.NonUnique
            )
            for l1, l2 in zip(levels_self, levels_other, strict=True)
        )


class Coo(FormatFactory):
    @classmethod
    def _get_levels_from_ndim(cls, ndim: int, /) -> tuple[Level, ...]:
        if ndim == 0:
            return ()

        level_base = Level(LevelFormat.Compressed)
        level_middle = Level(LevelFormat.Singleton, LevelProperties.SOA)

        levels = []

        for i in range(ndim):
            level = level_base if i == 0 else level_middle
            if i != ndim - 1:
                level = dataclasses.replace(level, properties=level.properties | LevelProperties.NonUnique)
            levels.append(level)

        return tuple(levels)


class Csf(FormatFactory):
    @classmethod
    def _get_levels_from_ndim(self, ndim: int, /) -> tuple[Level, ...]:
        if ndim == 0:
            return ()

        level_middle = Level(LevelFormat.Compressed)
        level_base = Level(LevelFormat.Dense)
        levels = []

        for i in range(ndim):
            level = level_base if i == 0 else level_middle
            levels.append(level)

        return tuple(levels)


class Dense(FormatFactory):
    @classmethod
    def _get_levels_from_ndim(self, ndim: int, /) -> tuple[Level, ...]:
        return (Level(LevelFormat.Dense),) * ndim


def get_concrete_format(
    *,
    levels: tuple[Level, ...],
    order: typing.Literal["C", "F"] | tuple[int, ...],
    pos_width: int,
    crd_width: int,
    dtype: DType,
) -> ConcreteFormat:
    levels = tuple(levels)
    if isinstance(order, str):
        if order == "C":
            order = tuple(range(len(levels)))
        if order == "F":
            order = tuple(reversed(range(len(levels))))
    return _get_concrete_format(
        levels=levels,
        order=order,
        pos_width=int(pos_width),
        crd_width=int(crd_width),
        dtype=asdtype(dtype),
    )


@fn_cache
def _get_concrete_format(
    *,
    levels: tuple[Level, ...],
    order: tuple[int, ...],
    pos_width: int,
    crd_width: int,
    dtype: DType,
) -> ConcreteFormat:
    return ConcreteFormat(
        levels=levels,
        order=order,
        pos_width=pos_width,
        crd_width=crd_width,
        dtype=dtype,
    )


def _is_sparse_level(lvl: Level | LevelFormat, /) -> bool:
    assert isinstance(lvl, Level | LevelFormat)
    if isinstance(lvl, Level):
        lvl = lvl.format
    return LevelFormat.Dense != lvl


def _count_sparse_levels(format: ConcreteFormat) -> int:
    return sum(_is_sparse_level(lvl) for lvl in format.levels)


def _count_dense_levels(format: ConcreteFormat) -> int:
    return sum(not _is_sparse_level(lvl) for lvl in format.levels)


def _get_sparse_dense_levels(
    *, n_sparse: int | None = None, n_dense: int | None = None, ndim: int | None = None
) -> tuple[Level, ...]:
    if (n_sparse is not None) + (n_dense is not None) + (ndim is not None) != 2:
        assert n_sparse is not None and n_dense is not None and ndim is not None  #
        assert n_sparse + n_dense == ndim
    if n_sparse is None:
        n_sparse = ndim - n_dense
    if n_dense is None:
        n_dense = ndim - n_sparse
    if ndim is None:
        ndim = n_dense + n_sparse

    assert ndim >= 0
    assert n_dense >= 0
    assert n_sparse >= 0

    return (Level(LevelFormat.Dense),) * n_dense + (Level(LevelFormat.Compressed),) * n_sparse


def _determine_format(
    *formats: ConcreteFormat, dtype: DType, union: bool, out_ndim: int | None = None
) -> ConcreteFormat:
    """Determines the output format from a group of input formats.

    1. Counts the sparse levels for `union=True`, and dense ones for `union=False`.
    2. Gets the max number of counted levels for each format.
    3. Constructs a format with rank of `out_ndim` (max rank of inputs is taken if it's `None`).
       If `union=False` counted levels is the number of sparse levels, otherwise dense.
       Sparse levels are replaced with `LevelFormat.Compressed`.

    Returns
    -------
    StorageFormat
        Output storage format.
    """
    if len(formats) == 0:
        if out_ndim is None:
            out_ndim = 0
        return get_concrete_format(
            levels=(Level(LevelFormat.Dense if union else LevelFormat.Compressed),) * out_ndim,
            order="C",
            pos_width=64,
            crd_width=64,
            dtype=dtype,
        )

    if out_ndim is None:
        out_ndim = max(fmt.rank for fmt in formats)

    pos_width = 0
    crd_width = 0
    counter = _count_sparse_levels if not union else _count_dense_levels
    n_counted = None
    order = ()
    for fmt in formats:
        n_counted = counter(fmt) if n_counted is None else max(n_counted, counter(fmt))
        pos_width = max(pos_width, fmt.pos_width)
        crd_width = max(crd_width, fmt.crd_width)
        if order != "C":
            if fmt.order[: len(order)] == order:
                order = fmt.order
            elif order[: len(fmt.order)] != fmt.order:
                order = "C"

    if not isinstance(order, str):
        order = order + tuple(range(len(order), out_ndim))
        order = order[:out_ndim]

    if out_ndim < n_counted:
        n_counted = out_ndim

    n_sparse = n_counted if not union else out_ndim - n_counted

    levels = _get_sparse_dense_levels(n_sparse=n_sparse, ndim=out_ndim)
    return get_concrete_format(
        levels=levels,
        order=order,
        pos_width=pos_width,
        crd_width=crd_width,
        dtype=dtype,
    )
