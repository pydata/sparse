from abc import abstractmethod

from llvmlite import ir
from numba import njit
from numba.core import types, cgutils, extending
from numba.core.datamodel import registry, models
from .sparsedim import PositionIterable, AppendAssembly
from typing import Sequence, List, Tuple


class Compressed(PositionIterable, AppendAssembly):
    properties: Sequence[str] = ("pos", "crd")

    def __init__(
        self,
        *,
        pos: List[int],
        crd: List[int],
        full: bool = True,
        ordered: bool = True,
        unique: bool = True,
    ):
        self.pos: List[int] = pos
        self.crd: List[int] = crd
        self._full: bool = full
        self._ordered: bool = ordered
        self._unique: bool = unique

    @property
    def full(self) -> bool:
        return self._full

    @property
    def ordered(self) -> bool:
        return self._ordered

    @property
    def unique(self) -> bool:
        return self._unique

    @property
    def branchless(self) -> bool:
        return False

    @property
    def compact(self) -> bool:
        return True

    @jit
    def pos_bounds(self, pkm1: int) -> Tuple[int, int]:
        return self.pos[pkm1], self.pos[pkm1] + 1

    @jit
    def pos_access(self, pk: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return self.crd[pk], True

    @jit
    def append_coord(self, pk: int, ik: int) -> None:
        self.crd.append(ik)

    @jit
    def append_edges(self, pkm1: int, pkbegin: int, pkend: int) -> None:
        self.pos.append(pkend - pkbegin)

    @jit
    def append_init(self, szkm1: int, szk: int) -> None:
        for _ in range(szkm1 + 1):
            self.pos.append(0)

    @jit
    def append_finalize(self, szkm1: int, szk: int) -> None:
        cumsum: int = self.pos[0]
        for pkm1 in range(1, szkm1 + 1):
            cumsum += self.pos[pkm1]
            self.pos[pkm1] = cumsum


class CompressedType(SparseDimType):

    # Type is mutable
    mutable = True

    def __init__(
        self,
        *,
        full: bool,
        ordered: bool,
        unique: bool,
        pos_type: types.Integer,
        crd_type: types.Integer,
    ):
        if not isinstance(pos_type, types.Integer):
            raise TypeError("pos_type must be a numba.types.Integer.")

        if not isinstance(crd_type, types.Integer):
            raise TypeError("crd_type must be a numba.types.Integer.")

        self.full: bool = bool(full)
        self.ordered: bool = bool(ordered)
        self.unique: bool = bool(unique)
        self.pos_type: types.Integer = pos_type
        self.crd_type: types.Integer = crd_type
        name: str = f"Compressed<{pos_type}, {crd_type}>"
        super().__init__(name)

    @property
    def key(self):
        return (self.full, self.ordered, self.unique, self.pos_type, self.crd_type)

    @property
    def full(self) -> bool:
        return self._full

    @property
    def ordered(self) -> bool:
        return self._ordered

    @property
    def unique(self) -> bool:
        return self._unique

    @property
    def branchless(self) -> bool:
        return False

    @property
    def compact(self) -> bool:
        return True


@registry.register_default(CompressedType)
class CompressedModel(models.StructModel):
    def __init__(self, dmm, fe_type: CompressedType):
        members = [
            ("pos", types.ListType[fe_type.pos_type]),
            ("crd", types.ListType[fe_type.crd_type]),
        ]
        super().__init__(dmm, fe_type, members)
