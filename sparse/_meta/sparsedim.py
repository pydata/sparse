from abc import abstractmethod, ABC
from functools import lru_cache, reduce
import numba as nb
from collections import namedtuple
from typing import Sequence, List, Tuple

__all__ = ["SparseDim"]

jit = nb.jit(nopython=True, nogil=True, inline="always")


class SparseDim(ABC):
    properties: Sequence[str]

    @classmethod
    @lru_cache
    def named_tuple(cls) -> namedtuple:
        return namedtuple(type(cls).__name__, cls.properties)

    @property
    def v(self):
        kwargs = {p: getattr(self, p) for p in self.properties}
        return self.named_tuple()(**kwargs)

    @property
    @abstractmethod
    def full(self) -> bool:
        pass

    @property
    @abstractmethod
    def ordered(self) -> bool:
        pass

    @property
    @abstractmethod
    def unique(self) -> bool:
        pass

    @property
    @abstractmethod
    def branchless(self) -> bool:
        pass

    @property
    @abstractmethod
    def compact(self) -> bool:
        pass

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) == type(other)

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return repr(self.v)


class ValueIterable(SparseDim, ABC):
    @abstractmethod
    def coord_bounds(self, i: Tuple[int, ...]) -> Tuple[int, int]:
        pass

    @abstractmethod
    def coord_access(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        pass


class PositionIterable(SparseDim, ABC):
    @abstractmethod
    def pos_bounds(self, pkm1: int) -> Tuple[int, int]:
        pass

    @abstractmethod
    def pos_access(self, pk: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        pass


class HasLocate(SparseDim, ABC):
    @abstractmethod
    def locate(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        pass


class HasInlineAssemly(SparseDim, ABC):
    @abstractmethod
    def size(self, szkm1: int) -> int:
        pass

    @abstractmethod
    def insert_coord(self, pk: int, ik: int) -> None:
        pass

    @abstractmethod
    def insert_init(self, szkm1: int, szk: int) -> None:
        pass

    @abstractmethod
    def insert_finalize(self, szkm1: int, szk: int) -> None:
        pass


class HasAppendAssembly(SparseDim, ABC):
    @abstractmethod
    def append_coord(self, pk: int, ik: int) -> None:
        pass

    @abstractmethod
    def append_edges(self, pkm1: int, pkbegin: int, pkend: int) -> None:
        pass

    @abstractmethod
    def append_init(self, szkm1: int, szk: int) -> None:
        pass

    @abstractmethod
    def append_finalize(self, szkm1: int, szk: int) -> None:
        pass


class Dense(HasLocate, ValueIterable, HasInlineAssemly):
    properties: Sequence[str] = ("N",)

    def __init__(self, *, N: int, ordered: bool = True, unique: bool = True):
        self.N: int = N
        self._ordered: bool = ordered
        self._unique: bool = unique

    @property
    def full(self) -> bool:
        return True

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
    def locate(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return pkm1 * self.N + i[-1], True

    @jit
    def coord_bounds(self, i: Tuple[int, ...]) -> Tuple[int, int]:
        return 0, self.N

    @jit
    def coord_access(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return pkm1 * self.N + i[-1], True

    @jit
    def size(self, szkm1: int) -> int:
        return szkm1 * self.N

    @jit
    def insert_coord(self, pk: int, ik: int) -> None:
        pass

    @jit
    def insert_init(self, szkm1: int, szk: int) -> None:
        pass

    @jit
    def insert_finalize(self, szkm1: int, szk: int) -> None:
        pass


class Range(ValueIterable):
    properties: Sequence[str] = ("pos", "crd")

    def __init__(
        self,
        *,
        offset: Sequence[int],
        N: int,
        M: int,
        ordered: bool = True,
        unique: bool = True
    ):
        self.offset: Sequence[int] = offset
        self.N: int = N
        self.M: int = M
        self._ordered: bool = ordered
        self._unique: bool = unique

    @property
    def full(self) -> bool:
        return False

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
        return False

    @jit
    def coord_bounds(self, i: Tuple[int, ...]) -> Tuple[int, int]:
        return max(0, -self.offset[i[-1]]), min(self.N, self.M - self.offset[i[-1]])

    @jit
    def coord_access(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return pkm1 * self.N + i[-1], True


class Compressed(PositionIterable, HasAppendAssembly):
    properties: Sequence[str] = ("pos", "crd")

    def __init__(
        self,
        *,
        pos: List[int],
        crd: List[int],
        full: bool = True,
        ordered: bool = True,
        unique: bool = True
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
        self.crd[pk] = ik

    @jit
    def append_edges(self, pkm1: int, pkbegin: int, pkend: int) -> None:
        self.pos[pkm1 + 1] = pkend - pkbegin

    @jit
    def append_init(self, szkm1: int, szk: int) -> None:
        for pkm1 in range(szkm1 + 1):
            self.pos[pkm1] = 0

    @jit
    def append_finalize(self, szkm1: int, szk: int) -> None:
        cumsum: int = self.pos[0]
        for pkm1 in range(1, szkm1 + 1):
            cumsum += self.pos[pkm1]
            self.pos[pkm1] = cumsum


class Singleton(PositionIterable, HasAppendAssembly):
    properties: Sequence[str] = ("crd",)

    def __init__(self, *, crd: List[int], full: bool = True, ordered: bool = True, unique: bool = True):
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
        return True

    @property
    def compact(self) -> bool:
        return True

    @jit
    def pos_bounds(self, pkm1: int) -> Tuple[int, int]:
        return pkm1, pkm1 + 1

    @jit
    def pos_access(self, pk: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return self.crd[pk], True

    @jit
    def append_coord(self, pk: int, ik: int) -> None:
        self.crd[pk] = ik

    @jit
    def append_edges(self, pkm1: int, pkbegin: int, pkend: int) -> None:
        pass

    @jit
    def append_init(self, szkm1: int, szk: int) -> None:
        pass

    @jit
    def append_finalize(self, szkm1: int, szk: int) -> None:
        pass


class Offset(PositionIterable):
    properties: Sequence[str] = ("offset",)

    def __init__(self, *, offset: List[int], ordered: bool = True, unique: bool = True):
        self.offset: List[int] = offset
        self._ordered: bool = ordered
        self._unique: bool = unique

    @property
    def full(self) -> bool:
        return False

    @property
    def ordered(self) -> bool:
        return self._ordered

    @property
    def unique(self) -> bool:
        return self._unique

    @property
    def branchless(self) -> bool:
        return True

    @property
    def compact(self) -> bool:
        return False

    @jit
    def pos_bounds(self, pkm1: int) -> Tuple[int, int]:
        return pkm1, pkm1 + 1

    @jit
    def pos_access(self, pk: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return i[-1] + self.offset[i[-2]], True


class Hashed(HasLocate, PositionIterable, HasInlineAssemly):
    properties: Sequence[str] = ("W", "crd")

    def __init__(self, *, W: int, crd: List[int], full: bool = True, unique: bool = True):
        self.W: int = W
        self.crd: List[int] = crd
        self._full: bool = full
        self._unique: bool = unique

    @property
    def full(self) -> bool:
        return self._full

    @property
    def ordered(self) -> bool:
        return False

    @property
    def unique(self) -> bool:
        return self._unique

    @property
    def branchless(self) -> bool:
        return False

    @property
    def compact(self) -> bool:
        return False

    @jit
    def locate(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        pk: int = i[-1] % self.W + pkm1 * self.W
        if self.crd[pk] != i[-1] and self.crd[pk] != -1:
            end: int = pk
            pk = (pk + 1) % self.W + pkm1 * self.W
            while self.crd[pk] != i[-1] and self.crd[pk] != -1 and pk != end:
                pk = (pk + 1) % self.W + pkm1 * self.W

        return pk, self.crd[pk] == i[-1]

    @jit
    def pos_bounds(self, pkm1: int) -> Tuple[int, int]:
        return pkm1 * self.W, (pkm1 + 1) * self.W

    @jit
    def pos_access(self, pk: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return self.crd[pk], self.crd[pk] != -1

    @jit
    def size(self, szkm1: int) -> int:
        return szkm1 * self.W

    @jit
    def insert_coord(self, pk: int, ik: int) -> None:
        self.crd[pk] = ik

    @jit
    def insert_init(self, szkm1: int, szk: int) -> None:
        for pk in range(szk):
            self.crd[pk] = -1

    @jit
    def insert_finalize(self, szkm1: int, szk: int) -> None:
        pass
