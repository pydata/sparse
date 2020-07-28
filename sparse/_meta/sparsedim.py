from abc import abstractmethod, ABC
from functools import lru_cache, reduce
import numba as nb
from numba.extending import typeof_impl
from numba.core import types, extending
from collections import namedtuple
from typing import Sequence, List, Tuple, Iterable

__all__ = ["SparseDim"]


class SparseDim(ABC):
    properties: Sequence[str]

    def __init_subclass__(cls, **kwargs):
        if ABC in cls.__bases__:
            return

        @typeof_impl.register(cls)
        def typeof_cls(val: SparseDim, c):
            return nb.typeof(val.v)

    @classmethod
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
        return hash(
            (
                type(self),
                self.full,
                self.ordered,
                self.unique,
                self.branchless,
                self.compact,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, SparseDim):
            return False
        return (
            type(self),
            self.full,
            self.ordered,
            self.unique,
            self.branchless,
            self.compact,
        ) == (
            type(other),
            other.full,
            other.ordered,
            other.unique,
            other.branchless,
            other.compact,
        )

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return repr(self.v)


class SparseDimType(types.Type):
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


@extending.overload_attribute(SparseDimType, "full")
def impl_full(S):
    full = S.full

    def impl(S):
        return full

    return impl


@extending.overload_attribute(SparseDimType, "ordered")
def impl_ordered(S):
    ordered = S.ordered

    def impl(S):
        return ordered

    return impl


@extending.overload_attribute(SparseDimType, "unique")
def impl_unique(S):
    unique = S.unique

    def impl(S):
        return unique

    return impl


@extending.overload_attribute(SparseDimType, "branchless")
def impl_branchless(S):
    branchless = S.branchless

    def impl(S):
        return branchless

    return impl


@extending.overload_attribute(SparseDimType, "compact")
def impl_compact(S):
    compact = S.compact

    def impl(S):
        return compact

    return impl


class ValueIterable(SparseDim, ABC):
    @abstractmethod
    def coord_iter(self, i: Tuple[int, ...]) -> Iterable[int]:
        pass

    @abstractmethod
    def coord_access(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        pass

    def iterate(self, pkm1, i):
        for ik in self.coord_iter(i):
            pk, found = self.coord_access(pkm1, i + (ik,))
            if found:
                yield pk, ik


class ValueIterableType(SparseDimType):
    @property
    def support_value_iterable(self) -> "ValueIterable":
        return self


class PositionIterable(SparseDim, ABC):
    @abstractmethod
    def pos_iter(self, pkm1: int) -> Iterable[int]:
        pass

    @abstractmethod
    def pos_access(self, pk: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        pass

    def iterate(self, pkm1, i):
        for pk in self.pos_iter(pkm1):
            ik, found = self.pos_access(pk, i)
            if found:
                yield pk, ik


class PositionIterableType(SparseDimType):
    pass


class Locate(SparseDim, ABC):
    @abstractmethod
    def locate(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        pass


class LocateType(SparseDimType):
    pass


class InlineAssembly(SparseDim, ABC):
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


class InlineAssemblyType(SparseDimType):
    pass


class AppendAssembly(SparseDim, ABC):
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


class AppendAssemblyType(SparseDimType):
    @property
    def support_append(self) -> "ValueIterable":
        return self


# XXX: Move classes below to its own file
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

    def coord_iter(self, i: Tuple[int, ...]) -> Iterable[int]:
        return range(
            max(0, -self.offset[i[-1]]), min(self.N, self.M - self.offset[i[-1]])
        )

    def coord_access(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return pkm1 * self.N + i[-1], True


class Singleton(PositionIterable, AppendAssembly):
    properties: Sequence[str] = ("crd",)

    def __init__(
        self,
        *,
        crd: List[int],
        full: bool = True,
        ordered: bool = True,
        unique: bool = True
    ):
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

    def pos_bounds(self, pkm1: int) -> Iterable[int]:
        return range(pkm1, pkm1 + 1)

    def pos_access(self, pk: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return self.crd[pk], True

    def append_coord(self, pk: int, ik: int) -> None:
        self.crd.append(ik)

    def append_edges(self, pkm1: int, pkbegin: int, pkend: int) -> None:
        pass

    def append_init(self, szkm1: int, szk: int) -> None:
        pass

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

    def pos_bounds(self, pkm1: int) -> Iterable[int]:
        return range(pkm1, pkm1 + 1)

    def pos_access(self, pk: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return i[-1] + self.offset[i[-2]], True
