from numba.core import types, extending
from numba.core.datamodel import registry, models
from .sparsedim import Locate, PositionIterable, InlineAssembly
from .sparsedim import LocateType, PositionIterableType, InlineAssemblyType
from typing import Sequence, Tuple, List, Dict, Iterable, Callable


class Hashed(Locate, PositionIterable, InlineAssembly):
    properties: Sequence[str] = ("W", "crd")

    def __init__(
        self,
        *,
        W: int,
        crd: List[Dict[int, int]],
        full: bool = True,
        unique: bool = True,
    ):
        self.W: int = W
        self.crd: List[Dict[int, int]] = crd
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

    def locate(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        pk = pkm1 * self.W + i[-1]
        d = self.crd[pkm1]
        return pk, pk in d

    def pos_iter(self, pkm1: int) -> Iterable[int]:
        return self.crd[pkm1].keys()

    def pos_access(self, pk: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        d = self.crd[pkm1]
        ik = d.get(pk, -1)
        return ik, ik != -1

    def size(self, szkm1: int) -> int:
        return szkm1 * self.W

    def insert_coord(self, pk: int, ik: int) -> None:
        pkm1 = pk // self.W
        self.crd[pkm1][pk] = ik

    def insert_init(self, szkm1: int, szk: int) -> None:
        for _ in range(szkm1):
            self.crd.append({})

    def insert_finalize(self, szkm1: int, szk: int) -> None:
        pass


class HashedType(LocateType, PositionIterableType, InlineAssemblyType):
    def __init__(
        self,
        *,
        W_type: types.Integer,
        crd_key_type: types.Integer,
        crd_value_type: types.Integer,
        full: bool = True,
        unique: bool = True,
    ):
        if not isinstance(W_type, types.Integer):
            raise TypeError("W_type must be a numba.types.Integer.")

        self.W_type: types.Integer = W_type
        self.crd_key_type: types.Integer = crd_key_type
        self.crd_value_type: types.Integer = crd_value_type
        self._full: bool = bool(full)
        self._unique: bool = bool(unique)
        name = f"Hashed<{W_type, crd_key_type, crd_value_type, full, unique}>"
        super().__init__(name)

    # Type is mutable
    mutable = True

    @property
    def key(self):
        return (self.W_type, self._full, self._unique)

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


@registry.register_default(HashedType)
class HashedModel(models.StructModel):
    def __init__(self, dmm, fe_type: HashedType):
        members = [
            ("W", fe_type.W_type),
            ("crd", types.DictType(fe_type.crd_key_type, fe_type.crd_value_type)),
        ]
        super().__init__(dmm, fe_type, members)


@extending.type_callable(Hashed)
def type_hashed(context):
    def typer(W, crd, full, unique):
        # return HashedType here
        key = types.int64
        value = types.int64
        return HashedType(W_type=W, crd_key_type=key, crd_value_type=value, full=full, unique=unique)

    return typer


# @extending.lower_builtin(Hashed, types.Any, types.Any, types.Any)
# def sparse_hashed_constructor(context, builder, sig, args):
#     raise NotImplementedError


# extending.make_attribute_wrapper(HashedType, "N", "N")


@extending.overload_method(HashedType, "locate")
def impl_hashed_locate(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
    return Hashed.locate


@extending.overload_method(HashedType, "coord_bounds")
def impl_hashed_coord_bounds(self, i: Tuple[int, ...]) -> Tuple[int, int]:
    return Hashed.coord_bounds


@extending.overload_method(HashedType, "coord_access")
def impl_hashed_coord_access(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
    return Hashed.coord_access


@extending.overload_method(HashedType, "size")
def impl_hashed_size(self, szkm1: int) -> int:
    return Hashed.size


@extending.overload_method(HashedType, "insert_coord")
def impl_hashed_insert_coord(self, pk: int, ik: int) -> Callable:
    return Hashed.insert_coord


@extending.overload_method(HashedType, "insert_init")
def impl_hashed_insert_init(self, szkm1: int, szk: int) -> Callable:
    return Hashed.insert_init


@extending.overload_method(HashedType, "insert_finalize")
def impl_hashed_insert_finalize(self, szkm1: int, szk: int) -> Callable:
    return Hashed.insert_finalize


@extending.typeof_impl.register(Hashed)
def typeof_index(val, c):
    raise NotImplementedError


@extending.box(HashedType)
def box_dense(typ: HashedType, val, c):
    raise NotImplementedError


@extending.unbox(HashedType)
def unbox_hashed(typ, obj, c):
    raise NotImplementedError
