from abc import abstractmethod

from llvmlite import ir
from numba.core import types, cgutils, extending
from numba.core.datamodel import registry, models
from .sparsedim import PositionIterable, AppendAssembly
from .sparsedim import PositionIterableType, AppendAssemblyType
from typing import Sequence, List, Tuple, Iterable, Callable


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

    def pos_bounds(self, pkm1: int) -> Tuple[int, int]:
        return (self.pos[pkm1], self.pos[pkm1 + 1])

    def pos_iter(self, pkm1: int) -> Iterable[int]:
        return range(self.pos[pkm1], self.pos[pkm1 + 1])

    def pos_access(self, pk: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return self.crd[pk], True

    def append_coord(self, pk: int, ik: int) -> None:
        self.crd.append(ik)

    def append_edges(self, pkm1: int, pkbegin: int, pkend: int) -> None:
        self.pos[pkm1 + 1] = pkend - pkbegin

    def append_init(self, szkm1: int, szk: int) -> None:
        for _ in range(szkm1 + 1):
            self.pos.append(0)

    def append_finalize(self, szkm1: int, szk: int) -> None:
        cumsum: int = self.pos[0]
        for pkm1 in range(1, szkm1 + 1):
            cumsum += self.pos[pkm1]
            self.pos[pkm1] = cumsum

    def size(self) -> int:
        return len(self.crd)


class CompressedType(PositionIterableType, AppendAssemblyType):

    # Type is mutable
    mutable = True

    def __init__(
        self,
        *,
        pos_type: types.Integer,
        crd_type: types.Integer,
        full: bool,
        ordered: bool,
        unique: bool,
    ):
        if not isinstance(pos_type, types.Integer):
            raise TypeError("pos_type must be a numba.types.Integer.")

        if not isinstance(crd_type, types.Integer):
            raise TypeError("crd_type must be a numba.types.Integer.")

        self._pos_type: types.Integer = pos_type
        self._crd_type: types.Integer = crd_type
        self._full: bool = bool(full)
        self._ordered: bool = bool(ordered)
        self._unique: bool = bool(unique)
        name: str = f"Compressed<{pos_type}, {crd_type}>"
        super().__init__(name)

    @property
    def key(self):
        return (self._full, self._ordered, self._unique, self._pos_type, self._crd_type)

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
    def pos_type(self):
        return self._pos_type

    @property
    def crd_type(self):
        return self._crd_type

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
            ("pos", types.ListType(fe_type.pos_type)),
            ("crd", types.ListType(fe_type.crd_type)),
        ]
        super().__init__(dmm, fe_type, members)


@extending.type_callable(Compressed)
def type_compressed(context):
    def typer(full, ordered, unique, pos, crd):
        # pos and crd are TypedLists
        return CompressedType(
            pos_type=pos.dtype,
            crd_type=crd.dtype,
            full=full,
            ordered=ordered,
            unique=unique,
        )

    return typer


@extending.lower_builtin(
    Compressed, types.Boolean, types.Boolean, types.Boolean, types.Any, types.Any
)
def sparse_compressed_constructor(context, builder, sig, args):
    _, _, _, pos, crd = args
    compressed = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    compressed.pos = pos
    compressed.crd = crd
    context.nrt.incref(builder, sig.args[3], pos)
    context.nrt.incref(builder, sig.args[4], crd)
    return compressed._getvalue()


extending.make_attribute_wrapper(CompressedType, "pos", "pos")
extending.make_attribute_wrapper(CompressedType, "crd", "crd")


@extending.overload_method(CompressedType, "pos_bounds")
def impl_pos_bounds(self, pkm1: int) -> Tuple[int, int]:
    return Compressed.pos_bounds


@extending.overload_method(CompressedType, "pos_iter")
def impl_pos_iter(self, pkm1: int) -> Tuple[int, int]:
    return Compressed.pos_iter


@extending.overload_method(CompressedType, "pos_access")
def impl_pos_access(self, pk: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
    return Compressed.pos_access


@extending.overload_method(CompressedType, "append_coord")
def impl_append_coord(self, pk: int, ik: int) -> Callable:
    return Compressed.append_coord


@extending.overload_method(CompressedType, "append_edges")
def impl_append_edges(self, pkm1: int, pkbegin: int, pkend: int) -> Callable:
    return Compressed.append_edges


@extending.overload_method(CompressedType, "append_init")
def impl_append_init(self, szkm1: int, szk: int) -> Callable:
    return Compressed.append_init


@extending.overload_method(CompressedType, "append_finalize")
def impl_append_finalize(self, szkm1: int, szk: int) -> Callable:
    return Compressed.append_finalize


@extending.typeof_impl.register(Compressed)
def typeof_index(val, c):
    N_type = types.int64
    full = val.full
    ordered = val.ordered
    unique = val.unique
    # pos and crd should always be TypedLists?
    pos_type = val.pos._dtype
    crd_type = val.crd._dtype
    return CompressedType(
        full=full, ordered=ordered, unique=unique, pos_type=pos_type, crd_type=crd_type
    )


@extending.box(CompressedType)
def box_dense(typ: CompressedType, val, c):
    """
    Convert a native compressed structure to a compressed object.
    """
    i1 = ir.IntType(1)
    context, builder = c.context, c.builder

    ctor = cgutils.create_struct_proxy(typ)
    lstruct = ctor(context, builder, value=val)

    boxed_crd = c.box(types.ListType(typ.pos_type), lstruct.crd)
    boxed_pos = c.box(types.ListType(typ.pos_type), lstruct.pos)

    full_obj = c.pyapi.bool_from_bool(i1(typ.full))
    ordered_obj = c.pyapi.bool_from_bool(i1(typ.ordered))
    unique_obj = c.pyapi.bool_from_bool(i1(typ.unique))

    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Compressed))

    kwds = c.pyapi.dict_pack(
        {
            "pos": boxed_pos,
            "crd": boxed_crd,
            "full": full_obj,
            "ordered": ordered_obj,
            "unique": unique_obj,
        }.items()
    )
    empty_tuple = c.pyapi.tuple_new(0)

    res = c.pyapi.call(class_obj, empty_tuple, kwds)
    c.pyapi.decref(boxed_crd)
    c.pyapi.decref(boxed_pos)
    c.pyapi.decref(class_obj)
    c.pyapi.decref(full_obj)
    c.pyapi.decref(ordered_obj)
    c.pyapi.decref(unique_obj)
    c.pyapi.decref(empty_tuple)
    c.pyapi.decref(kwds)
    return res


@extending.unbox(CompressedType)
def unbox_dense(typ: CompressedType, obj, c):
    """
    Convert a Compressed object to a native compressed structure.
    """
    context, builder = c.context, c.builder

    pos_obj = c.pyapi.object_getattr_string(obj, "pos")  # i8*
    crd_obj = c.pyapi.object_getattr_string(obj, "crd")  # i8*
    compressed = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # Assuming pos and crd are TypedLists
    pos_type = compressed.pos.type  # {i8*, i8*}
    crd_type = compressed.crd.type  # {i8*, i8*}

    compressed.pos = c.unbox(types.ListType(typ.pos_type), pos_obj).value
    compressed.crd = c.unbox(types.ListType(typ.crd_type), crd_obj).value

    c.pyapi.decref(pos_obj)
    c.pyapi.decref(crd_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return extending.NativeValue(compressed._getvalue(), is_error=is_error)
