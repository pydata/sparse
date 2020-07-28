from numba.core import types, cgutils, extending
from numba.core.datamodel import registry, models
from llvmlite import ir
from .sparsedim import Locate, ValueIterable, InlineAssembly
from .sparsedim import LocateType, ValueIterableType, InlineAssemblyType
from typing import Sequence, Tuple, Iterable


class Dense(Locate, ValueIterable, InlineAssembly):
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

    def locate(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return pkm1 * self.N + i[-1], True

    def coord_bounds(self, i: Tuple[int, ...]) -> Tuple[int, int]:
        return (0, self.N)

    def coord_iter(self, i: Tuple[int, ...]) -> Iterable[int]:
        return range(self.N)

    def coord_access(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
        return pkm1 * self.N + i[-1], True

    def size(self, szkm1: int) -> int:
        return szkm1 * self.N

    def insert_coord(self, pk: int, ik: int) -> None:
        pass

    def insert_init(self, szkm1: int, szk: int) -> None:
        pass

    def insert_finalize(self, szkm1: int, szk: int) -> None:
        pass


class DenseType(LocateType, ValueIterableType, InlineAssemblyType):
    def __init__(
        self, *, N_type: types.Integer, ordered: bool = True, unique: bool = True
    ):
        if not isinstance(N_type, types.Integer):
            raise TypeError("N_type must be a numba.types.Integer.")

        self.N_type: types.Integer = N_type
        self._ordered: bool = bool(ordered)
        self._unique: bool = bool(unique)
        name = f"Dense<{N_type, ordered, unique}>"
        super().__init__(name)

    # Type is mutable
    mutable = True

    @property
    def key(self):
        return (self.N_type, self._ordered, self._unique)

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


@registry.register_default(DenseType)
class DenseModel(models.StructModel):
    def __init__(self, dmm, fe_type: DenseType):
        members = [
            ("N", fe_type.N_type),
        ]
        super().__init__(dmm, fe_type, members)


@extending.type_callable(Dense)
def type_dense(context):
    def typer(N, ordered, unique):
        return DenseType(N_type=N, ordered=ordered, unique=unique)

    return typer


@extending.lower_builtin(Dense, types.Any, types.Any, types.Any)
def sparse_dense_constructor(context, builder, sig, args):
    N, _, _ = args
    dense = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    dense.N = N
    return dense._getvalue()


extending.make_attribute_wrapper(DenseType, "N", "N")


@extending.overload_method(DenseType, "locate")
def impl_dense_locate(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
    return Dense.locate


@extending.overload_method(DenseType, "coord_bounds")
def impl_dense_coord_bounds(self, i: Tuple[int, ...]) -> Tuple[int, int]:
    return Dense.coord_bounds


@extending.overload_method(DenseType, "coord_iter")
def impl_dense_coord_bounds(self, i: Tuple[int, ...]) -> Tuple[int, int]:
    return Dense.coord_iter


@extending.overload_method(DenseType, "coord_access")
def impl_dense_coord_access(self, pkm1: int, i: Tuple[int, ...]) -> Tuple[int, bool]:
    return Dense.coord_access


@extending.overload_method(DenseType, "size")
def impl_dense_size(self, szkm1: int) -> int:
    return Dense.size


@extending.overload_method(DenseType, "insert_coord")
def impl_dense_insert_coord(self, pk: int, ik: int) -> None:
    return Dense.insert_coord


@extending.overload_method(DenseType, "insert_init")
def impl_dense_insert_init(self, szkm1: int, szk: int) -> None:
    return Dense.insert_init


@extending.overload_method(DenseType, "insert_finalize")
def impl_dense_insert_finalize(self, szkm1: int, szk: int) -> None:
    return Dense.insert_finalize


@extending.typeof_impl.register(Dense)
def typeof_index(val, c):
    N_type = types.int64
    ordered = val.ordered
    unique = val.unique
    return DenseType(N_type=N_type, ordered=ordered, unique=unique)


@extending.box(DenseType)
def box_dense(typ: DenseType, val, c):
    """
    Convert a native dense structure to a Dense object.
    """
    i1 = ir.IntType(1)

    dense = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

    N_obj = c.pyapi.long_from_long(dense.N)
    unique_obj = c.pyapi.bool_from_bool(i1(typ.unique))
    ordered_obj = c.pyapi.bool_from_bool(i1(typ.ordered))
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Dense))

    kwds = c.pyapi.dict_pack(
        {"N": N_obj, "unique": unique_obj, "ordered": ordered_obj}.items()
    )
    empty_tuple = c.pyapi.tuple_new(0)

    res = c.pyapi.call(class_obj, empty_tuple, kwds)

    c.pyapi.decref(N_obj)
    c.pyapi.decref(class_obj)
    c.pyapi.decref(unique_obj)
    c.pyapi.decref(ordered_obj)
    c.pyapi.decref(empty_tuple)
    c.pyapi.decref(kwds)
    return res


@extending.unbox(DenseType)
def unbox_dense(typ, obj, c):
    """
    Convert a Dense object to a native dense structure.
    """
    N_obj = c.pyapi.object_getattr_string(obj, "N")
    dense = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    dense.N = c.pyapi.long_as_longlong(N_obj)
    c.pyapi.decref(N_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return extending.NativeValue(dense._getvalue(), is_error=is_error)
