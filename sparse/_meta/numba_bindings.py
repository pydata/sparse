from abc import abstractmethod

from llvmlite import ir
from numba import njit
from numba.core import types, cgutils, extending
from numba.core.datamodel import registry, models
from .sparsedim import Dense, Compressed


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


class Locate(SparseDimType):
    pass


class ValueIterable(SparseDimType):
    @property
    def support_value_iterable(self) -> "ValueIterable":
        return self


class InlineAssembly(SparseDimType):
    pass


class PositionIterable(SparseDimType):
    pass


class AppendAssembly(SparseDimType):
    @property
    def support_append(self) -> "ValueIterable":
        return self


class DenseType(Locate, ValueIterable, InlineAssembly):
    def __init__(
        self, *, N_type: types.Integer, ordered: bool = True, unique: bool = True
    ):
        if not isinstance(N_type, types.Integer):
            raise TypeError("N_type must be a numba.types.Integer.")

        self.N_type: int = N_type
        self._ordered: bool = bool(ordered)
        self._unique: bool = bool(unique)
        name = f"Dense<{N_type}>"
        super().__init__(name)

    # Type is mutable
    mutable = True
    # Wether the type is reflected at the python <-> nopython modes
    reflected = True

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


class CompressedType(SparseDimType):

    # Type is mutable
    mutable = True
    # Wether the type is reflected at the python <-> nopython modes
    reflected = True

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


@registry.register_default(DenseType)
class DenseModel(models.StructModel):
    def __init__(self, dmm, fe_type: DenseType):
        members = [
            ("N", fe_type.N_type),
        ]
        super().__init__(dmm, fe_type, members)


@registry.register_default(CompressedType)
class CompressedModel(models.StructModel):
    def __init__(self, dmm, fe_type: CompressedType):
        members = [
            ("pos", types.ListType[fe_type.pos_type]),
            ("crd", types.ListType[fe_type.crd_type]),
        ]
        super().__init__(dmm, fe_type, members)


@extending.type_callable(Dense)
def type_interval(context):
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
    kwds = c.pyapi.dict_pack({'N': N_obj, 'unique': unique_obj, 'ordered': ordered_obj}.items())
    res = c.pyapi.call(class_obj, None, kwds)
    c.pyapi.decref(N_obj)
    c.pyapi.decref(class_obj)
    c.pyapi.decref(unique_obj)
    c.pyapi.decref(ordered_obj)
    c.pyapi.decref(kwds)
    return res
