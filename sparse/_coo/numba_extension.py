"""
Numba support for COO objects.

For now, this just supports attribute access
"""
import numpy as np
import numba
from numba.extending import (
    models,
    register_model,
    box,
    unbox,
    NativeValue,
    make_attribute_wrapper,
    type_callable,
    lower_builtin,
)
from numba.targets.imputils import impl_ret_borrowed, lower_constant
from numba.typing.typeof import typeof_impl
from numba import cgutils, types
from sparse._utils import _zero_of_dtype

from . import COO

__all__ = ["COOType"]


class COOType(types.Type):
    def __init__(self, data_dtype: np.dtype, coords_dtype: np.dtype, ndim: int):
        assert isinstance(data_dtype, np.dtype)
        assert isinstance(coords_dtype, np.dtype)
        self.data_dtype = data_dtype
        self.coords_dtype = coords_dtype
        self.ndim = ndim
        super().__init__(
            name="COOType[{!r}, {!r}, {!r}]".format(
                numba.from_dtype(data_dtype), numba.from_dtype(coords_dtype), ndim
            )
        )

    @property
    def key(self):
        return self.data_dtype, self.coords_dtype, self.ndim

    @property
    def data_type(self):
        return numba.from_dtype(self.data_dtype)[:]

    @property
    def coords_type(self):
        return numba.from_dtype(self.coords_dtype)[:, :]

    @property
    def shape_type(self):
        return types.UniTuple(types.int64, self.ndim)

    @property
    def fill_value_type(self):
        return numba.from_dtype(self.data_dtype)


@typeof_impl.register(COO)
def _typeof_COO(val: COO, c) -> COOType:
    return COOType(
        data_dtype=val.data.dtype, coords_dtype=val.coords.dtype, ndim=val.ndim
    )


@register_model(COOType)
class COOModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data_type),
            ("coords", fe_type.coords_type),
            ("shape", fe_type.shape_type),
            ("fill_value", fe_type.fill_value_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@type_callable(COO)
def type_COO(context):
    # TODO: accept a fill_value kwarg
    def typer(coords, data, shape):
        return COOType(
            coords_dtype=numba.numpy_support.as_dtype(coords.dtype),
            data_dtype=numba.numpy_support.as_dtype(data.dtype),
            ndim=len(shape),
        )

    return typer


@lower_builtin(COO, types.Any, types.Any, types.Any)
def impl_COO(context, builder, sig, args):
    typ = sig.return_type
    coords, data, shape = args
    coo = cgutils.create_struct_proxy(typ)(context, builder)
    coo.coords = coords
    coo.data = data
    coo.shape = shape
    coo.fill_value = context.get_constant_generic(
        builder, typ.fill_value_type, _zero_of_dtype(typ.data_dtype)
    )
    return impl_ret_borrowed(context, builder, sig.return_type, coo._getvalue())


@lower_constant(COOType)
def lower_constant_COO(context, builder, typ, pyval):
    coords = context.get_constant_generic(builder, typ.coords_type, pyval.coords)
    data = context.get_constant_generic(builder, typ.data_type, pyval.data)
    shape = context.get_constant_generic(builder, typ.shape_type, pyval.shape)
    fill_value = context.get_constant_generic(
        builder, typ.fill_value_type, pyval.fill_value
    )
    return impl_ret_borrowed(
        context,
        builder,
        typ,
        cgutils.pack_struct(builder, (data, coords, shape, fill_value)),
    )


@unbox(COOType)
def unbox_COO(typ: COOType, obj: COO, c) -> NativeValue:
    data = c.pyapi.object_getattr_string(obj, "data")
    coords = c.pyapi.object_getattr_string(obj, "coords")
    shape = c.pyapi.object_getattr_string(obj, "shape")
    fill_value = c.pyapi.object_getattr_string(obj, "fill_value")
    coo = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    coo.coords = c.unbox(typ.coords_type, coords).value
    coo.data = c.unbox(typ.data_type, data).value
    coo.shape = c.unbox(typ.shape_type, shape).value
    coo.fill_value = c.unbox(typ.fill_value_type, fill_value).value
    c.pyapi.decref(data)
    c.pyapi.decref(coords)
    c.pyapi.decref(shape)
    c.pyapi.decref(fill_value)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(coo._getvalue(), is_error=is_error)


@box(COOType)
def box_COO(typ: COOType, val: NativeValue, c) -> COO:
    coo = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    data_obj = c.box(typ.data_type, coo.data)
    coords_obj = c.box(typ.coords_type, coo.coords)
    shape_obj = c.box(typ.shape_type, coo.shape)
    fill_value_obj = c.box(typ.fill_value_type, coo.fill_value)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(COO))
    args = c.pyapi.tuple_pack([coords_obj, data_obj, shape_obj])
    kwargs = c.pyapi.dict_pack([("fill_value", fill_value_obj)])
    res = c.pyapi.call(class_obj, args, kwargs)
    c.pyapi.decref(class_obj)
    c.pyapi.decref(data_obj)
    c.pyapi.decref(coords_obj)
    c.pyapi.decref(shape_obj)
    c.pyapi.decref(fill_value_obj)
    return res


make_attribute_wrapper(COOType, "data", "data")
make_attribute_wrapper(COOType, "coords", "coords")
make_attribute_wrapper(COOType, "shape", "shape")
make_attribute_wrapper(COOType, "fill_value", "fill_value")
