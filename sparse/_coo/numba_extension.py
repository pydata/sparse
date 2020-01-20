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
from contextlib import ExitStack, contextmanager

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


@contextmanager
def early_exit_if(builder, stack: ExitStack, cond):
    """
    Emit code similar to::

        if (cond) {
            <body>
            return;
        }
        <everything after this call>

    However, this "return" will just break out of the current `ExitStack`,
    rather than out of the whole function
    """
    then, otherwise = stack.enter_context(builder.if_else(cond, likely=False))
    with then:
        yield
    stack.enter_context(otherwise)


def early_exit_if_null(builder, stack, obj):
    return early_exit_if(builder, stack, cgutils.is_null(builder, obj))


def _unbox_native_field(typ, obj, field_name: str, c):
    ret_ptr = cgutils.alloca_once(c.builder, c.context.get_value_type(typ))
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    fail_obj = c.context.get_constant_null(typ)

    with ExitStack() as stack:
        field_obj = c.pyapi.object_getattr_string(obj, field_name)
        with early_exit_if_null(c.builder, stack, field_obj):
            c.builder.store(cgutils.true_bit, is_error_ptr)
            c.builder.store(fail_obj, ret_ptr)

        field_native = c.unbox(typ, field_obj)
        c.pyapi.decref(field_obj)
        with early_exit_if(c.builder, stack, field_native.is_error):
            c.builder.store(cgutils.true_bit, is_error_ptr)
            c.builder.store(fail_obj, ret_ptr)

        c.builder.store(cgutils.false_bit, is_error_ptr)
        c.builder.store(field_native.value, ret_ptr)

    return NativeValue(c.builder.load(ret_ptr), is_error=c.builder.load(is_error_ptr))


@unbox(COOType)
def unbox_COO(typ: COOType, obj: COO, c) -> NativeValue:
    ret_ptr = cgutils.alloca_once(c.builder, c.context.get_value_type(typ))
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    fail_obj = c.context.get_constant_null(typ)

    with ExitStack() as stack:
        data = _unbox_native_field(typ.data_type, obj, "data", c)
        with early_exit_if(c.builder, stack, data.is_error):
            c.builder.store(cgutils.true_bit, is_error_ptr)
            c.builder.store(fail_obj, ret_ptr)

        coords = _unbox_native_field(typ.coords_type, obj, "coords", c)
        with early_exit_if(c.builder, stack, coords.is_error):
            c.builder.store(cgutils.true_bit, is_error_ptr)
            c.builder.store(fail_obj, ret_ptr)

        shape = _unbox_native_field(typ.shape_type, obj, "shape", c)
        with early_exit_if(c.builder, stack, shape.is_error):
            c.builder.store(cgutils.true_bit, is_error_ptr)
            c.builder.store(fail_obj, ret_ptr)

        fill_value = _unbox_native_field(typ.fill_value_type, obj, "fill_value", c)
        with early_exit_if(c.builder, stack, fill_value.is_error):
            c.builder.store(cgutils.true_bit, is_error_ptr)
            c.builder.store(fail_obj, ret_ptr)

        coo = cgutils.create_struct_proxy(typ)(c.context, c.builder)
        coo.coords = coords.value
        coo.data = data.value
        coo.shape = shape.value
        coo.fill_value = fill_value.value
        c.builder.store(cgutils.false_bit, is_error_ptr)
        c.builder.store(coo._getvalue(), ret_ptr)
    return NativeValue(c.builder.load(ret_ptr), is_error=c.builder.load(is_error_ptr))


@box(COOType)
def box_COO(typ: COOType, val: "some LLVM thing", c) -> COO:
    ret_ptr = cgutils.alloca_once(c.builder, c.pyapi.pyobj)
    fail_obj = c.pyapi.get_null_object()

    with ExitStack() as stack:
        coo = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

        data_obj = c.box(typ.data_type, coo.data)
        with early_exit_if_null(c.builder, stack, data_obj):
            c.builder.store(fail_obj, ret_ptr)

        coords_obj = c.box(typ.coords_type, coo.coords)
        with early_exit_if_null(c.builder, stack, coords_obj):
            c.pyapi.decref(data_obj)
            c.builder.store(fail_obj, ret_ptr)

        shape_obj = c.box(typ.shape_type, coo.shape)
        with early_exit_if_null(c.builder, stack, shape_obj):
            c.pyapi.decref(coords_obj)
            c.pyapi.decref(data_obj)
            c.builder.store(fail_obj, ret_ptr)

        fill_value_obj = c.box(typ.fill_value_type, coo.fill_value)
        with early_exit_if_null(c.builder, stack, fill_value_obj):
            c.pyapi.decref(shape_obj)
            c.pyapi.decref(coords_obj)
            c.pyapi.decref(data_obj)
            c.builder.store(fail_obj, ret_ptr)

        class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(COO))
        with early_exit_if_null(c.builder, stack, class_obj):
            c.pyapi.decref(shape_obj)
            c.pyapi.decref(coords_obj)
            c.pyapi.decref(data_obj)
            c.pyapi.decref(fill_value_obj)
            c.builder.store(fail_obj, ret_ptr)

        args = c.pyapi.tuple_pack([coords_obj, data_obj, shape_obj])
        c.pyapi.decref(shape_obj)
        c.pyapi.decref(coords_obj)
        c.pyapi.decref(data_obj)
        with early_exit_if_null(c.builder, stack, args):
            c.pyapi.decref(fill_value_obj)
            c.pyapi.decref(class_obj)
            c.builder.store(fail_obj, ret_ptr)

        kwargs = c.pyapi.dict_pack([("fill_value", fill_value_obj)])
        c.pyapi.decref(fill_value_obj)
        with early_exit_if_null(c.builder, stack, kwargs):
            c.pyapi.decref(class_obj)
            c.builder.store(fail_obj, ret_ptr)
        c.builder.store(c.pyapi.call(class_obj, args, kwargs), ret_ptr)

    return c.builder.load(ret_ptr)


make_attribute_wrapper(COOType, "data", "data")
make_attribute_wrapper(COOType, "coords", "coords")
make_attribute_wrapper(COOType, "shape", "shape")
make_attribute_wrapper(COOType, "fill_value", "fill_value")
