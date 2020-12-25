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
)
from numba.core.imputils import impl_ret_borrowed, lower_constant, lower_builtin
from numba.core.typing.typeof import typeof_impl
from numba.core import cgutils, types
from sparse._utils import _zero_of_dtype
import contextlib

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
        dt = numba.np.numpy_support.from_dtype(self.coords_dtype)
        return types.UniTuple(dt, self.ndim)

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
            coords_dtype=numba.np.numpy_support.as_dtype(coords.dtype),
            data_dtype=numba.np.numpy_support.as_dtype(data.dtype),
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


@contextlib.contextmanager
def local_return(builder):
    """
    Create a scope which can be broken from locally.

    Used as::

        with local_return(c.builder) as ret:
            with c.builder.if(abort_cond):
                ret()
            do_some_other_stuff
            # no ret needed at the end, it's implied

        stuff_that_runs_unconditionally
    """
    end_blk = builder.append_basic_block("end")

    def return_():
        builder.branch(end_blk)

    yield return_
    builder.branch(end_blk)
    # make sure all remaining code goes to the next block
    builder.position_at_end(end_blk)


def _unbox_native_field(typ, obj, field_name: str, c):
    ret_ptr = cgutils.alloca_once(c.builder, c.context.get_value_type(typ))
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    fail_obj = c.context.get_constant_null(typ)

    with local_return(c.builder) as ret:
        fail_blk = c.builder.append_basic_block("fail")
        with c.builder.goto_block(fail_blk):
            c.builder.store(cgutils.true_bit, is_error_ptr)
            c.builder.store(fail_obj, ret_ptr)
            ret()

        field_obj = c.pyapi.object_getattr_string(obj, field_name)
        with cgutils.if_unlikely(c.builder, cgutils.is_null(c.builder, field_obj)):
            c.builder.branch(fail_blk)

        field_native = c.unbox(typ, field_obj)
        c.pyapi.decref(field_obj)
        with cgutils.if_unlikely(c.builder, field_native.is_error):
            c.builder.branch(fail_blk)

        c.builder.store(cgutils.false_bit, is_error_ptr)
        c.builder.store(field_native.value, ret_ptr)

    return NativeValue(c.builder.load(ret_ptr), is_error=c.builder.load(is_error_ptr))


@unbox(COOType)
def unbox_COO(typ: COOType, obj: COO, c) -> NativeValue:
    ret_ptr = cgutils.alloca_once(c.builder, c.context.get_value_type(typ))
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    fail_obj = c.context.get_constant_null(typ)

    with local_return(c.builder) as ret:
        fail_blk = c.builder.append_basic_block("fail")
        with c.builder.goto_block(fail_blk):
            c.builder.store(cgutils.true_bit, is_error_ptr)
            c.builder.store(fail_obj, ret_ptr)
            ret()

        data = _unbox_native_field(typ.data_type, obj, "data", c)
        with cgutils.if_unlikely(c.builder, data.is_error):
            c.builder.branch(fail_blk)

        coords = _unbox_native_field(typ.coords_type, obj, "coords", c)
        with cgutils.if_unlikely(c.builder, coords.is_error):
            c.builder.branch(fail_blk)

        shape = _unbox_native_field(typ.shape_type, obj, "shape", c)
        with cgutils.if_unlikely(c.builder, shape.is_error):
            c.builder.branch(fail_blk)

        fill_value = _unbox_native_field(typ.fill_value_type, obj, "fill_value", c)
        with cgutils.if_unlikely(c.builder, fill_value.is_error):
            c.builder.branch(fail_blk)

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

    coo = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

    with local_return(c.builder) as ret:
        data_obj = c.box(typ.data_type, coo.data)
        with cgutils.if_unlikely(c.builder, cgutils.is_null(c.builder, data_obj)):
            c.builder.store(fail_obj, ret_ptr)
            ret()

        coords_obj = c.box(typ.coords_type, coo.coords)
        with cgutils.if_unlikely(c.builder, cgutils.is_null(c.builder, coords_obj)):
            c.pyapi.decref(data_obj)
            c.builder.store(fail_obj, ret_ptr)
            ret()

        shape_obj = c.box(typ.shape_type, coo.shape)
        with cgutils.if_unlikely(c.builder, cgutils.is_null(c.builder, shape_obj)):
            c.pyapi.decref(coords_obj)
            c.pyapi.decref(data_obj)
            c.builder.store(fail_obj, ret_ptr)
            ret()

        fill_value_obj = c.box(typ.fill_value_type, coo.fill_value)
        with cgutils.if_unlikely(c.builder, cgutils.is_null(c.builder, fill_value_obj)):
            c.pyapi.decref(shape_obj)
            c.pyapi.decref(coords_obj)
            c.pyapi.decref(data_obj)
            c.builder.store(fail_obj, ret_ptr)
            ret()

        class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(COO))
        with cgutils.if_unlikely(c.builder, cgutils.is_null(c.builder, class_obj)):
            c.pyapi.decref(shape_obj)
            c.pyapi.decref(coords_obj)
            c.pyapi.decref(data_obj)
            c.pyapi.decref(fill_value_obj)
            c.builder.store(fail_obj, ret_ptr)
            ret()

        args = c.pyapi.tuple_pack([coords_obj, data_obj, shape_obj])
        c.pyapi.decref(shape_obj)
        c.pyapi.decref(coords_obj)
        c.pyapi.decref(data_obj)
        with cgutils.if_unlikely(c.builder, cgutils.is_null(c.builder, args)):
            c.pyapi.decref(fill_value_obj)
            c.pyapi.decref(class_obj)
            c.builder.store(fail_obj, ret_ptr)
            ret()

        kwargs = c.pyapi.dict_pack([("fill_value", fill_value_obj)])
        c.pyapi.decref(fill_value_obj)
        with cgutils.if_unlikely(c.builder, cgutils.is_null(c.builder, kwargs)):
            c.pyapi.decref(class_obj)
            c.builder.store(fail_obj, ret_ptr)
            ret()

        c.builder.store(c.pyapi.call(class_obj, args, kwargs), ret_ptr)
        c.pyapi.decref(class_obj)
        c.pyapi.decref(args)
        c.pyapi.decref(kwargs)

    return c.builder.load(ret_ptr)


make_attribute_wrapper(COOType, "data", "data")
make_attribute_wrapper(COOType, "coords", "coords")
make_attribute_wrapper(COOType, "shape", "shape")
make_attribute_wrapper(COOType, "fill_value", "fill_value")
