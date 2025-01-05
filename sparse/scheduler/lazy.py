import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .compiler import LogicCompiler
from .executor import LogicExecutor
from .finch_logic import (
    Aggregate,
    Alias,
    Field,
    Immediate,
    MapJoin,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    Subquery,
    Table,
)
from .optimize import DefaultLogicOptimizer
from .rewrite_tools import gensym


@dataclass
class LazyTensor:
    data: Any
    extrude: tuple
    fill_value: Any

    @property
    def ndim(self) -> int:
        return self.data.ndim


def lazy(arr) -> LazyTensor:
    name = Alias(gensym("A"))
    idxs = [Field(gensym("i")) for _ in range(arr.ndims)]
    extrude = tuple(arr.shape[i] == 1 for i in range(arr.ndims))
    tns = Subquery(name, Table(Immediate(arr), idxs))
    return LazyTensor(tns, extrude, arr.fill_value)


def get_at_idxs(arr, idxs):
    return [arr[i] for i in idxs]


def permute_dims(arg: LazyTensor, perm) -> LazyTensor:
    idxs = [Field(gensym("i")) for _ in range(arg.ndim)]
    return LazyTensor(
        Reorder(Relabel(arg.data, idxs), get_at_idxs(idxs, perm)),
        get_at_idxs(arg.extrude, perm),
        arg.fill_value,
    )


def identify(data):
    lhs = Alias(gensym("A"))
    return Subquery(lhs, data)


def reduce(op: Callable, arg: LazyTensor, dims=..., fill_value=0.0) -> LazyTensor:
    dims = list(range(arg.ndim) if dims is ... else dims)
    extrude = tuple(arg.extrude[n] for n in range(arg.ndim) if n not in dims)
    fields = [Field(gensym("i")) for _ in range(arg.ndim)]
    data = Aggregate(Immediate(op), Immediate(fill_value), Relabel(arg.data, fields), [fields[i] for i in dims])
    return LazyTensor(identify(data), extrude, fill_value)


def map(f: Callable, src: LazyTensor, *args) -> LazyTensor:
    largs = [src, *args]
    extrude = largs[next(filter(lambda x: len(x.extrude) > 0, largs), 0)].extrude
    idxs = [Field(gensym("i") for _ in src.extrude)]
    ldatas = []
    for larg in largs:
        if larg.extrude == extrude:
            ldatas.append(Relabel(larg.data, idxs))
        elif larg.extrude == ():
            ldatas.append(Relabel(larg.data))
        else:
            raise Exception("Cannot map across arrays with different sizes.")
    new_fill_value = f(*[x.fill_value for x in largs])
    data = MapJoin(Immediate(f), ldatas)
    return LazyTensor(identify(data), src.extrude, new_fill_value)


def prod(arr: LazyTensor, dims) -> LazyTensor:
    return reduce(operator.mul, arr, dims, arr.fill_value)


def multiply(x1: LazyTensor, x2: LazyTensor) -> LazyTensor:
    return map(operator.mul, x1, x2)


_ds = LogicExecutor(DefaultLogicOptimizer(LogicCompiler()))


def get_default_scheduler():
    return _ds


def compute(*args, ctx=_ds):
    vars = tuple(Alias("A") for _ in args)
    bodies = tuple(*map(lambda arg, var: Query(var, arg.data), args, vars))
    prgm = Plan(bodies + (Produces(vars),))
    return ctx(prgm)
