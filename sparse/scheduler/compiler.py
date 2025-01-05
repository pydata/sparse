from collections.abc import Hashable
from textwrap import dedent
from typing import Any

from .finch_logic import (
    Alias,
    Deferred,
    Field,
    Immediate,
    LogicNode,
    MapJoin,
    Query,
    Reformat,
    Relabel,
    Reorder,
    Subquery,
    Table,
)


def get_or_insert(dictionary: dict[Hashable, Any], key: Hashable, default: Any) -> Any:
    if key in dictionary:
        return dictionary[key]
    dictionary[key] = default
    return default


def get_structure(node: LogicNode, fields: dict[str, LogicNode], aliases: dict[str, LogicNode]) -> LogicNode:
    match node:
        case Field(name):
            return get_or_insert(fields, name, Immediate(len(fields) + len(aliases)))
        case Alias(name):
            return get_or_insert(aliases, name, Immediate(len(fields) + len(aliases)))
        case Subquery(Alias(name) as lhs, arg):
            if name in aliases:
                return aliases[name]
            return Subquery(get_structure(lhs, fields, aliases), get_structure(arg, fields, aliases))
        case Table(tns, idxs):
            return Table(Immediate(type(tns.val)), tuple(get_structure(idx, fields, aliases) for idx in idxs))
        case any if any.is_tree():
            return any.from_arguments(*[get_structure(arg, fields, aliases) for arg in any.get_arguments()])
        case _:
            return node


class PointwiseLowerer:
    def __init__(self):
        self.bound_idxs = []

    def __call__(self, ex):
        match ex:
            case MapJoin(Immediate(val), args):
                return f":({val}({','.join([self(arg) for arg in args])}))"
            case Reorder(Relabel(Alias(name), idxs_1), idxs_2):
                self.bound_idxs.append(idxs_1)
                return f":({name}[{','.join([idx.name if idx in idxs_2 else 1 for idx in idxs_1])}])"
            case Reorder(Immediate(val), _):
                return val
            case Immediate(val):
                return val
            case _:
                raise Exception(f"Unrecognized logic: {ex}")


def compile_pointwise_logic(ex: LogicNode) -> tuple:
    ctx = PointwiseLowerer()
    code = ctx(ex)
    return (code, ctx.bound_idxs)


def compile_logic_constant(ex: LogicNode) -> str:
    match ex:
        case Immediate(val):
            return val
        case Deferred(ex, type_):
            return f":({ex}::{type_})"
        case _:
            raise Exception(f"Invalid constant: {ex}")


def intersect(x1: tuple, x2: tuple) -> tuple:
    return tuple(x for x in x1 if x in x2)


def with_subsequence(x1: tuple, x2: tuple) -> tuple:
    res = list(x2)
    indices = [idx for idx, val in enumerate(x2) if val in x1]
    for idx, i in enumerate(indices):
        res[i] = x1[idx]
    return tuple(res)


class LogicLowerer:
    def __init__(self, mode: str = "fast"):
        self.mode = mode

    def __call__(self, ex: LogicNode):
        match ex:
            case Query(Alias(name), Table(tns, _)):
                return f":({name} = {compile_logic_constant(tns)})"

            case Query(Alias(_) as lhs, Reformat(tns, Reorder(Relabel(Alias(_) as arg, idxs_1), idxs_2))):
                loop_idxs = [idx.name for idx in with_subsequence(intersect(idxs_1, idxs_2), idxs_2)]
                lhs_idxs = [idx.name for idx in idxs_2]
                (rhs, rhs_idxs) = compile_pointwise_logic(Reorder(Relabel(arg, idxs_1), idxs_2))
                body = f":({lhs.name}[{','.join(lhs_idxs)}] = {rhs})"
                for idx in loop_idxs:
                    if Field(idx) in rhs_idxs:
                        body = f":(for {idx} = _ \n {body} end)"
                    elif idx in lhs_idxs:
                        body = f":(for {idx} = 1:1 \n {body} end)"

                result = f"""\
                    quote
                        {lhs.name} = {compile_logic_constant(tns)}
                        @finch mode = {self.mode} begin
                            {lhs.name} .= {tns.fill_value}
                            {body}
                            return {lhs.name}
                        end
                    end
                    """
                return dedent(result)

            # TODO: ...

            case _:
                raise Exception(f"Unrecognized logic: {ex}")


class LogicCompiler:
    def __init__(self):
        self.ll = LogicLowerer()

    def __call__(self, prgm):
        # prgm = format_queries(prgm, True)  # noqa: F821
        return self.ll(prgm)
