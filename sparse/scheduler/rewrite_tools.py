from collections.abc import Callable, Iterable, Iterator

from .finch_logic import LogicNode

RwCallable = Callable[[LogicNode], LogicNode | None]


class SymbolGenerator:
    counter: int = 0

    @classmethod
    def gensym(cls, name: str) -> str:
        sym = f"#{name}#{cls.counter}"
        cls.counter += 1
        return sym


_sg = SymbolGenerator()
gensym: Callable[[str], str] = _sg.gensym


def get_or_else(x: LogicNode | None, y: LogicNode) -> LogicNode:
    return x if x is not None else y


def PostOrderDFS(node: LogicNode) -> Iterator[LogicNode]:
    if node.is_tree():
        for arg in node.get_arguments():
            yield from PostOrderDFS(arg)
    yield node


class Rewrite:
    def __init__(self, rw: RwCallable):
        self.rw = rw

    def __call__(self, x: LogicNode) -> LogicNode:
        return get_or_else(self.rw(x), x)


class PreWalk:
    def __init__(self, rw: RwCallable):
        self.rw = rw

    def __call__(self, x: LogicNode) -> LogicNode | None:
        y = self.rw(x)
        if y is not None:
            if y.is_tree():
                args = y.get_arguments()
                return y.from_arguments(*[get_or_else(self(arg), arg) for arg in args])
            return y
        if x.is_tree():
            args = x.get_arguments()
            new_args = list(map(self, args))
            if not all(arg is None for arg in new_args):
                return x.from_arguments(*map(lambda x1, x2: get_or_else(x1, x2), new_args, args))
            return None
        return None


class PostWalk:
    def __init__(self, rw: RwCallable):
        self.rw = rw

    def __call__(self, x: LogicNode) -> LogicNode | None:
        if x.is_tree():
            args = x.get_arguments()
            new_args = list(map(self, args))
            if all(arg is None for arg in new_args):
                return self.rw(x)
            y = x.from_arguments(*map(lambda x1, x2: get_or_else(x1, x2), new_args, args))
            return get_or_else(self.rw(y), y)
        return self.rw(x)


class Chain:
    def __init__(self, rws: Iterable[RwCallable]):
        self.rws = rws

    def __call__(self, x: LogicNode) -> LogicNode | None:
        is_success = False
        for rw in self.rws:
            y = rw(x)
            if y is not None:
                is_success = True
                x = y
        if is_success:
            return x
        return None


class Fixpoint:
    def __init__(self, rw: RwCallable):
        self.rw = rw

    def __call__(self, x: LogicNode) -> LogicNode | None:
        y = self.rw(x)
        if y is not None:
            while y is not None and x != y:
                x = y
                y = self.rw(x)
        else:
            return None
