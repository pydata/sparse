from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(eq=True, frozen=True)
class LogicNode:
    @staticmethod
    @abstractmethod
    def is_tree(): ...

    @staticmethod
    @abstractmethod
    def is_stateful(): ...

    @abstractmethod
    def get_arguments(self): ...

    @classmethod
    @abstractmethod
    def from_arguments(cls): ...


@dataclass(eq=True, frozen=True)
class Immediate(LogicNode):
    val: Any

    @staticmethod
    def is_tree():
        return False

    @staticmethod
    def is_stateful():
        return False

    def get_arguments(self):
        return [self.val]

    @classmethod
    def from_arguments(cls, val):
        return cls(val)


@dataclass(eq=True, frozen=True)
class Deferred(LogicNode):
    ex: Any
    type_: Any

    @staticmethod
    def is_tree():
        return False

    @staticmethod
    def is_stateful():
        return False

    def get_arguments(self):
        return [self.val, self.type_]

    @classmethod
    def from_arguments(cls, val, type_):
        return cls(val, type_)


@dataclass(eq=True, frozen=True)
class Field(LogicNode):
    name: str

    @staticmethod
    def is_tree():
        return False

    @staticmethod
    def is_stateful():
        return False

    def get_arguments(self):
        return [self.name]

    @classmethod
    def from_arguments(cls, name):
        return cls(name)


@dataclass(eq=True, frozen=True)
class Alias(LogicNode):
    name: str

    @staticmethod
    def is_tree():
        return False

    @staticmethod
    def is_stateful():
        return False

    def get_arguments(self):
        return [self.name]

    @classmethod
    def from_arguments(cls, name):
        return cls(name)


@dataclass(eq=True, frozen=True)
class Table(LogicNode):
    tns: Any
    idxs: Iterable[Any]

    @staticmethod
    def is_tree():
        return True

    @staticmethod
    def is_stateful():
        return False

    def get_arguments(self):
        return [self.tns, *self.idxs]

    @classmethod
    def from_arguments(cls, tns, *idxs):
        return cls(tns, idxs)


@dataclass(eq=True, frozen=True)
class MapJoin(LogicNode):
    op: Any
    args: Iterable[Any]

    @staticmethod
    def is_tree():
        return True

    @staticmethod
    def is_stateful():
        return False

    def get_arguments(self):
        return [self.op, *self.args]

    @classmethod
    def from_arguments(cls, op, *args):
        return cls(op, args)


@dataclass(eq=True, frozen=True)
class Aggregate(LogicNode):
    op: Any
    init: Any
    arg: Any
    idxs: Iterable[Any]

    @staticmethod
    def is_tree():
        return True

    @staticmethod
    def is_stateful():
        return False

    def get_arguments(self):
        return [self.op, self.init, self.arg, *self.idxs]

    @classmethod
    def from_arguments(cls, op, init, arg, *idxs):
        return cls(op, init, arg, idxs)


@dataclass(eq=True, frozen=True)
class Reorder(LogicNode):
    arg: Any
    idxs: Iterable[Any]

    @staticmethod
    def is_tree():
        return True

    @staticmethod
    def is_stateful():
        return False

    def get_arguments(self):
        return [self.arg, *self.idxs]

    @classmethod
    def from_arguments(cls, arg, *idxs):
        return cls(arg, idxs)


@dataclass(eq=True, frozen=True)
class Relabel(LogicNode):
    arg: Any
    idxs: Iterable[Any]

    @staticmethod
    def is_tree():
        return True

    @staticmethod
    def is_stateful():
        return False

    def get_arguments(self):
        return [self.arg, *self.idxs]

    @classmethod
    def from_arguments(cls, arg, *idxs):
        return cls(arg, idxs)


@dataclass(eq=True, frozen=True)
class Reformat(LogicNode):
    tns: Any
    arg: Any

    @staticmethod
    def is_tree():
        return True

    @staticmethod
    def is_stateful():
        return False

    def get_arguments(self):
        return [self.tns, self.arg]

    @classmethod
    def from_arguments(cls, tns, arg):
        return cls(tns, arg)


@dataclass(eq=True, frozen=True)
class Subquery(LogicNode):
    lhs: Any
    arg: Any

    @staticmethod
    def is_tree():
        return True

    @staticmethod
    def is_stateful():
        return False

    def get_arguments(self):
        return [self.lhs, self.arg]

    @classmethod
    def from_arguments(cls, lhs, arg):
        return cls(lhs, arg)


@dataclass(eq=True, frozen=True)
class Query(LogicNode):
    lhs: Any
    rhs: Any

    @staticmethod
    def is_tree():
        return True

    @staticmethod
    def is_stateful():
        return True

    def get_arguments(self):
        return [self.lhs, self.rhs]

    @classmethod
    def from_arguments(cls, lhs, rhs):
        return cls(lhs, rhs)


@dataclass(eq=True, frozen=True)
class Produces(LogicNode):
    args: Iterable[Any]

    @staticmethod
    def is_tree():
        return True

    @staticmethod
    def is_stateful():
        return True

    def get_arguments(self):
        return [*self.args]

    @classmethod
    def from_arguments(cls, *args):
        return cls(args)


@dataclass(eq=True, frozen=True)
class Plan(LogicNode):
    bodies: Iterable[Any] = ()

    @staticmethod
    def is_tree():
        return True

    @staticmethod
    def is_stateful():
        return True

    def get_arguments(self):
        return [*self.bodies]

    @classmethod
    def from_arguments(cls, *args):
        return cls(args)
