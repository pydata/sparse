from collections.abc import Sequence
from numbers import Integral
from abc import abstractmethod, ABCMeta
from .sparsedim import SparseDim

__all__ = ["SparseMeta"]


class SparseMeta(ABCMeta):
    def __init__(cls, name, bases, slots):
        if "SparseBase" not in globals():
            super().__init__(name, bases, slots)
            return

        slots = dict(slots)
        if (
            "__sparse_dims__" not in slots
            or not isinstance(slots["__sparse_dims__"], Sequence)
            or not all(isinstance(s, SparseDim) for s in slots["__sparse_dims__"])
        ):
            raise TypeError(
                "Sparse classes must have a __sparse_dims__ attribute that is a sequence of SparseDim."
            )

        slots["__sparse_dims__"] = tuple(slots["__sparse_dims__"])

        if "__sparse_dim_order__" not in slots:
            slots["__sparse_dim_order__"] = range(len(slots["__sparse_dims__"]))

        if (
            not isinstance(slots["__sparse_dim_order__"], Sequence)
            or not all(isinstance(s, Integral) for s in slots["__sparse_dim_order__"])
            or len(slots["__sparse_dim_order__"]) != len(slots["__sparse_dims__"])
        ):
            raise TypeError(
                "SparseBase subclasses can only have a __sparse_dim_order__ attribute that is a sequence of integers "
                "with the same length as __sparse_dims__."
            )

        slots["__sparse_dim_order__"] = tuple(
            int(d) for d in slots["__sparse_dim_order__"]
        )

        super().__init__(name, bases, slots)


class SparseBase(metaclass=SparseMeta):
    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    def ndim(self):
        return len(self.shape)
