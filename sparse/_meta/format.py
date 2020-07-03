import typing
from abc import abstractmethod, ABC

import numpy as np
import dask
import dask.base

from .iteration_graph import IterationGraph, Access
from .sparsedim import SparseDim
import uuid


class Format(object):
    def __init__(self, *, name: str = None, levels: typing.Tuple[SparseDim, ...]):
        self._levels = levels
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def levels(self):
        return self._levels

    def __str__(self):
        levels_str = tuple(map(lambda x: str(x.__name__), self.levels))
        s = f"{self.name}"
        s += "(" + ", ".join(levels_str) + ")"
        return s

    def __len__(self):
        return len(self.levels)


class TensorBase(ABC, np.lib.mixins.NDArrayOperatorsMixin):
    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @property
    @abstractmethod
    def access(self) -> Access:
        pass

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if kwargs.pop("out", None) is not None:
            raise NotImplementedError

        if method != "__call__":
            raise NotImplementedError

        if getattr(ufunc, "signature", None) is not None:
            raise NotImplementedError

        if not all(isinstance(t, TensorBase) for t in inputs):
            raise NotImplementedError

        ndim = max(t.ndim for t in inputs)
        access = Access.from_ndim(ndim)

        return LazyTensor(tensors=inputs, access=access, op=ufunc)

    def __getitem__(self, k):
        return LazyTensor((self,), access=Access.from_numpy_notation(k, ndim=self.ndim))


class Tensor(TensorBase):
    def __init__(self, *, shape: typing.Tuple[int, ...], dims: Format):
        assert len(shape) == len(dims)
        self._shape = shape
        self._dims = dims
        self._key = f"Tensor-f{uuid.uuid4()}"

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        return self._shape

    @property
    def dims(self) -> Format:
        return self._dims

    @property
    def access(self):
        return Access.from_ndim(self.ndim)

    def __str__(self):
        s = ""
        s += f"{self.dims.name}"
        z = zip(self.shape, self.dims.levels)
        s += "(" + ", ".join(f"{level.__name__}[{dim}]" for dim, level in z) + ")"
        return s

    def __getitem__(self, t):
        if isinstance(t, slice):
            t = (t,)

        access = Access.from_numpy_notation(key=t, ndim=self.ndim)
        return Lazy.from_access(self, access)

    __repr__ = __str__


def identity(x):
    return x


identity.nin = 1


class LazyTensor(TensorBase, dask.base.DaskMethodsMixin):
    def __init__(
        self,
        tensors: typing.Sequence[TensorBase],
        access: Access,
        op: typing.Optional[typing.Callable] = identity,
    ):
        if hasattr(op, "nin") and not len(tensors) == op.nin:
            raise ValueError(
                f"operator {op!r} requires {op.nin} args, received {len(op)}."
            )

        self_key = f"{op}-{access!r}-{uuid.uuid4()}"
        other_keys = []
        dsk = {}
        for t in tensors:
            if isinstance(t, Tensor):
                dsk[t._key] = t
                other_keys.append(t._key)
            elif isinstance(t, LazyTensor):
                dsk.update(t.__dask_graph__())
                other_keys.append(t._key)
            else:
                raise NotImplementedError(type(t))
        dsk[self_key] = (op, *other_keys)
        self._key = self_key
        self._dsk = dsk
        self._tensors = tuple(tensors)
        self._op = op
        self._access = access

    def __dask_graph__(self):
        return self._dsk

    def __dask_keys__(self):
        return (self._key,)

    __dask_scheduler__ = staticmethod(dask.get)

    def __dask_tokenize__(self):
        return (self.op, *self.tensors)

    def __dask_postcompute__(self):
        return identity, ()

    def __dask_postpersist__(self):
        return LazyTensor, (self.tensors, self.op)

    @property
    def ndim(self) -> int:
        return self.access.ndim

    @property
    def tensors(self) -> typing.List[TensorBase]:
        return self._tensors

    @property
    def op(self) -> typing.Callable:
        return self._op

    @property
    def access(self) -> Access:
        return self._access

    def __repr__(self) -> str:
        return f"LazyTensor(op={self.op!r}, tensors={self.tensors!r}, access={self.access!r})"

    __str__ = __repr__
