import typing
from abc import abstractmethod, ABC

from collections import defaultdict

import numpy as np
import dask
import dask.base

from .dense_level import Dense
from .compressed_level import Compressed
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
    def __init__(self, *, shape: typing.Tuple[int, ...], fmt: Format):
        assert len(shape) == len(fmt)
        self._shape = shape
        self._fmt = fmt
        self._key = f"Tensor-f{uuid.uuid4()}"
        self._data = []
        self._levels = [None] * len(shape)

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        return self._shape

    @property
    def fmt(self) -> Format:
        return self._fmt

    @property
    def access(self):
        return Access.from_ndim(self.ndim)

    @property
    def data(self):
        return self._data

    @property
    def levels(self):
        return self._levels

    def group_coords(self, *, coords, idx):
        """
        group coordinates of a given index based on previous ones

        >>> c = [(0, 0, 0), (0, 0, 1), (0, 2, 1), (2, 0, 1), 
                 (2, 2, 0), (2, 2, 1), (2, 3, 0), (2, 3, 1)]

        >>> group_coords(coords=c, idx=0)
        {(): [0, 0, 0, 2, 2, 2, 2, 2]}

        >>> group_coords(coords=c, idx=1)
        {(0,): [0, 0, 2], (2,): [0, 2, 2, 3, 3]}

        >>> group_coords(coords=c, idx=2)
        {(0, 0): [0, 1],
         (0, 2): [1],
         (2, 0): [1],
         (2, 2): [0, 1],
         (2, 3): [0, 1]}
        """
        d = defaultdict(set)
        for i, c in enumerate(coords):
            prev = tuple(c[:idx]) if idx != 0 else (0,)
            curr = c[idx]
            d[prev].add(curr)
        return d

    def insert_data(self, *, coords, data):
        # coords = [(i1, i2, ..., iN)]
        # data = [x1, x2, ..., xN]
        self._data = data
        # The size of the zero level is 1 (unary level)
        self._init_level(idx=0, coords=coords, szkm1=1)

    def _store_level(self, *, level, idx):
        self._levels[idx] = level

    def _init_level(self, *, idx, coords, szkm1=None):
        if idx >= self.ndim:
            return None

        assert szkm1 is not None

        level = self._fmt.levels[idx]
        if level == Dense:
            fn = self._init_dense_level
        elif level == Compressed:
            fn = self._init_compressed_level
        else:
            raise NotImplementedError(level)

        return fn(idx=idx, coords=coords, szkm1=szkm1)

    def _init_dense_level(self, *, idx, coords, szkm1=None):
        assert szkm1 is not None

        N = self.shape[idx]
        d = Dense(N=N)
        self._store_level(level=d, idx=idx)

        szk = d.size(szkm1)
        d.insert_init(szkm1, szk)

        # init next levels
        self._init_level(idx=idx + 1, coords=coords, szkm1=szk)

        d.insert_finalize(0, szk)

    def _init_compressed_level(self, *, idx, coords, szkm1=None):
        assert szkm1 is not None

        c = Compressed(pos=[], crd=[])
        self._store_level(level=c, idx=idx)

        group = self.group_coords(coords=coords, idx=idx)
        # Count the number of elements in each key of group
        # the size of a compressed level is the number of elements
        # that will be inserted in the crd array
        szk = 0
        for k in group.keys():
            szk += len(group[k])

        c.append_init(szkm1, szk)

        pk = 0
        for k in group.keys():
            pbegink = pk
            pkm1 = k[-1]
            print(k, c.pos)

            for v in group[k]:
                c.append_coord(pk, v)

            pk += len(group[k])
            pendk = pk

            c.append_edges(pkm1, pbegink, pendk)

        # init next levels
        self._init_level(idx=idx + 1, coords=coords, szkm1=szk)

        c.append_finalize(szkm1, szk)

    def __str__(self):
        s = ""
        s += f"{self.fmt.name}"
        z = zip(self.shape, self.fmt.levels)
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
