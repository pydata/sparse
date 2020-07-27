import typing
from abc import abstractmethod, ABC

from collections import defaultdict

import numpy as np
import dask
import dask.base

from .dense_level import Dense
from .compressed_level import Compressed
from .iteration_graph import IterationGraph, Access
from .sparsedim import SparseDim, InlineAssembly, AppendAssembly
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
        self._levels = []

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
        {(): [0, 2]}

        >>> group_coords(coords=c, idx=1)
        {(0,): [0, 2], (2,): [0, 2, 3]}

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
        ret = {}
        for k, v in d.items():
            ret[k] = list(sorted(v))
        return ret

    def insert_data(self, *, coords, data):
        # coords = [(i1, i2, ..., iN)]
        # data = [x1, x2, ..., xN]
        self._data = data

        # size of the previous level
        szkm1 = 1
        parent = None
        for idx, level in enumerate(self._fmt.levels):
            if level == Dense:
                fn = lambda: Dense(N=self.shape[idx])
            elif level == Compressed:
                fn = lambda: Compressed(pos=[], crd=[])
            else:
                raise NotImplementedError(level)

            level = fn()
            if isinstance(level,)

            parent, szkm1 = fn(coords=coords, idx=idx, parent=parent, szkm1=szkm1)
            self._levels.append(parent)

    def _init_dense_level(self, *, idx, coords, parent=None, szkm1=None):
        assert szkm1 is not None

        N = self.shape[idx]
        d = Dense(N=N)
        szk = d.size(szkm1)
        d.insert_init(szkm1, szk)
        d.insert_finalize(0, szk)
        return d, szk

    def iterate(self, levels):
        if len(levels) == n:
            return 0, ()
        for p0, i0 in levels[0].iterate():
            for p, i in self.iterate(levels[1:]):
                yield p, i0 + (i,) 

    def _init_compressed_level(self, *, idx, coords, parent=None, szkm1=None):
        assert szkm1 is not None

        # Algo 1
        # for pk, ik in parent.iterate(pkm1, (i0, i1, ..., ikm1)):
        #     level.append/insert_coord
        # level.append/insert_edges

        # for k, level in enumerate(levels):
        #     init level
        #     for (levels[:k] if k >= 2 else ()).iterate():
        #         algo 1 for level

        #     finl level


        c = Compressed(pos=[], crd=[])
        group = self.group_coords(coords=coords, idx=idx)
        # Count the number of elements in each key of group
        # the size of a compressed level is the number of elements
        # that will be inserted in the crd array
        szk = 0
        for k in group.keys():
            szk += len(group[k])
        
        c.append_init(szkm1, szk)

        # if we are at the top level
        if parent is None:
            pk = 0
            for i in group.keys():
                pbegink = pk
                for v in group[i]:
                    c.append_coord(pk, v)

                pk += len(group[i])
                pendk = pk
            c.append_edges(pkm1, pbegink, pendk)
        else:
            for i in group.keys():
                for pkm1, ikm1 in parent.iterate(pkm2, i):
                    pbegink = pk
                    for v in group[i]:
                        c.append_coord(pk, v)

                    pk += len(group[i])
                    pendk = pk

                    # if isinstance(parent, Dense):
                    #     pkm1 = i[-1]
                    c.append_edges(pkm1, pbegink, pendk)

        c.append_finalize(szkm1, szk)
        return c, szk

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
