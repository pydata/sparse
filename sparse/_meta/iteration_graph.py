import typing
import collections.abc as cabc
from collections import defaultdict

from .._utils import normalize_axis


class Access(object):
    def __init__(self, idxs, *, ndim, is_output=False):
        assert isinstance(ndim, int)
        assert ndim >= 0
        assert isinstance(idxs, cabc.Sequence)
        for idx in idxs:
            assert isinstance(idx, int)

        self._idxs = idxs
        self._ndim = ndim
        self._is_output = is_output

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def idxs(self):
        return self._idxs

    @property
    def is_output(self):
        return self._is_output

    def broadcast(self, ndim):
        ndim_diff = ndim - self.ndim
        if ndim_diff < 0:
            raise ValueError("Cannot make access smaller")

        idxs = []
        for idx in self.idxs:
            idxs.append(idx + ndim_diff)

        return Access(idxs, ndim=ndim, is_output=self.is_output)

    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(range(self.ndim)[::-1])

        axes = normalize_axis(axes, self.ndim)
        if not set(axes) == set(range(self.ndim)):
            raise ValueError("axes don't match array")

        idxs = [axes[i] for i in self.idxs]
        return Access(idxs, ndim=self.ndim, is_output=self.is_output)

    @classmethod
    def from_numpy_notation(cls, key: typing.Tuple, *, ndim, is_output=False):
        idx = 0
        idxs = []
        for e in key:
            if isinstance(e, slice) and e == slice(None):
                # slice(...)
                idxs.append(idx)
                idx += 1
            elif e is None:
                idx += 1
            else:
                raise NotImplementedError(e)
        if len(idxs) > ndim:
            raise ValueError("too many indices for Access.")

        while len(idxs) < ndim:
            idxs.append(idx)
            idx += 1

        return Access(idxs, ndim=len(key) + ndim - len(idxs), is_output=is_output)

    def __add__(self, other):
        return IterationGraph(self, other)

    def __sub__(self, other):
        return IterationGraph(self, other)

    def __mul__(self, other):
        return IterationGraph(self, other)

    def __div__(self, other):
        return IterationGraph(self, other)

    def __iter__(self):
        yield from self._idxs

    def __getitem__(self, i):
        return self._idxs[i]

    def __str__(self):
        return f"Access({self.idxs}, ndim={self.ndim}, is_output={self.is_output})"

    __repr__ = __str__

    def pairwise(self):
        ret = {(-1, self.idxs[0])}
        for i in range(0, len(self.idxs) - 1):
            ret.add((self.idxs[i], self.idxs[i + 1]))

        return ret


class IterationGraph(object):
    def __init__(self, *args):
        self._args: typing.List[Access] = []

        for arg in args:
            if isinstance(arg, Access):
                self._args.append(arg)
            elif isinstance(arg, IterationGraph):
                self._args.extend(arg._args)
            else:
                raise ValueError(arg)
        self._graph = defaultdict(list)

    def _broadcast(self):
        ndim_max = max(a.ndim for a in self._args)
        self._args = [a.broadcast(ndim_max) for a in self._args]
        return self

    def _compute(self):
        for access in self._args:
            for (u, v) in access.pairwise():
                self._graph[u].append(v)

    def has_cycle(self):
        raise NotImplementedError()

    def root(self):
        raise NotImplementedError()

    def __str__(self):
        s = ""
        for u, l in self._graph.items():
            for v in l:
                s += f"{u} ~~> {v}\n"
        return s

        from graphviz import Digraph

        g = Digraph("G")
        for u, l in self._graph.items():
            for edge in l:
                u = edge.u.name
                v = edge.v.name
                label = edge.label
                g.edge(u, v, label)
        return g

    def view(self):
        self._to_graphviz().view()

    def to_dot(self):
        return self._to_graphviz().source
