import operator
import typing
import collections.abc as cabc
import numbers

from .._utils import normalize_axis


class Access(object):
    def __init__(self, idxs, *, ndim, is_output=False):
        if not isinstance(ndim, numbers.Integral):
            raise ValueError("ndim must be an int")
        if not isinstance(idxs, cabc.Mapping):
            raise ValueError("idxs must be a Mapping")
        if not set(idxs.values()) == set(range(len(idxs))):
            raise ValueError("idxs must have values spanning the entire range of axes")
        for idx in idxs.keys():
            if not isinstance(idx, numbers.Integral):
                raise ValueError("idxs must be a sequence of int")

            if not (0 <= idx < ndim):
                raise ValueError("axis out of range")

        self._idxs = dict(idxs)
        self._ndim = int(ndim)
        self._is_output = is_output

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def base_ndim(self) -> int:
        return len(self.idxs)

    @property
    def idxs(self) -> typing.Mapping[int, int]:
        return self._idxs

    @property
    def is_output(self) -> bool:
        return self._is_output

    def broadcast(self, ndim) -> "Access":
        ndim_diff = ndim - self.ndim
        if ndim_diff < 0:
            raise ValueError("Cannot make access smaller")

        idxs = {k + ndim_diff: v for k, v in self.idxs.items()}
        return Access(idxs, ndim=ndim, is_output=self.is_output)

    def transpose(self, axes=None) -> "Access":
        if axes is None:
            axes = tuple(range(self.ndim)[::-1])

        axes = normalize_axis(axes, self.ndim)
        if not set(axes) == set(range(self.ndim)):
            raise ValueError("axes don't match array")

        idxs = {axes[k]: v for k, v in self.idxs.items()}
        return Access(idxs, ndim=self.ndim, is_output=self.is_output)

    @classmethod
    def from_numpy_notation(cls, key: typing.Tuple, *, ndim, is_output=False):
        idx = 0
        idxs = []
        num_additonal_dims = 0
        for e in key:
            if isinstance(e, slice) and e == slice(None):
                # slice(...)
                idxs.append(idx)
                idx += 1
            elif e is None:
                idx += 1
                num_additonal_dims += 1
            else:
                raise NotImplementedError(e)
        if len(idxs) > ndim:
            raise ValueError("too many indices for Access.")

        while len(idxs) < ndim:
            idxs.append(idx)
            idx += 1

        return Access(
            {k: v for v, k in enumerate(idxs)},
            ndim=ndim + num_additonal_dims,
            is_output=is_output,
        )

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

    def __eq__(self, other):
        if not isinstance(other, Access):
            return NotImplemented

        return (
            self.idxs == other.idxs
            and self.ndim == other.ndim
            and self.is_output == other.is_output
        )

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

        self._broadcast()

    def _broadcast(self):
        ndim_max = max(a.ndim for a in self._args)
        self._args = [a.broadcast(ndim_max) for a in self._args]

    @property
    def args(self):
        return self._args

    def has_cycle(self):
        raise NotImplementedError()

    def root(self):
        raise NotImplementedError()

    def __eq__(self, other):
        if not isinstance(other, IterationGraph):
            return NotImplemented

        return self.args == other.args
