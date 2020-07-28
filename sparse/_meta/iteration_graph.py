import operator
import typing
import numbers

from .._utils import normalize_axis


class Access(object):
    def __init__(self, idxs, *, ndim):
        if not isinstance(ndim, numbers.Integral):
            raise ValueError("ndim must be an int")
        for idx in idxs:
            if not isinstance(idx, numbers.Integral):
                raise ValueError("idxs must be a sequence of int")

            if not (0 <= idx < ndim):
                raise ValueError("axis out of range")

        self._idxs = tuple(operator.index(i) for i in idxs)
        self._ndim = operator.index(ndim)

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def base_ndim(self) -> int:
        return len(self.idxs)

    @property
    def idxs(self) -> typing.List[int]:
        return self._idxs

    def broadcast(self, ndim) -> "Access":
        ndim_diff = ndim - self.ndim
        if ndim_diff < 0:
            raise ValueError("Cannot make access smaller")

        idxs = [k + ndim_diff for k in self.idxs]
        return Access(idxs, ndim=ndim)

    def transpose(self, axes=None) -> "Access":
        if axes is None:
            axes = tuple(range(self.ndim)[::-1])

        axes = normalize_axis(axes, self.ndim)
        if not set(axes) == set(range(self.ndim)):
            raise ValueError("axes don't match array")

        idxs = [axes[k] for k in self.idxs]
        return Access(idxs, ndim=self.ndim)

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

        return Access(idxs, ndim=ndim + num_additonal_dims)

    def __getitem__(self, i):
        """
        >>> a = Access((0, 1), ndim=2)
        >>> a[:, None, :, None]
        Access((0, 3), ndim=4)
        >>> a[None]
        Access((1, 2), ndim=3)
        >>> a[()]
        Access((0, 1), ndim=2)
        >>> a[:]
        Access((0, 1), ndim=2)
        >>> a[:, :]
        Access((0, 1), ndim=2)
        >>> a[None, None, :]
        Access((2, 3), ndim=4)
        """
        raise NotImplementedError

    @classmethod
    def from_ndim(cls, ndim):
        return Access(range(ndim), ndim=ndim)

    def __str__(self):
        return f"Access({self.idxs}, ndim={self.ndim})"

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, Access):
            return NotImplemented

        return self.idxs == other.idxs and self.ndim == other.ndim

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

        if len(args) != 0:
            self._broadcast()

    def _broadcast(self):
        ndim_max = max(a.ndim for a in self._args)
        self._args = [a.broadcast(ndim_max) for a in self._args]

    @property
    def ndim(self) -> int:
        return self.args[0].ndim if len(self.args) else 0

    @property
    def args(self):
        return self._args

    def __getitem__(self, k):
        args = [arg[k] for arg in self.args]
        return IterationGraph(*args)

    def has_cycle(self):
        raise NotImplementedError()

    def root(self):
        raise NotImplementedError()

    def __eq__(self, other):
        if not isinstance(other, IterationGraph):
            return NotImplemented

        return self.args == other.args
