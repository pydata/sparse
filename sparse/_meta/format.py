import typing
from .iteration_graph import Access
from .sparsedim import SparseDim


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


class LazyTensor(object):
    def __init__(self, *, shape: typing.Tuple[int, ...], format: Format):
        assert len(shape) == len(format)
        self._shape = shape
        self._format = format

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def format(self):
        return self._format

    def __str__(self):
        s = f"{self.format.name}"
        z = zip(self.shape, self.format.levels)
        s += "(" + ", ".join(f"{level.__name__}[{dim}]" for dim, level in z) + ")"
        return s

    def __getitem__(self, t):
        if isinstance(t, slice):
            t = (t,)

        return Access.from_numpy_notation(key=t, ndim=self.ndim)


class Tensor(LazyTensor):
    pass
