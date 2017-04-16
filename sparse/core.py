import pandas as pd
import numpy as np

class COO(object):
    def __init__(self, coords, data=None, shape=None):
        if shape is None:
            shape = tuple(ind.max(axis=0).tolist())
        if data is None and isinstance(coords, (tuple, list)):
            if coords:
                assert len(coords[0]) == 2
            data = list(pluck(1, coords))
            coords = list(pluck(0, coords))

        self.shape = shape
        self.data = np.asarray(data)
        self.coords = np.asarray(coords).astype(np.uint64)

    @classmethod
    def from_numpy(cls, x):
        coords = np.where(x)
        data = x[coords]
        coords = np.vstack(coords).T
        return cls(coords, data, shape=x.shape)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nnz(self):
        return self.coords.shape[0]

    @property
    def nbytes(self):
        return self.data.nbytes + self.coords.nbytes

    def __getitem__(self, ind):
        ind = np.asarray(ind)
        for i in range(self.nnz):
            if (self.coords[i] == ind).all():
                return self.data[i]

    def __str__(self):
        return "<COO: shape=%s, dtype=%s, nnz=%d>" % (self.shape, self.dtype,
                self.nnz)

    __repr__ = __str__

    def sum(self, axis=None):
        if axis is None:
            return self.data.sum()

        neg_axis = list(range(self.ndim))
        if isinstance(axis, int):
            neg_axis.remove(axis)
        else:
            for ax in axis:
                neg_axis.remove(ax)

        df = pd.DataFrame(self.coords[:, neg_axis])
        columns = list(df.columns)
        df['.data'] = self.data

        result = df.groupby(columns)['.data'].sum()
        data = result.values
        coords = result.reset_index()[columns].values

        return COO(coords, data, tuple(self.shape[i] for i in neg_axis))

    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))

        shape = tuple(self.shape[ax] for ax in axes)
        return COO(self.coords[:, axes], self.data, shape)

    def reshape(self, shape):
        linear_loc = np.zeros(self.nnz, dtype=np.uint64)
        strides = 1
        for i, d in enumerate(self.shape[::-1]):
            linear_loc += self.coords[:, -(i + 1)] * strides
            strides *= d

        coords = np.empty((self.nnz, len(shape)), dtype=self.coords.dtype)
        strides = 1
        for i, d in enumerate(shape[::-1]):
            coords[:, -(i + 1)] = linear_loc // strides % d
            strides *= d

        return COO(coords, self.data, shape)

    def __array__(self, *args, **kwargs):
        x = np.zeros(shape=self.shape, dtype=self.dtype)
        coords = tuple([self.coords[:, i] for i in range(self.ndim)])
        x[coords] = self.data
        return x
