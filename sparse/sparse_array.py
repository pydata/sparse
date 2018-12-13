from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod
from collections import Iterable
from numbers import Integral
from functools import reduce
import operator

import numpy as np

from .utils import _zero_of_dtype
from .compatibility import int


class SparseArray(object):
    """
    An abstract base class for all the sparse array classes.

    Attributes
    ----------
    dtype : numpy.dtype
        The data type of this array.
    fill_value : scalar
        The fill value of this array.
    """

    __metaclass__ = ABCMeta

    def __init__(self, shape, fill_value=None):
        if not isinstance(shape, Iterable):
            shape = (shape,)

        if not all(isinstance(l, Integral) and int(l) >= 0 for l in shape):
            raise ValueError('shape must be an non-negative integer or a tuple '
                             'of non-negative integers.')

        self.shape = tuple(int(l) for l in shape)

        if fill_value is not None:
            if not hasattr(fill_value, 'dtype') or fill_value.dtype != self.dtype:
                self.fill_value = self.dtype.type(fill_value)
            else:
                self.fill_value = fill_value
        else:
            self.fill_value = _zero_of_dtype(self.dtype)

    dtype = None

    @property
    @abstractmethod
    def nnz(self):
        """
        The number of nonzero elements in this array. Note that any duplicates in
        :code:`coords` are counted multiple times. To avoid this, call :obj:`COO.sum_duplicates`.

        Returns
        -------
        int
            The number of nonzero elements in this array.

        See Also
        --------
        DOK.nnz : Equivalent :obj:`DOK` array property.
        numpy.count_nonzero : A similar Numpy function.
        scipy.sparse.coo_matrix.nnz : The Scipy equivalent property.

        Examples
        --------
        >>> import numpy as np
        >>> from sparse import COO
        >>> x = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 0])
        >>> np.count_nonzero(x)
        6
        >>> s = COO.from_numpy(x)
        >>> s.nnz
        6
        >>> np.count_nonzero(x) == s.nnz
        True
        """

    @property
    def ndim(self):
        """
        The number of dimensions of this array.

        Returns
        -------
        int
            The number of dimensions of this array.

        See Also
        --------
        DOK.ndim : Equivalent property for :obj:`DOK` arrays.
        numpy.ndarray.ndim : Numpy equivalent property.

        Examples
        --------
        >>> from sparse import COO
        >>> import numpy as np
        >>> x = np.random.rand(1, 2, 3, 1, 2)
        >>> s = COO.from_numpy(x)
        >>> s.ndim
        5
        >>> s.ndim == x.ndim
        True
        """
        return len(self.shape)

    @property
    def size(self):
        """
        The number of all elements (including zeros) in this array.

        Returns
        -------
        int
            The number of elements.

        See Also
        --------
        numpy.ndarray.size : Numpy equivalent property.

        Examples
        --------
        >>> from sparse import COO
        >>> import numpy as np
        >>> x = np.zeros((10, 10))
        >>> s = COO.from_numpy(x)
        >>> s.size
        100
        """
        # We use this instead of np.prod because np.prod
        # returns a float64 for an empty shape.
        return reduce(operator.mul, self.shape, 1)

    @property
    def density(self):
        """
        The ratio of nonzero to all elements in this array.

        Returns
        -------
        float
            The ratio of nonzero to all elements.

        See Also
        --------
        COO.size : Number of elements.
        COO.nnz : Number of nonzero elements.

        Examples
        --------
        >>> import numpy as np
        >>> from sparse import COO
        >>> x = np.zeros((8, 8))
        >>> x[0, :] = 1
        >>> s = COO.from_numpy(x)
        >>> s.density
        0.125
        """
        return self.nnz / self.size

    @abstractmethod
    def asformat(self, format):
        """
        Convert this sparse array to a given format.

        Parameters
        ----------
        format : str
            A format string.

        Returns
        -------
        out : SparseArray
            The converted array.

        Raises
        ------
        NotImplementedError
            If the format isn't supported.
        """

    @abstractmethod
    def todense(self):
        """
        Convert this :obj:`SparseArray` array to a dense :obj:`numpy.ndarray`. Note that
        this may take a large amount of memory and time.

        Returns
        -------
        numpy.ndarray
            The converted dense array.

        See Also
        --------
        DOK.todense : Equivalent :obj:`DOK` array method.
        COO.todense : Equivalent :obj:`COO` array method.
        scipy.sparse.coo_matrix.todense : Equivalent Scipy method.

        Examples
        --------
        >>> import sparse
        >>> x = np.random.randint(100, size=(7, 3))
        >>> s = sparse.COO.from_numpy(x)
        >>> x2 = s.todense()
        >>> np.array_equal(x, x2)
        True
        """

    def __array__(self, **kwargs):
        from . import _AUTO_DENSIFICATION_ENABLED as densify
        if not densify:
            raise RuntimeError('Cannot convert a sparse array to dense automatically. '
                               'To manually densify, use the todense method.')

        return np.asarray(self.todense(), **kwargs)
