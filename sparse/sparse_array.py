from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod
from collections import Iterable
from numbers import Integral
from functools import reduce
import operator


class SparseArray(object):
    """
    An abstract base class for all the sparse array classes.

    Attributes
    ----------
    dtype : numpy.dtype
        The data type of this array.
    """

    __metaclass__ = ABCMeta

    def __init__(self, shape):
        if not isinstance(shape, Iterable):
            shape = (int(shape),)

        if not all(isinstance(l, Integral) and int(l) >= 0 for l in shape):
            raise ValueError('shape must be an non-negative integer or a tuple '
                             'of non-negative integers.')

        self.shape = tuple(int(l) for l in shape)

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
        pass

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
