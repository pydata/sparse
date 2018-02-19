import numpy as np
from numbers import Integral
from collections import Iterable


def assert_eq(x, y, check_nnz=True, compare_dtype=True, **kwargs):
    from .coo import COO
    assert x.shape == y.shape

    if compare_dtype:
        assert x.dtype == y.dtype

    if isinstance(x, COO):
        if x.sorted:
            assert is_lexsorted(x)
    if isinstance(y, COO):
        if y.sorted:
            assert is_lexsorted(y)

    if hasattr(x, 'todense'):
        xx = x.todense()
        if check_nnz:
            assert (xx != 0).sum() == x.nnz
    else:
        xx = x
    if hasattr(y, 'todense'):
        yy = y.todense()
        if check_nnz:
            assert (yy != 0).sum() == y.nnz
    else:
        yy = y
    assert np.allclose(xx, yy, **kwargs)


def is_lexsorted(x):
    return not x.shape or (np.diff(x.linear_loc()) > 0).all()


def _zero_of_dtype(dtype):
    """
    Creates a ()-shaped 0-dimensional zero array of a given dtype.

    Parameters
    ----------
    dtype : numpy.dtype
        The dtype for the array.

    Returns
    -------
    np.ndarray
        The zero array.
    """
    return np.zeros((), dtype=dtype)


def random(
        shape,
        density=0.01,
        canonical_order=False,
        random_state=None,
        data_rvs=None,
        format='coo'
):
    """ Generate a random sparse multidimensional array

    Parameters
    ----------
    shape: Tuple[int]
        Shape of the array
    density: float, optional
        Density of the generated array.
    canonical_order : bool, optional
        Whether or not to put the output :obj:`COO` object into canonical
        order. :code:`False` by default.
    random_state : Union[numpy.random.RandomState, int], optional
        Random number generator or random seed. If not given, the
        singleton numpy.random will be used. This random state will be used
        for sampling the sparsity structure, but not necessarily for sampling
        the values of the structurally nonzero entries of the matrix.
    data_rvs : Callable
        Data generation callback. Must accept one single parameter: number of
        :code:`nnz` elements, and return one single NumPy array of exactly
        that length.
    format: {'coo', 'dok'}
        The format to return the output array in.

    Returns
    -------
    {COO, DOK}
        The generated random matrix.

    See Also
    --------
    :obj:`scipy.sparse.rand`
        Equivalent Scipy function.
    :obj:`numpy.random.rand`
        Similar Numpy function.

    Examples
    --------

    >>> from sparse import random
    >>> from scipy import stats
    >>> rvs = lambda x: stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(1))
    >>> s = random((2, 3, 4), density=0.25, random_state=np.random.RandomState(1), data_rvs=rvs)
    >>> s.todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  0,  0,  0],
            [ 0, 34,  0,  0],
            [33, 34,  0, 29]],
    <BLANKLINE>
           [[30,  0,  0, 34],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0]]])

    """
    # Copied, in large part, from scipy.sparse.random
    # See https://github.com/scipy/scipy/blob/master/LICENSE.txt
    from .coo import COO
    from .dok import DOK

    elements = np.prod(shape)

    nnz = int(elements * density)

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, Integral):
        random_state = np.random.RandomState(random_state)
    if data_rvs is None:
        data_rvs = random_state.rand

    # Use the algorithm from python's random.sample for k < mn/3.
    if elements < 3 * nnz:
        ind = random_state.choice(elements, size=nnz, replace=False)
    else:
        ind = np.empty(nnz, dtype=np.min_scalar_type(elements - 1))
        selected = set()
        for i in range(nnz):
            j = random_state.randint(elements)
            while j in selected:
                j = random_state.randint(elements)
            selected.add(j)
            ind[i] = j

    data = data_rvs(nnz)

    ar = COO(ind[None, :], data, shape=nnz).reshape(shape)

    if canonical_order:
        ar.sum_duplicates()

    if format == 'dok':
        ar = DOK(ar)

    return ar


def isscalar(x):
    from .sparse_array import SparseArray
    return not isinstance(x, SparseArray) and np.isscalar(x)


class PositinalArgumentPartial(object):
    def __init__(self, func, pos, posargs):
        if not isinstance(pos, Iterable):
            pos = (pos,)
            posargs = (posargs,)

        n_partial_args = len(pos)

        self.pos = pos
        self.posargs = posargs
        self.func = func

        self.n = n_partial_args

        self.__doc__ = func.__doc__

    def __call__(self, *args, **kwargs):
        j = 0
        totargs = []

        for i in range(len(args) + self.n):
            if j >= self.n or i != self.pos[j]:
                totargs.append(args[i - j])
            else:
                totargs.append(self.posargs[j])
                j += 1

        return self.func(*totargs, **kwargs)

    def __str__(self):
        return str(self.func)

    def __repr__(self):
        return repr(self.func)


def random_value_array(value, fraction):
    def replace_values(n):
        i = int(n * fraction)

        ar = np.empty((n,), dtype=np.float_)
        ar[:i] = value
        ar[i:] = np.random.rand(n - i)
        return ar

    return replace_values
