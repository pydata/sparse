import functools
from collections import Iterable
from numbers import Integral

import numpy as np


def assert_eq(x, y, check_nnz=True, compare_dtype=True, **kwargs):
    from .coo import COO
    assert x.shape == y.shape

    if compare_dtype:
        assert x.dtype == y.dtype

    check_equal = np.array_equal \
        if np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.integer) \
        else functools.partial(np.allclose, equal_nan=True)

    if isinstance(x, COO):
        assert is_canonical(x)
    if isinstance(y, COO):
        assert is_canonical(y)

    if isinstance(x, COO) and isinstance(y, COO) and check_nnz:
        assert np.array_equal(x.coords, y.coords)
        assert check_equal(x.data, y.data, **kwargs)
        assert x.fill_value == y.fill_value
        return

    if hasattr(x, 'todense'):
        xx = x.todense()
        if check_nnz:
            assert_nnz(x, xx)
    else:
        xx = x
    if hasattr(y, 'todense'):
        yy = y.todense()
        if check_nnz:
            assert_nnz(y, yy)
    else:
        yy = y
    assert check_equal(xx, yy, **kwargs)


def assert_nnz(s, x):
    fill_value = s.fill_value if hasattr(s, 'fill_value') else _zero_of_dtype(s.dtype)
    assert np.sum(~equivalent(x, fill_value)) == s.nnz


def is_canonical(x):
    return not x.shape or ((np.diff(x.linear_loc()) > 0).all() and not equivalent(x.data, x.fill_value).any())


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
    return np.zeros((), dtype=dtype)[()]


def random(
        shape,
        density=0.01,
        random_state=None,
        data_rvs=None,
        format='coo',
        fill_value=None
):
    """ Generate a random sparse multidimensional array

    Parameters
    ----------
    shape: Tuple[int]
        Shape of the array
    density: float, optional
        Density of the generated array.
    random_state : Union[numpy.random.RandomState, int], optional
        Random number generator or random seed. If not given, the
        singleton numpy.random will be used. This random state will be used
        for sampling the sparsity structure, but not necessarily for sampling
        the values of the structurally nonzero entries of the matrix.
    data_rvs : Callable
        Data generation callback. Must accept one single parameter: number of
        :code:`nnz` elements, and return one single NumPy array of exactly
        that length.
    format : str
        The format to return the output array in.
    fill_value : scalar
        The fill value of the output array.

    Returns
    -------
    SparseArray
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

    ar = COO(ind[None, :], data, shape=nnz, fill_value=fill_value).reshape(shape)

    return ar.asformat(format)


def isscalar(x):
    from .sparse_array import SparseArray
    return not isinstance(x, SparseArray) and np.isscalar(x)


def random_value_array(value, fraction):
    def replace_values(n):
        i = int(n * fraction)

        ar = np.empty((n,), dtype=np.float_)
        ar[:i] = value
        ar[i:] = np.random.rand(n - i)
        return ar

    return replace_values


def normalize_axis(axis, ndim):
    """
    Normalize negative axis indices to their positive counterpart for a given
    number of dimensions.

    Parameters
    ----------
    axis : Union[int, Iterable[int], None]
        The axis indices.
    ndim : int
        Number of dimensions to normalize axis indices against.

    Returns
    -------
    axis
        The normalized axis indices.
    """
    if axis is None:
        return None

    if isinstance(axis, Integral):
        axis = int(axis)
        if axis < 0:
            axis += ndim

        if axis >= ndim or axis < 0:
            raise ValueError('Invalid axis index %d for ndim=%d' % (axis, ndim))

        return axis

    if isinstance(axis, Iterable):
        if not all(isinstance(a, Integral) for a in axis):
            raise ValueError("axis %s not understood" % axis)

        return tuple(normalize_axis(a, ndim) for a in axis)

    raise ValueError("axis %s not understood" % axis)


def equivalent(x, y):
    """
    Checks the equivalence of two scalars or arrays with broadcasting. Assumes
    a consistent dtype.

    Parameters
    ----------
    x : scalar or numpy.ndarray
    y : scalar or numpy.ndarray

    Returns
    -------
    equivalent : scalar or numpy.ndarray
        The element-wise comparison of where two arrays are equivalent.

    Examples
    --------
    >>> equivalent(1, 1)
    True
    >>> equivalent(np.nan, np.nan + 1)
    True
    >>> equivalent(1, 2)
    False
    >>> equivalent(np.inf, np.inf)
    True
    >>> equivalent(np.PZERO, np.NZERO)
    True
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # Can't contain NaNs
    if any(np.issubdtype(x.dtype, t) for t in
           [np.integer, np.bool_, np.character]):
        return x == y

    # Can contain NaNs
    # FIXME: Complex floats and np.void with multiple values can't be compared properly.
    return (x == y) | ((x != x) & (y != y))


def check_zero_fill_value(*args):
    """
    Checks if all the arguments have zero fill-values.

    Parameters
    ----------
    args : Iterable[SparseArray]

    Raises
    ------
    ValueError
        If all arguments don't have zero fill-values.

    Examples
    --------
    >>> import sparse
    >>> s1 = sparse.random((10,), density=0.5)
    >>> s2 = sparse.random((10,), density=0.5, fill_value=0.5)
    >>> check_zero_fill_value(s1)
    >>> check_zero_fill_value(s2)
    Traceback (most recent call last):
        ...
    ValueError: This operation requires zero fill values, but argument 0 had a fill value of 0.5.
    >>> check_zero_fill_value(s1, s2)
    Traceback (most recent call last):
        ...
    ValueError: This operation requires zero fill values, but argument 1 had a fill value of 0.5.
    """
    for i, arg in enumerate(args):
        if (hasattr(arg, 'fill_value') and
                not equivalent(arg.fill_value, _zero_of_dtype(arg.dtype))):
            raise ValueError('This operation requires zero fill values, '
                             'but argument {:d} had a fill value of {!s}.'.format(i, arg.fill_value))


def check_consistent_fill_value(arrays):
    """
    Checks if all the arguments have consistent fill-values.

    Parameters
    ----------
    args : Iterable[SparseArray]

    Raises
    ------
    ValueError
        If all elements of :code:`arrays` don't have the same fill-value.

    Examples
    --------
    >>> import sparse
    >>> s1 = sparse.random((10,), density=0.5, fill_value=0.1)
    >>> s2 = sparse.random((10,), density=0.5, fill_value=0.5)
    >>> check_consistent_fill_value([s1, s1])
    >>> check_consistent_fill_value([s1, s2])  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
        ...
    ValueError: This operation requires consistent fill-values, but argument 1 had a fill value of 0.5,\
        which is different from a fill_value of 0.1 in the first argument.
    """
    arrays = list(arrays)
    from .sparse_array import SparseArray

    if not all(isinstance(s, SparseArray) for s in arrays):
        raise ValueError('All arrays must be instances of SparseArray.')
    if len(arrays) == 0:
        raise ValueError('At least one array required.')

    fv = arrays[0].fill_value

    for i, arg in enumerate(arrays):
        if not equivalent(fv, arg.fill_value):
            raise ValueError('This operation requires consistent fill-values, '
                             'but argument {:d} had a fill value of {!s}, which '
                             'is different from a fill_value of {!s} in the first '
                             'argument.'.format(i, arg.fill_value, fv))
