import functools
from collections.abc import Iterable
from numbers import Integral
from functools import reduce

import operator
import numpy as np


def assert_eq(x, y, check_nnz=True, compare_dtype=True, **kwargs):
    from ._coo import COO

    assert x.shape == y.shape

    if compare_dtype:
        assert x.dtype == y.dtype

    check_equal = (
        np.array_equal
        if np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.integer)
        else functools.partial(np.allclose, equal_nan=True)
    )

    if isinstance(x, COO):
        assert is_canonical(x)
    if isinstance(y, COO):
        assert is_canonical(y)

    if isinstance(x, COO) and isinstance(y, COO) and check_nnz:
        assert np.array_equal(x.coords, y.coords)
        assert check_equal(x.data, y.data, **kwargs)
        assert x.fill_value == y.fill_value
        return

    if hasattr(x, "todense"):
        xx = x.todense()
        if check_nnz:
            assert_nnz(x, xx)
    else:
        xx = x
    if hasattr(y, "todense"):
        yy = y.todense()
        if check_nnz:
            assert_nnz(y, yy)
    else:
        yy = y
    assert check_equal(xx, yy, **kwargs)


def assert_nnz(s, x):
    fill_value = s.fill_value if hasattr(s, "fill_value") else _zero_of_dtype(s.dtype)
    assert np.sum(~equivalent(x, fill_value)) == s.nnz


def is_canonical(x):
    return not x.shape or (
        (np.diff(x.linear_loc()) > 0).all()
        and not equivalent(x.data, x.fill_value).any()
    )


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
    density=None,
    nnz=None,
    random_state=None,
    data_rvs=None,
    format="coo",
    fill_value=None,
    idx_dtype=None,
    **kwargs,
):
    """Generate a random sparse multidimensional array

    Parameters
    ----------
    shape : Tuple[int]
        Shape of the array
    density : float, optional
        Density of the generated array; default is 0.01.
        Mutually exclusive with `nnz`.
    nnz : int, optional
        Number of nonzero elements in the generated array.
        Mutually exclusive with `density`.
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
    :obj:`scipy.sparse.rand` : Equivalent Scipy function.
    :obj:`numpy.random.rand` : Similar Numpy function.

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
    from ._coo import COO

    if density is not None and nnz is not None:
        raise ValueError("'density' and 'nnz' are mutually exclusive")

    if density is None:
        density = 0.01
    if not (0 <= density <= 1):
        raise ValueError("density {} is not in the unit interval".format(density))

    elements = np.prod(shape, dtype=np.intp)

    if nnz is None:
        nnz = int(elements * density)
    if not (0 <= nnz <= elements):
        raise ValueError(
            "cannot generate {} nonzero elements "
            "for an array with {} total elements".format(nnz, elements)
        )

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

    ar = COO(
        ind[None, :],
        data,
        shape=elements,
        fill_value=fill_value,
    ).reshape(shape)

    if idx_dtype:
        if can_store(idx_dtype, max(shape)):
            ar.coords = ar.coords.astype(idx_dtype)
        else:
            raise ValueError(
                "cannot cast array with shape {} to dtype {}.".format(shape, idx_dtype)
            )

    return ar.asformat(format, **kwargs)


def isscalar(x):
    from ._sparse_array import SparseArray

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
            raise ValueError("Invalid axis index %d for ndim=%d" % (axis, ndim))

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
    if any(np.issubdtype(x.dtype, t) for t in [np.integer, np.bool_, np.character]):
        return x == y

    # Can contain NaNs
    # FIXME: Complex floats and np.void with multiple values can't be compared properly.
    # lgtm [py/comparison-of-identical-expressions]
    return (x == y) | ((x != x) & (y != y))


# copied from zarr
# See https://github.com/zarr-developers/zarr-python/blob/master/zarr/util.py
def human_readable_size(size):
    if size < 2 ** 10:
        return "%s" % size
    elif size < 2 ** 20:
        return "%.1fK" % (size / float(2 ** 10))
    elif size < 2 ** 30:
        return "%.1fM" % (size / float(2 ** 20))
    elif size < 2 ** 40:
        return "%.1fG" % (size / float(2 ** 30))
    elif size < 2 ** 50:
        return "%.1fT" % (size / float(2 ** 40))
    else:
        return "%.1fP" % (size / float(2 ** 50))


def html_table(arr):
    table = "<table>"
    table += "<tbody>"
    headings = ["Format", "Data Type", "Shape", "nnz", "Density", "Read-only"]

    density = np.float_(arr.nnz) / np.float_(arr.size)

    info = [
        type(arr).__name__.lower(),
        str(arr.dtype),
        str(arr.shape),
        str(arr.nnz),
        str(density),
    ]

    # read-only
    info.append(str(not hasattr(arr, "__setitem__")))

    if hasattr(arr, "nbytes"):
        headings.append("Size")
        info.append(human_readable_size(arr.nbytes))
        headings.append("Storage ratio")
        info.append(
            "%.1f"
            % (
                np.float_(arr.nbytes)
                / np.float_(reduce(operator.mul, arr.shape, 1) * arr.dtype.itemsize)
            )
        )

    # compressed_axes
    if type(arr).__name__ == "GCXS":
        headings.append("Compressed Axes")
        info.append(str(arr.compressed_axes))

    for h, i in zip(headings, info):
        table += (
            "<tr>"
            '<th style="text-align: left">%s</th>'
            '<td style="text-align: left">%s</td>'
            "</tr>" % (h, i)
        )
    table += "</tbody>"
    table += "</table>"
    return table


def check_compressed_axes(ndim, compressed_axes):
    """
    Checks if the given compressed_axes are compatible with the shape of the array.

    Parameters
    ----------
    ndim : int
    compressed_axes : Iterable

    Raises
    ------
    ValueError
        If the compressed_axes are incompatible with the number of dimensions
    """
    if compressed_axes is None:
        return
    if isinstance(ndim, Iterable):
        ndim = len(ndim)
    if not isinstance(compressed_axes, Iterable):
        raise ValueError("compressed_axes must be an iterable")
    if len(compressed_axes) == ndim:
        raise ValueError("cannot compress all axes")
    if not np.array_equal(list(set(compressed_axes)), compressed_axes):
        raise ValueError("axes must be sorted without repeats")
    if not all(isinstance(a, Integral) for a in compressed_axes):
        raise ValueError("axes must be represented with integers")
    if min(compressed_axes) < 0 or max(compressed_axes) >= ndim:
        raise ValueError("axis out of range")


def check_zero_fill_value(*args):
    """
    Checks if all the arguments have zero fill-values.

    Parameters
    ----------
    *args : Iterable[SparseArray]

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
        if hasattr(arg, "fill_value") and not equivalent(
            arg.fill_value, _zero_of_dtype(arg.dtype)
        ):
            raise ValueError(
                "This operation requires zero fill values, "
                "but argument {:d} had a fill value of {!s}.".format(i, arg.fill_value)
            )


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
    from ._sparse_array import SparseArray

    if not all(isinstance(s, SparseArray) for s in arrays):
        raise ValueError("All arrays must be instances of SparseArray.")
    if len(arrays) == 0:
        raise ValueError("At least one array required.")

    fv = arrays[0].fill_value

    for i, arg in enumerate(arrays):
        if not equivalent(fv, arg.fill_value):
            raise ValueError(
                "This operation requires consistent fill-values, "
                "but argument {:d} had a fill value of {!s}, which "
                "is different from a fill_value of {!s} in the first "
                "argument.".format(i, arg.fill_value, fv)
            )


def get_out_dtype(arr, scalar):
    out_type = arr.dtype
    if not can_store(out_type, scalar):
        out_type = np.min_scalar_type(scalar)
    return out_type


def can_store(dtype, scalar):
    return np.array(scalar, dtype=dtype) == np.array(scalar)


def is_unsigned_dtype(dtype):
    return not np.array(-1, dtype=dtype) == np.array(-1)


def convert_format(format):
    from ._sparse_array import SparseArray

    if isinstance(format, type):
        if not issubclass(format, SparseArray):
            raise ValueError(f"Invalid format: {format}")
        return format.__name__.lower()
    if isinstance(format, str):
        return format

    raise ValueError(f"Invalid format: {format}")
