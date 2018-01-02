# Most of this file is taken from https://github.com/dask/dask/blob/master/dask/array/slicing.py
# See license at https://github.com/dask/dask/blob/master/LICENSE.txt

import math
from numbers import Integral, Number
import numpy as np


def normalize_index(idx, shape):
    """ Normalize slicing indexes
    1.  Replaces ellipses with many full slices
    2.  Adds full slices to end of index
    3.  Checks bounding conditions
    4.  Replaces numpy arrays with lists
    5.  Posify's integers and lists
    6.  Normalizes slices to canonical form
    Examples
    --------
    >>> normalize_index(1, (10,))
    (1,)
    >>> normalize_index(-1, (10,))
    (9,)
    >>> normalize_index([-1], (10,))
    (array([9]),)
    >>> normalize_index(slice(-3, 10, 1), (10,))
    (slice(7, None, None),)
    >>> normalize_index((Ellipsis, None), (10,))
    (slice(None, None, None), None)
    """
    if not isinstance(idx, tuple):
        idx = (idx,)
    idx = replace_ellipsis(len(shape), idx)
    n_sliced_dims = 0
    for i in idx:
        if hasattr(i, 'ndim') and i.ndim >= 1:
            n_sliced_dims += i.ndim
        elif i is None:
            continue
        else:
            n_sliced_dims += 1
    idx = idx + (slice(None),) * (len(shape) - n_sliced_dims)
    if len([i for i in idx if i is not None]) > len(shape):
        raise IndexError("Too many indices for array")

    none_shape = []
    i = 0
    for ind in idx:
        if ind is not None:
            none_shape.append(shape[i])
            i += 1
        else:
            none_shape.append(None)

    for i, d in zip(idx, none_shape):
        if d is not None:
            check_index(i, d)
    idx = tuple(map(sanitize_index, idx))
    idx = tuple(map(normalize_slice, idx, none_shape))
    idx = posify_index(none_shape, idx)
    return idx


def replace_ellipsis(n, index):
    """ Replace ... with slices, :, : ,:
    >>> replace_ellipsis(4, (3, Ellipsis, 2))
    (3, slice(None, None, None), slice(None, None, None), 2)
    >>> replace_ellipsis(2, (Ellipsis, None))
    (slice(None, None, None), slice(None, None, None), None)
    """
    # Careful about using in or index because index may contain arrays
    isellipsis = [i for i, ind in enumerate(index) if ind is Ellipsis]
    if not isellipsis:
        return index
    elif len(isellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    else:
        loc = isellipsis[0]
    extra_dimensions = n - (len(index) - sum(i is None for i in index) - 1)
    return index[:loc] + (slice(None, None, None),) * extra_dimensions + index[loc + 1:]


def check_index(ind, dimension):
    """ Check validity of index for a given dimension
    Examples
    --------
    >>> check_index(3, 5)
    >>> check_index(5, 5)
    Traceback (most recent call last):
    ...
    IndexError: Index is not smaller than dimension 5 >= 5
    >>> check_index(6, 5)
    Traceback (most recent call last):
    ...
    IndexError: Index is not smaller than dimension 6 >= 5
    >>> check_index(-1, 5)
    >>> check_index(-6, 5)
    Traceback (most recent call last):
    ...
    IndexError: Negative index is not greater than negative dimension -6 <= -5
    >>> check_index([1, 2], 5)
    >>> check_index([6, 3], 5)
    Traceback (most recent call last):
    ...
    IndexError: Index out of bounds 5
    >>> check_index(slice(0, 3), 5)
    """
    # unknown dimension, assumed to be in bounds
    if np.isnan(dimension):
        return
    elif isinstance(ind, (list, np.ndarray)):
        x = np.asanyarray(ind)
        if np.issubdtype(x.dtype, np.integer) and \
                ((x >= dimension).any() or (x < -dimension).any()):
            raise IndexError("Index out of bounds %s" % dimension)
        elif x.dtype == bool and len(x) != dimension:
            raise IndexError("boolean index did not match indexed array; dimension is %s "
                             "but corresponding boolean dimension is %s", (dimension, len(x)))
    elif isinstance(ind, slice):
        return
    elif ind is None:
        return

    elif ind >= dimension:
        raise IndexError("Index is not smaller than dimension %d >= %d" %
                         (ind, dimension))

    elif ind < -dimension:
        msg = "Negative index is not greater than negative dimension %d <= -%d"
        raise IndexError(msg % (ind, dimension))


def sanitize_index(ind):
    """ Sanitize the elements for indexing along one axis
    >>> sanitize_index([2, 3, 5])
    array([2, 3, 5])
    >>> sanitize_index([True, False, True, False])
    array([0, 2])
    >>> sanitize_index(np.array([1, 2, 3]))
    array([1, 2, 3])
    >>> sanitize_index(np.array([False, True, True]))
    array([1, 2])
    >>> type(sanitize_index(np.int32(0))) # doctest: +SKIP
    <type 'int'>
    >>> sanitize_index(1.0)
    1
    >>> sanitize_index(0.5)
    Traceback (most recent call last):
    ...
    IndexError: Bad index.  Must be integer-like: 0.5
    """
    if ind is None:
        return None
    elif isinstance(ind, slice):
        return slice(_sanitize_index_element(ind.start),
                     _sanitize_index_element(ind.stop),
                     _sanitize_index_element(ind.step))
    elif isinstance(ind, Number):
        return _sanitize_index_element(ind)
    index_array = np.asanyarray(ind)
    if index_array.dtype == bool:
        nonzero = np.nonzero(index_array)
        if len(nonzero) == 1:
            # If a 1-element tuple, unwrap the element
            nonzero = nonzero[0]
        return np.asanyarray(nonzero)
    elif np.issubdtype(index_array.dtype, np.integer):
        return index_array
    elif np.issubdtype(index_array.dtype, float):
        int_index = index_array.astype(np.intp)
        if np.allclose(index_array, int_index):
            return int_index
        else:
            check_int = np.isclose(index_array, int_index)
            first_err = index_array.ravel(
            )[np.flatnonzero(~check_int)[0]]
            raise IndexError("Bad index.  Must be integer-like: %s" %
                             first_err)
    else:
        raise TypeError("Invalid index type", type(ind), ind)


def _sanitize_index_element(ind):
    """Sanitize a one-element index."""
    if isinstance(ind, Number):
        ind2 = int(ind)
        if ind2 != ind:
            raise IndexError("Bad index.  Must be integer-like: %s" % ind)
        else:
            return ind2
    elif ind is None:
        return None
    else:
        raise TypeError("Invalid index type", type(ind), ind)


def normalize_slice(idx, dim):
    """ Normalize slices to canonical form
    Parameters
    ----------
    idx: slice or other index
    dim: dimension length
    Examples
    --------
    >>> normalize_slice(slice(0, 10, 1), 10)
    slice(None, None, None)
    """

    if isinstance(idx, slice):
        start, stop, step = idx.start, idx.stop, idx.step
        if start is not None:
            if start < 0 and not math.isnan(dim):
                start = max(0, start + dim)
            elif start > dim:
                start = dim
        if stop is not None:
            if stop < 0 and not math.isnan(dim):
                stop = max(0, stop + dim)
            elif stop > dim:
                stop = dim

        step = 1 if step is None else step

        if step > 0:
            if start == 0:
                start = None
            if stop == dim:
                stop = None
        else:
            if start == dim - 1:
                start = None
            if stop == -1:
                stop = None

        if step == 1:
            step = None
        return slice(start, stop, step)
    return idx


def posify_index(shape, ind):
    """ Flip negative indices around to positive ones
    >>> posify_index(10, 3)
    3
    >>> posify_index(10, -3)
    7
    >>> posify_index(10, [3, -3])
    array([3, 7])
    >>> posify_index((10, 20), (3, -3))
    (3, 17)
    >>> posify_index((10, 20), (3, [3, 4, -3]))  # doctest: +NORMALIZE_WHITESPACE
    (3, array([ 3,  4, 17]))
    """
    if isinstance(ind, tuple):
        return tuple(map(posify_index, shape, ind))
    if isinstance(ind, Integral):
        if ind < 0 and not math.isnan(shape):
            return ind + shape
        else:
            return ind
    if isinstance(ind, (np.ndarray, list)) and not math.isnan(shape):
        ind = np.asanyarray(ind)
        return np.where(ind < 0, ind + shape, ind)

    return ind
