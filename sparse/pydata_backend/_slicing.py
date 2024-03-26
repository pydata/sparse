# Most of this file is taken from https://github.com/dask/dask/blob/main/dask/array/slicing.py
# See license at https://github.com/dask/dask/blob/main/LICENSE.txt

import math
from collections.abc import Iterable
from numbers import Integral, Number

import numpy as np


def normalize_index(idx, shape):
    """Normalize slicing indexes
    1.  Replaces ellipses with many full slices
    2.  Adds full slices to end of index
    3.  Checks bounding conditions
    4.  Replaces numpy arrays with lists
    5.  Posify's slices integers and lists
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
    (slice(7, 10, 1),)
    >>> normalize_index((Ellipsis, None), (10,))
    (slice(0, 10, 1), None)
    """
    if not isinstance(idx, tuple):
        idx = (idx,)
    idx = replace_ellipsis(len(shape), idx)
    n_sliced_dims = 0
    for i in idx:
        if hasattr(i, "ndim") and i.ndim >= 1:
            n_sliced_dims += i.ndim
        elif i is None:
            continue
        else:
            n_sliced_dims += 1
    idx += (slice(None),) * (len(shape) - n_sliced_dims)
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

    for i, d in zip(idx, none_shape, strict=True):
        if d is not None:
            check_index(i, d)
    idx = tuple(map(sanitize_index, idx))
    idx = tuple(map(replace_none, idx, none_shape))
    idx = posify_index(none_shape, idx)
    return tuple(map(clip_slice, idx, none_shape))


def replace_ellipsis(n, index):
    """Replace ... with slices, :, : ,:
    >>> replace_ellipsis(4, (3, Ellipsis, 2))
    (3, slice(None, None, None), slice(None, None, None), 2)
    >>> replace_ellipsis(2, (Ellipsis, None))
    (slice(None, None, None), slice(None, None, None), None)
    """
    # Careful about using in or index because index may contain arrays
    isellipsis = [i for i, ind in enumerate(index) if ind is Ellipsis]
    if not isellipsis:
        return index
    if len(isellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    loc = isellipsis[0]
    extra_dimensions = n - (len(index) - sum(i is None for i in index) - 1)
    return index[:loc] + (slice(None, None, None),) * extra_dimensions + index[loc + 1 :]


def check_index(ind, dimension):
    """Check validity of index for a given dimension
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
    IndexError: Index out of bounds for dimension 5
    >>> check_index(slice(0, 3), 5)
    """
    # unknown dimension, assumed to be in bounds
    if isinstance(ind, Iterable):
        x = np.asanyarray(ind)
        if np.issubdtype(x.dtype, np.integer) and ((x >= dimension) | (x < -dimension)).any():
            raise IndexError(f"Index out of bounds for dimension {dimension:d}")
        if x.dtype == np.bool_ and len(x) != dimension:
            raise IndexError(
                f"boolean index did not match indexed array; dimension is {dimension:d} "
                f"but corresponding boolean dimension is {len(x):d}"
            )
    elif isinstance(ind, slice):
        return
    elif not isinstance(ind, Integral):
        raise IndexError(
            "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and "
            "integer or boolean arrays are valid indices"
        )

    elif ind >= dimension:
        raise IndexError(f"Index is not smaller than dimension {ind:d} >= {dimension:d}")

    elif ind < -dimension:
        msg = "Negative index is not greater than negative dimension {:d} <= -{:d}"
        raise IndexError(msg.format(ind, dimension))


def sanitize_index(ind):
    """Sanitize the elements for indexing along one axis
    >>> sanitize_index([2, 3, 5])
    array([2, 3, 5])
    >>> sanitize_index([True, False, True, False])
    array([0, 2])
    >>> sanitize_index(np.array([1, 2, 3]))
    array([1, 2, 3])
    >>> sanitize_index(np.array([False, True, True]))
    array([1, 2])
    >>> type(sanitize_index(np.int32(0)))  # doctest: +SKIP
    <type 'int'>
    >>> sanitize_index(0.5)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    IndexError: only integers, slices (`:`), ellipsis (`...`),
    numpy.newaxis (`None`) and integer or boolean arrays are valid indices
    """
    if ind is None:
        return None
    if isinstance(ind, slice):
        return slice(
            _sanitize_index_element(ind.start),
            _sanitize_index_element(ind.stop),
            _sanitize_index_element(ind.step),
        )
    if isinstance(ind, Number):
        return _sanitize_index_element(ind)
    if not hasattr(ind, "dtype") and len(ind) == 0:
        ind = np.array([], dtype=np.intp)
    ind = np.asarray(ind)
    if ind.dtype == np.bool_:
        nonzero = np.nonzero(ind)
        if len(nonzero) == 1:
            # If a 1-element tuple, unwrap the element
            nonzero = nonzero[0]
        return np.asanyarray(nonzero)
    if np.issubdtype(ind.dtype, np.integer):
        return ind

    raise IndexError(
        "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and "
        "integer or boolean arrays are valid indices"
    )


def _sanitize_index_element(ind):
    """Sanitize a one-element index."""
    if ind is None:
        return None

    return int(ind)


def posify_index(shape, ind):
    """Flip negative indices around to positive ones
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

        return ind
    if isinstance(ind, np.ndarray | list) and not math.isnan(shape):
        ind = np.asanyarray(ind)
        return np.where(ind < 0, ind + shape, ind)
    if isinstance(ind, slice):
        start, stop, step = ind.start, ind.stop, ind.step

        if start < 0:
            start += shape

        if not (0 > stop >= step) and stop < 0:
            stop += shape

        return slice(start, stop, ind.step)

    return ind


def clip_slice(idx, dim):
    """
    Clip slice to its effective size given the shape.

    Parameters
    ----------
    idx : The index.
    dim : The size along the corresponding dimension.

    Returns
    -------
    idx : slice

    Examples
    --------
    >>> clip_slice(slice(0, 20, 1), 10)
    slice(0, 10, 1)
    """
    if not isinstance(idx, slice):
        return idx

    start, stop, step = idx.start, idx.stop, idx.step

    if step > 0:
        start = max(start, 0)
        stop = min(stop, dim)

        if start > stop:
            start = stop
    else:
        start = min(start, dim - 1)
        stop = max(stop, -1)

        if start < stop:
            start = stop

    return slice(start, stop, step)


def replace_none(idx, dim):
    """
    Normalize slices to canonical form, i.e.
    replace ``None`` with the appropriate integers.

    Parameters
    ----------
    idx : slice or other index
    dim : dimension length

    Examples
    --------
    >>> replace_none(slice(None, None, None), 10)
    slice(0, 10, 1)
    """
    if not isinstance(idx, slice):
        return idx

    start, stop, step = idx.start, idx.stop, idx.step

    if step is None:
        step = 1

    if step > 0:
        if start is None:
            start = 0

        if stop is None:
            stop = dim
    else:
        if start is None:
            start = dim - 1

        if stop is None:
            stop = -1

    return slice(start, stop, step)
