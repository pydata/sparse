from collections import Iterable
from numbers import Integral

import numba
import numpy as np

from ..compatibility import range, zip_longest
from ..slicing import normalize_index
from ..utils import _zero_of_dtype


def getitem(x, index):
    """
    This function implements the indexing functionality for COO.

    The overall algorithm has three steps:

    1. Normalize the index to canonical form. Function: normalize_index
    2. Get the mask, which is a list of integers corresponding to
       the indices in coords/data for the output data. Function: _mask
    3. Transform the coordinates to what they will be in the output.

    Parameters
    ----------
    x : COO
        The array to apply the indexing operation on.
    index : {tuple, str}
        The index into the array.
    """
    from .core import COO

    # If string, this is an index into an np.void
    # Custom dtype.
    if isinstance(index, str):
        data = x.data[index]
        idx = np.where(data)
        coords = list(x.coords[:, idx[0]])
        coords.extend(idx[1:])

        return COO(coords, data[idx].flatten(),
                   shape=x.shape + x.data.dtype[index].shape,
                   has_duplicates=False,
                   sorted=True)

    # Otherwise, convert into a tuple.
    if not isinstance(index, tuple):
        index = (index,)

    # Check if the last index is an ellipsis.
    last_ellipsis = len(index) > 0 and index[-1] is Ellipsis

    # Normalize the index into canonical form.
    index = normalize_index(index, x.shape)

    # zip_longest so things like x[..., None] are picked up.
    if len(index) != 0 and all(ind == slice(0, dim, 1) for ind, dim in zip_longest(index, x.shape)):
        return x

    # Get the mask
    mask = _mask(x.coords, index, x.shape)

    # Get the length of the mask
    if isinstance(mask, slice):
        n = len(range(mask.start, mask.stop, mask.step))
    else:
        n = len(mask)

    coords = []
    shape = []
    i = 0
    for ind in index:
        # Nothing is added to shape or coords if the index is an integer.
        if isinstance(ind, Integral):
            i += 1
            continue
        # Add to the shape and transform the coords in the case of a slice.
        elif isinstance(ind, slice):
            shape.append(len(range(ind.start, ind.stop, ind.step)))
            dt = np.min_scalar_type(min(-(dim - 1) if dim != 0 else -1 for dim in shape))
            coords.append((x.coords[i, mask].astype(dt) - ind.start) // ind.step)
            i += 1
        elif isinstance(ind, Iterable):
            raise NotImplementedError('Advanced indexing is not yet supported.')
        # Add a dimension for None.
        elif ind is None:
            coords.append(np.zeros(n))
            shape.append(1)

    # Join all the transformed coords.
    if coords:
        coords = np.stack(coords, axis=0)
    else:
        # If index result is a scalar, return a 0-d COO or
        # a scalar depending on whether the last index is an ellipsis.
        if last_ellipsis:
            coords = np.empty((0, n), dtype=np.uint8)
        else:
            if n != 0:
                return x.data[mask][0]
            else:
                return _zero_of_dtype(x.dtype)[()]

    shape = tuple(shape)
    data = x.data[mask]

    return COO(coords, data, shape=shape,
               has_duplicates=False,
               sorted=True)


def _mask(coords, indices, shape):
    indices = _prune_indices(indices, shape)

    ind_ar = np.empty((len(indices), 3), dtype=np.intp)

    for i, idx in enumerate(indices):
        if isinstance(idx, slice):
            ind_ar[i] = [idx.start, idx.stop, idx.step]
        else:  # idx is an integer
            ind_ar[i] = [idx, idx + 1, 1]

    mask, is_slice = _compute_mask(coords, ind_ar)

    if is_slice:
        return slice(mask[0], mask[1], 1)
    else:
        return mask


def _prune_indices(indices, shape, prune_none=True):
    """
    Gets rid of the indices that do not contribute to the
    overall mask, e.g. None and full slices.

    Parameters
    ----------
    indices : tuple
        The indices to the array.
    shape : tuple[int]
        The shape of the array.

    Returns
    -------
    indices : tuple
        The filtered indices.

    Examples
    --------
    >>> _prune_indices((None, 5), (10,)) # None won't affect the mask
    [5]
    >>> _prune_indices((slice(0, 10, 1),), (10,)) # Full slices don't affect the mask
    []
    """
    if prune_none:
        indices = [idx for idx in indices if idx is not None]

    i = 0
    for idx, l in zip(indices[::-1], shape[::-1]):
        if not isinstance(idx, slice):
            break

        if idx.start == 0 and idx.stop == l and idx.step == 1:
            i += 1
            continue

        if idx.start == l - 1 and idx.stop == -1 and idx.step == -1:
            i += 1
            continue

        break
    if i != 0:
        indices = indices[:-i]
    return indices


@numba.jit(nopython=True)
def _compute_mask(coords, indices):  # pragma: no cover
    """
    Gets the mask for the coords given the indices in slice format.

    Works with either start-stop ranges of matching indices into coords
    called "pairs" (start-stop pairs) or filters the mask directly, based
    on which is faster.

    Exploits the structure in sorted coords, which is that for a constant
    value of coords[i - 1], coords[i - 2] and so on, coords[i] is sorted.
    Concretely, ``coords[i, coords[i - 1] == v1 & coords[i - 2] = v2, ...]``
    is always sorted. It uses this sortedness to find sub-pairs for each
    dimension given the previous, and so on. This is efficient for small
    slices or ints, but not for large ones.

    After it detects that working with pairs is rather inefficient (or after
    going through each possible index), it constructs a filtered mask from the
    start-stop pairs.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates of the array.
    indices : np.ndarray
        The indices in the form of slices such that indices[:, 0] are starts,
        indices[:, 1] are stops and indices[:, 2] are steps.

    Returns
    -------
    mask : np.ndarray
        The starts and stops in the mask.
    is_slice : bool
        Whether or not the array represents a continuous slice.

    Examples
    --------
    Let's create some mock coords and indices

    >>> import numpy as np
    >>> coords = np.array([[0, 0, 1, 1, 2, 2]])
    >>> indices = np.array([[0, 3, 2]])  # Equivalent to slice(0, 3, 2)

    Now let's get the mask. Notice that the indices of ``0`` and ``2`` are matched.

    >>> _compute_mask(coords, indices)
    (array([0, 1, 4, 5]), False)

    Now, let's try with a more "continuous" slice. Matches ``0`` and ``1``.

    >>> indices = np.array([[0, 2, 1]])
    >>> _compute_mask(coords, indices)
    (array([0, 4]), True)

    This is equivalent to mask being ``slice(0, 4, 1)``.
    """
    # Set the initial mask to be the entire range of coordinates.
    starts = [0]
    stops = [coords.shape[1]]
    n_matches = coords.shape[1]

    i = 0
    while i < len(indices):
        # Guesstimate whether working with pairs is more efficient or
        # working with the mask directly.
        # One side is the estimate of time taken for binary searches
        # (n_searches * log(avg_length))
        # The other is an estimated time of a linear filter for the mask.
        n_pairs = len(starts)
        n_current_slices = _get_slice_len(indices[i]) * n_pairs + 2
        if n_current_slices * np.log(n_current_slices / max(n_pairs, 1)) > \
                n_matches + n_pairs:
            break

        # For each of the pairs, search inside the coordinates for other
        # matching sub-pairs.
        # This gets the start-end coordinates in coords for each 'sub-array'
        # Which would come out of indexing a single integer.
        starts, stops, n_matches = _get_mask_pairs(starts, stops, coords[i], indices[i])

        i += 1

    # Combine adjacent pairs
    starts, stops = _join_adjacent_pairs(starts, stops)

    # If just one pair is left over, treat it as a slice.
    if i == len(indices) and len(starts) == 1:
        return np.array([starts[0], stops[0]]), True

    # Convert start-stop pairs into mask, filtering by remaining
    # coordinates.
    mask = _filter_pairs(starts, stops, coords[i:], indices[i:])

    return np.array(mask, dtype=np.intp), False


@numba.jit(nopython=True)
def _get_mask_pairs(starts_old, stops_old, c, idx):  # pragma: no cover
    """
    Gets the pairs for a following dimension given the pairs for
    a dimension.

    For each pair, it searches in the following dimension for
    matching coords and returns those.

    The total combined length of all pairs is returned to
    help with the performance guesstimate.

    Parameters
    ----------
    starts_old, stops_old : list[int]
        The starts and stops from the previous index.
    c : np.ndarray
        The coords for this index's dimension.
    idx : np.ndarray
        The index in the form of a slice.
        idx[0], idx[1], idx[2] = start, stop, step

    Returns
    -------
    starts, stops: list
        The starts and stops after applying the current index.
    n_matches : int
        The sum of elements in all ranges.

    Examples
    --------
    >>> c = np.array([1, 2, 1, 2, 1, 1, 2, 2])
    >>> starts_old = [4]
    >>> stops_old = [8]
    >>> idx = np.array([1, 2, 1])
    >>> _get_mask_pairs(starts_old, stops_old, c, idx)
    ([4], [6], 2)
    """
    starts = []
    stops = []
    n_matches = 0

    for j in range(len(starts_old)):
        # For each matching "integer" in the slice, search within the "sub-coords"
        # Using binary search.
        for p_match in range(idx[0], idx[1], idx[2]):
            start = np.searchsorted(c[starts_old[j]:stops_old[j]], p_match) + starts_old[j]
            stop = np.searchsorted(c[starts_old[j]:stops_old[j]], p_match + 1) + starts_old[j]

            if start != stop:
                starts.append(start)
                stops.append(stop)
                n_matches += stop - start

    return starts, stops, n_matches


@numba.jit(nopython=True)
def _get_slice_len(idx):  # pragma: no cover
    """
    Get the number of elements in a slice.

    Parameters
    ----------
    idx : np.ndarray
        A (3,) shaped array containing start, stop, step

    Returns
    -------
    n : int
        The length of the slice.

    Examples
    --------
    >>> idx = np.array([5, 15, 5])
    >>> _get_slice_len(idx)
    2
    """
    start, stop, step = idx[0], idx[1], idx[2]

    if step > 0:
        return (stop - start + step - 1) // step
    else:
        return (start - stop - step - 1) // (-step)


@numba.jit(nopython=True)
def _filter_pairs(starts, stops, coords, indices):  # pragma: no cover
    """
    Converts all the pairs into a single integer mask, additionally filtering
    by the indices.

    Parameters
    ----------
    starts, stops : list[int]
        The starts and stops to convert into an array.
    coords : np.ndarray
        The coordinates to filter by.
    indices : np.ndarray
        The indices in the form of slices such that indices[:, 0] are starts,
        indices[:, 1] are stops and indices[:, 2] are steps.

    Returns
    -------
    mask : list
        The output integer mask.

    Examples
    --------
    >>> import numpy as np
    >>> starts = [2]
    >>> stops = [7]
    >>> coords = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    >>> indices = np.array([[2, 8, 2]]) # Start, stop, step pairs
    >>> _filter_pairs(starts, stops, coords, indices)
    [2, 4, 6]
    """
    mask = []

    # For each pair,
    for i in range(len(starts)):
        # For each element match within the pair range
        for j in range(starts[i], stops[i]):
            match = True

            # Check if it matches all indices
            for k in range(len(indices)):
                idx = indices[k]
                elem = coords[k, j]

                match &= ((elem - idx[0]) % idx[2] == 0 and
                          ((idx[2] > 0 and idx[0] <= elem < idx[1])
                           or (idx[2] < 0 and idx[0] >= elem > idx[1])))

            # and append to the mask if so.
            if match:
                mask.append(j)

    return mask


@numba.jit(nopython=True)
def _join_adjacent_pairs(starts_old, stops_old):  # pragma: no cover
    """
    Joins adjacent pairs into one. For example, 2-5 and 5-7
    will reduce to 2-7 (a single pair). This may help in
    returning a slice in the end which could be faster.

    Parameters
    ----------
    starts_old, stops_old : list[int]
        The input starts and stops

    Returns
    -------
    starts, stops : list[int]
        The reduced starts and stops.

    Examples
    --------
    >>> starts = [2, 5]
    >>> stops = [5, 7]
    >>> _join_adjacent_pairs(starts, stops)
    ([2], [7])
    """
    if len(starts_old) <= 1:
        return starts_old, stops_old

    starts = [starts_old[0]]
    stops = []

    for i in range(1, len(starts_old)):
        if starts_old[i] != stops_old[i - 1]:
            starts.append(starts_old[i])
            stops.append(stops_old[i - 1])

    stops.append(stops_old[-1])

    return starts, stops
