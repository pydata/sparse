from itertools import product

import numpy as np
import scipy.sparse

import numba

from ..utils import isscalar, PositinalArgumentPartial, _zero_of_dtype
from ..compatibility import range, zip, zip_longest


def elemwise(func, *args, **kwargs):
    """
    Apply a function to any number of arguments.

    Parameters
    ----------
    func : Callable
        The function to apply. Must support broadcasting.
    args : tuple, optional
        The arguments to the function. Can be :obj:`SparseArray` objects
        or :obj:`scipy.sparse.spmatrix` objects.
    kwargs : dict, optional
        Any additional arguments to pass to the function.

    Returns
    -------
    COO
        The result of applying the function.

    Raises
    ------
    ValueError
        If the operation would result in a dense matrix, or if the operands
        don't have broadcastable shapes.

    See Also
    --------
    :obj:`numpy.ufunc` : A similar Numpy construct. Note that any :code:`ufunc` can be used
        as the :code:`func` input to this function.

    Notes
    -----
    Previously, operations with Numpy arrays were sometimes supported. Now,
    it is necessary to convert Numpy arrays to :obj:`COO` objects.
    """
    # Because we need to mutate args.
    from .core import COO
    from ..sparse_array import SparseArray

    args = list(args)
    posargs = []
    pos = []
    for i, arg in enumerate(args):
        if isinstance(arg, scipy.sparse.spmatrix):
            args[i] = COO.from_scipy_sparse(arg)
        elif isscalar(arg) or (isinstance(arg, np.ndarray)
                               and not arg.shape):
            # Faster and more reliable to pass ()-shaped ndarrays as scalars.
            args[i] = np.asarray(arg)[()]

            pos.append(i)
            posargs.append(args[i])
        elif isinstance(arg, SparseArray) and not isinstance(arg, COO):
            args[i] = COO(arg)
        elif not isinstance(arg, COO):
            return NotImplemented

    # Filter out scalars as they are 'baked' into the function.
    func = PositinalArgumentPartial(func, pos, posargs)
    args = [arg for arg in args if not isscalar(arg)]

    if len(args) == 0:
        return func(**kwargs)

    return _elemwise_n_ary(func, *args, **kwargs)


@numba.jit(nopython=True)
def _match_arrays(a, b):    # pragma: no cover
    """
    Finds all indexes into a and b such that a[i] = b[j]. The outputs are sorted
    in lexographical order.

    Parameters
    ----------
    a, b : np.ndarray
        The input 1-D arrays to match. If matching of multiple fields is
        needed, use np.recarrays. These two arrays must be sorted.

    Returns
    -------
    a_idx, b_idx : np.ndarray
        The output indices of every possible pair of matching elements.
    """
    if len(a) == 0 or len(b) == 0:
        return np.empty(0, dtype=np.uintp), np.empty(0, dtype=np.uintp)

    a_ind, b_ind = [], []
    nb = len(b)
    ib = 0
    match = 0

    for ia, j in enumerate(a):
        if j == b[match]:
            ib = match

        while ib < nb and j >= b[ib]:
            if j == b[ib]:
                a_ind.append(ia)
                b_ind.append(ib)

                if b[match] < b[ib]:
                    match = ib

            ib += 1

    return np.array(a_ind, dtype=np.uintp), np.array(b_ind, dtype=np.uintp)


def _elemwise_n_ary(func, *args, **kwargs):
    """
    Apply a function to any number of arguments with broadcasting.

    Parameters
    ----------
    func : Callable
        The function to apply to arguments. Must support broadcasting.
    args : list
        Input :obj:`COO` or :obj:`numpy.ndarray`s.
    kwargs : dict
        Additional arguments to pass to the function.

    Returns
    -------
    COO
        The output array.

    Raises
    ------
    ValueError
        If the input shapes aren't compatible or the result will be dense.
    """
    from .core import COO

    args = list(args)

    args_zeros = tuple(_zero_of_dtype(np.dtype(arg)) for arg in args)

    func_value = func(*args_zeros, **kwargs)
    func_zero = _zero_of_dtype(func_value.dtype)
    if func_value != func_zero:
        raise ValueError("Performing this operation would produce "
                         "a dense result: %s" % str(func))
    data_list = []
    coords_list = []

    cache = {}
    for mask in product([True, False], repeat=len(args)):
        if not any(mask):
            continue

        ci, di = _unmatch_coo(func, args, mask, cache, **kwargs)

        coords_list.extend(ci)
        data_list.extend(di)

    result_shape = _get_nary_broadcast_shape(*[arg.shape for arg in args])

    # Concatenate matches and mismatches
    data = np.concatenate(data_list) if len(data_list) else np.empty((0,), dtype=func_value.dtype)
    coords = np.concatenate(coords_list, axis=1) if len(coords_list) else \
        np.empty((0, len(result_shape)), dtype=np.min_scalar_type(max(result_shape) - 1))

    return COO(coords, data, shape=result_shape, has_duplicates=False)


def _match_coo(*args, **kwargs):
    """
    Matches the coordinates for any number of input :obj:`COO` arrays.
    Equivalent to "sparse" broadcasting for all arrays.

    Parameters
    ----------
    args : Tuple[COO]
        The input :obj:`COO` arrays.
    return_midx : bool
        Whether to return matched indices or matched arrays. Matching
        only supported for two arrays. ``False`` by default.
    cache : dict
        Cache of things already matched. No cache by default.

    Returns
    -------
    matched_idx : List[ndarray]
        The indices of matched elements in the original arrays. Only returned if
        ``return_midx`` is ``True``.
    matched_arrays : List[COO]
        The expanded, matched :obj:`COO` objects. Only returned if
        ``return_midx`` is ``False``.
    """
    from .core import COO
    from .common import linear_loc

    return_midx = kwargs.pop('return_midx', False)
    cache = kwargs.pop('cache', None)

    if kwargs:
        raise ValueError('Unknown kwargs %s' % kwargs.keys())

    if return_midx and (len(args) != 2 or cache is not None):
        raise NotImplementedError('Matching indices only supported for two args, and no cache.')

    matched_arrays = [args[0]]
    cache_key = [id(args[0])]
    for arg2 in args[1:]:
        cache_key.append(id(arg2))
        key = tuple(cache_key)
        if cache is not None and key in cache:
            matched_arrays = cache[key]
            continue

        cargs = [matched_arrays[0], arg2]
        current_shape = _get_broadcast_shape(matched_arrays[0].shape, arg2.shape)
        params = [_get_broadcast_parameters(arg.shape, current_shape) for arg in cargs]
        reduced_params = [all(p) for p in zip(*params)]
        reduced_shape = _get_reduced_shape(arg2.shape,
                                           reduced_params[-arg2.ndim:])

        reduced_coords = [_get_reduced_coords(arg.coords, reduced_params[-arg.ndim:])
                          for arg in cargs]

        linear = [linear_loc(rc, reduced_shape) for rc in reduced_coords]
        sorted_idx = [np.argsort(idx) for idx in linear]
        linear = [idx[s] for idx, s in zip(linear, sorted_idx)]
        matched_idx = _match_arrays(*linear)

        if return_midx:
            matched_idx = [sidx[midx] for sidx, midx in zip(sorted_idx, matched_idx)]
            return matched_idx

        coords = [arg.coords[:, s] for arg, s in zip(cargs, sorted_idx)]
        mcoords = [c[:, idx] for c, idx in zip(coords, matched_idx)]
        mcoords = _get_matching_coords(mcoords, params, current_shape)
        mdata = [arg.data[sorted_idx[0]][matched_idx[0]] for arg in matched_arrays]
        mdata.append(arg2.data[sorted_idx[1]][matched_idx[1]])
        # The coords aren't truly sorted, but we don't need them, so it's
        # best to avoid the extra cost.
        matched_arrays = [
            COO(mcoords, md, shape=current_shape, sorted=True, has_duplicates=False)
            for md in mdata]

        if cache is not None:
            cache[key] = matched_arrays

    return matched_arrays


def _unmatch_coo(func, args, mask, cache, **kwargs):
    """
    Matches the coordinates for any number of input :obj:`COO` arrays.

    First computes the matches, then filters out the non-matches.

    Parameters
    ----------
    func : Callable
        The function to compute matches
    args : tuple[COO]
        The input :obj:`COO` arrays.
    mask : tuple[bool]
        Specifies the inputs that are zero and the ones that are
        nonzero.
    kwargs: dict
        Extra keyword arguments to pass to func.

    Returns
    -------
    matched_coords : list[ndarray]
        The matched coordinates.
    matched_data : list[ndarray]
        The matched data.
    """
    from .core import COO

    matched_args = [a for a, m in zip(args, mask) if m]
    unmatched_args = [a for a, m in zip(args, mask) if not m]

    matched_arrays = _match_coo(*matched_args, cache=cache)

    pos = tuple(i for i, m in enumerate(mask) if not m)
    posargs = [_zero_of_dtype(arg.dtype) for arg, m in zip(args, mask) if not m]
    result_shape = _get_nary_broadcast_shape(*[arg.shape for arg in args])

    partial = PositinalArgumentPartial(func, pos, posargs)
    matched_func = partial(*[a.data for a in matched_arrays], **kwargs)

    unmatched_mask = matched_func != _zero_of_dtype(matched_func.dtype)

    if not unmatched_mask.any():
        return [], []

    func_data = matched_func[unmatched_mask]
    func_coords = matched_arrays[0].coords[:, unmatched_mask]

    # The coords aren't truly sorted, but we don't need them, so it's
    # best to avoid the extra cost.
    func_array = COO(func_coords, func_data, shape=matched_arrays[0].shape,
                     sorted=True, has_duplicates=False).broadcast_to(result_shape)

    if all(mask):
        return [func_array.coords], [func_array.data]

    unmatched_mask = np.ones(func_array.nnz, dtype=np.bool)

    for arg in unmatched_args:
        matched_idx = _match_coo(func_array, arg, return_midx=True)[0]
        unmatched_mask[matched_idx] = False

    coords = np.asarray(func_array.coords[:, unmatched_mask], order='C')
    data = np.asarray(func_array.data[unmatched_mask], order='C')

    return [coords], [data]


def _get_nary_broadcast_shape(*shapes):
    """
    Broadcast any number of shapes to a result shape.

    Parameters
    ----------
    shapes : tuple[tuple[int]]
        The shapes to broadcast.

    Returns
    -------
    tuple[int]
        The output shape.

    Raises
    ------
    ValueError
        If the input shapes cannot be broadcast to a single shape.
    """
    result_shape = ()

    for shape in shapes:
        try:
            result_shape = _get_broadcast_shape(shape, result_shape)
        except ValueError:
            shapes_str = ', '.join(str(shape) for shape in shapes)
            raise ValueError('operands could not be broadcast together with shapes %s'
                             % shapes_str)

    return result_shape


def _get_broadcast_shape(shape1, shape2, is_result=False):
    """
    Get the overall broadcasted shape.

    Parameters
    ----------
    shape1, shape2 : tuple[int]
        The input shapes to broadcast together.
    is_result : bool
        Whether or not shape2 is also the result shape.

    Returns
    -------
    result_shape : tuple[int]
        The overall shape of the result.

    Raises
    ------
    ValueError
        If the two shapes cannot be broadcast together.
    """
    # https://stackoverflow.com/a/47244284/774273
    if not all((l1 == l2) or (l1 == 1) or ((l2 == 1) and not is_result) for l1, l2 in
               zip(shape1[::-1], shape2[::-1])):
        raise ValueError('operands could not be broadcast together with shapes %s, %s' %
                         (shape1, shape2))

    result_shape = tuple(max(l1, l2) for l1, l2 in
                         zip_longest(shape1[::-1], shape2[::-1], fillvalue=1))[::-1]

    return result_shape


def _get_broadcast_parameters(shape, broadcast_shape):
    """
    Get the broadcast parameters.

    Parameters
    ----------
    shape : tuple[int]
        The input shape.
    broadcast_shape
        The shape to broadcast to.

    Returns
    -------
    params : list
        A list containing None if the dimension isn't in the original array, False if
        it needs to be broadcast, and True if it doesn't.
    """
    params = [None if l1 is None else l1 == l2 for l1, l2
              in zip_longest(shape[::-1], broadcast_shape[::-1], fillvalue=None)][::-1]

    return params


def _get_reduced_coords(coords, params):
    """
    Gets only those dimensions of the coordinates that don't need to be broadcast.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates to reduce.
    params : list
        The params from which to check which dimensions to get.

    Returns
    -------
    reduced_coords : np.ndarray
        The reduced coordinates.
    """

    reduced_params = [bool(param) for param in params]

    return coords[reduced_params]


def _get_reduced_shape(shape, params):
    """
    Gets only those dimensions of the coordinates that don't need to be broadcast.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates to reduce.
    params : list
        The params from which to check which dimensions to get.

    Returns
    -------
    reduced_coords : np.ndarray
        The reduced coordinates.
    """
    reduced_shape = tuple(l for l, p in zip(shape, params) if p)

    return reduced_shape


def _get_expanded_coords_data(coords, data, params, broadcast_shape):
    """
    Expand coordinates/data to broadcast_shape. Does most of the heavy lifting for broadcast_to.
    Produces sorted output for sorted inputs.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates to expand.
    data : np.ndarray
        The data corresponding to the coordinates.
    params : list
        The broadcast parameters.
    broadcast_shape : tuple[int]
        The shape to broadcast to.

    Returns
    -------
    expanded_coords : np.ndarray
        List of 1-D arrays. Each item in the list has one dimension of coordinates.
    expanded_data : np.ndarray
        The data corresponding to expanded_coords.
    """
    first_dim = -1
    expand_shapes = []
    for d, p, l in zip(range(len(broadcast_shape)), params, broadcast_shape):
        if p and first_dim == -1:
            expand_shapes.append(coords.shape[1])
            first_dim = d

        if not p:
            expand_shapes.append(l)

    all_idx = _cartesian_product(*(np.arange(d, dtype=np.min_scalar_type(d - 1)) for d in expand_shapes))
    dt = np.result_type(*(np.min_scalar_type(l - 1) for l in broadcast_shape))

    false_dim = 0
    dim = 0

    expanded_coords = np.empty((len(broadcast_shape), all_idx.shape[1]), dtype=dt)
    expanded_data = data[all_idx[first_dim]]

    for d, p, l in zip(range(len(broadcast_shape)), params, broadcast_shape):
        if p:
            expanded_coords[d] = coords[dim, all_idx[first_dim]]
        else:
            expanded_coords[d] = all_idx[false_dim + (d > first_dim)]
            false_dim += 1

        if p is not None:
            dim += 1

    return np.asarray(expanded_coords), np.asarray(expanded_data)


# (c) senderle
# Taken from https://stackoverflow.com/a/11146645/774273
# License: https://creativecommons.org/licenses/by-sa/3.0/
def _cartesian_product(*arrays):
    """
    Get the cartesian product of a number of arrays.

    Parameters
    ----------
    arrays : Tuple[np.ndarray]
        The arrays to get a cartesian product of. Always sorted with respect
        to the original array.

    Returns
    -------
    out : np.ndarray
        The overall cartesian product of all the input arrays.
    """
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)
    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows)


def _get_matching_coords(coords, params, shape):
    """
    Get the matching coords across a number of broadcast operands.

    Parameters
    ----------
    coords : list[numpy.ndarray]
        The input coordinates.
    params : list[Union[bool, none]]
        The broadcast parameters.
    Returns
    -------
    numpy.ndarray
        The broacasted coordinates
    """
    matching_coords = []
    dims = np.zeros(len(coords), dtype=np.uint8)

    for p_all in zip(*params):
        for i, p in enumerate(p_all):
            if p:
                matching_coords.append(coords[i][dims[i]])
                break
        else:
            matching_coords.append(coords[dims[0]])

        for i, p in enumerate(p_all):
            if p is not None:
                dims[i] += 1

    dtype = np.min_scalar_type(max(shape) - 1)

    return np.asarray(matching_coords, dtype=dtype)


def broadcast_to(x, shape):
    """
    Performs the equivalent of :obj:`numpy.broadcast_to` for :obj:`COO`. Note that
    this function returns a new array instead of a view.

    Parameters
    ----------
    shape : tuple[int]
        The shape to broadcast the data to.

    Returns
    -------
    COO
        The broadcasted sparse array.

    Raises
    ------
    ValueError
        If the operand cannot be broadcast to the given shape.

    See also
    --------
    :obj:`numpy.broadcast_to` : NumPy equivalent function
    """
    from .core import COO

    if shape == x.shape:
        return x

    result_shape = _get_broadcast_shape(x.shape, shape, is_result=True)
    params = _get_broadcast_parameters(x.shape, result_shape)
    coords, data = _get_expanded_coords_data(x.coords, x.data, params, result_shape)

    return COO(coords, data, shape=result_shape, has_duplicates=False,
               sorted=True)
