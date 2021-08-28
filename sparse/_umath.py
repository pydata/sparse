import itertools

import numba
import numpy as np
import scipy.sparse

from itertools import zip_longest

from ._utils import isscalar, equivalent, _zero_of_dtype


def elemwise(func, *args, **kwargs):
    """
    Apply a function to any number of arguments.

    Parameters
    ----------
    func : Callable
        The function to apply. Must support broadcasting.
    *args : tuple, optional
        The arguments to the function. Can be :obj:`SparseArray` objects
        or :obj:`scipy.sparse.spmatrix` objects.
    **kwargs : dict, optional
        Any additional arguments to pass to the function.

    Returns
    -------
    SparseArray
        The result of applying the function.

    Raises
    ------
    ValueError
        If the operation would result in a dense matrix, or if the operands
        don't have broadcastable shapes.

    See Also
    --------
    :obj:`numpy.ufunc` :
        A similar Numpy construct. Note that any :code:`ufunc` can be used
        as the :code:`func` input to this function.

    Notes
    -----
    Previously, operations with Numpy arrays were sometimes supported. Now,
    it is necessary to convert Numpy arrays to :obj:`COO` objects.
    """

    return _Elemwise(func, *args, **kwargs).get_result()


@numba.jit(nopython=True, nogil=True)
def _match_arrays(a, b):  # pragma: no cover
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


def _get_nary_broadcast_shape(*shapes):
    """
    Broadcast any number of shapes to a result shape.

    Parameters
    ----------
    *shapes : tuple[tuple[int]]
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
            shapes_str = ", ".join(str(shape) for shape in shapes)
            raise ValueError(
                "operands could not be broadcast together with shapes %s" % shapes_str
            )

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
    if not all(
        (l1 == l2) or (l1 == 1) or ((l2 == 1) and not is_result)
        for l1, l2 in zip(shape1[::-1], shape2[::-1])
    ):
        raise ValueError(
            "operands could not be broadcast together with shapes %s, %s"
            % (shape1, shape2)
        )

    result_shape = tuple(
        l1 if l1 != 1 else l2
        for l1, l2 in zip_longest(shape1[::-1], shape2[::-1], fillvalue=1)
    )[::-1]

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
    params = [
        None if l1 is None else l1 == l2
        for l1, l2 in zip_longest(shape[::-1], broadcast_shape[::-1], fillvalue=None)
    ][::-1]

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
    shape : np.ndarray
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

    all_idx = _cartesian_product(*(np.arange(d, dtype=np.intp) for d in expand_shapes))

    false_dim = 0
    dim = 0

    expanded_coords = np.empty((len(broadcast_shape), all_idx.shape[1]), dtype=np.intp)

    if first_dim != -1:
        expanded_data = data[all_idx[first_dim]]
    else:
        expanded_coords = (
            all_idx if len(data) else np.empty((0, all_idx.shape[1]), dtype=np.intp)
        )
        expanded_data = np.repeat(data, np.prod(broadcast_shape, dtype=np.int64))
        return np.asarray(expanded_coords), np.asarray(expanded_data)

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
    *arrays : Tuple[np.ndarray]
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


def _get_matching_coords(coords, params):
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

    return np.asarray(matching_coords, dtype=np.intp)


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

    See Also
    --------
    :obj:`numpy.broadcast_to` : NumPy equivalent function
    """
    from ._coo import COO

    if shape == x.shape:
        return x

    result_shape = _get_broadcast_shape(x.shape, shape, is_result=True)
    params = _get_broadcast_parameters(x.shape, result_shape)
    coords, data = _get_expanded_coords_data(x.coords, x.data, params, result_shape)

    # Check if all the non-broadcast axes are next to each other
    nonbroadcast_idx = [idx for idx, p in enumerate(params) if p]
    diff_nonbroadcast_idx = [
        a - b for a, b in zip(nonbroadcast_idx[1:], nonbroadcast_idx[:-1])
    ]
    sorted = all(d == 1 for d in diff_nonbroadcast_idx)

    return COO(
        coords,
        data,
        shape=result_shape,
        has_duplicates=False,
        sorted=sorted,
        fill_value=x.fill_value,
    )


class _Elemwise:
    def __init__(self, func, *args, **kwargs):
        """
        Initialize the element-wise function calculator.

        Parameters
        ----------
        func : types.Callable
            The function to compute
        *args : tuple[Union[SparseArray, ndarray, scipy.sparse.spmatrix]]
            The arguments to compute the function on.
        **kwargs : dict
            Extra arguments to pass to the function.
        """
        from ._coo import COO
        from ._sparse_array import SparseArray
        from ._compressed import GCXS
        from ._dok import DOK

        processed_args = []
        out_type = GCXS

        sparse_args = [arg for arg in args if isinstance(arg, SparseArray)]

        if all(isinstance(arg, DOK) for arg in sparse_args):
            out_type = DOK
        elif all(isinstance(arg, GCXS) for arg in sparse_args):
            out_type = GCXS
        else:
            out_type = COO

        for arg in args:
            if isinstance(arg, scipy.sparse.spmatrix):
                processed_args.append(COO.from_scipy_sparse(arg))
            elif isscalar(arg) or isinstance(arg, np.ndarray):
                # Faster and more reliable to pass ()-shaped ndarrays as scalars.
                processed_args.append(np.asarray(arg))
            elif isinstance(arg, SparseArray):
                if not isinstance(arg, COO):
                    arg = arg.asformat(COO)
                if arg.ndim == 0:
                    arg = arg.todense()
                processed_args.append(arg)
            else:
                self.args = None
                return

        self.out_type = out_type
        self.args = tuple(processed_args)
        self.func = func
        self.dtype = kwargs.pop("dtype", None)
        self.kwargs = kwargs
        self.cache = {}
        self._dense_result = False

        self._check_broadcast()
        self._get_fill_value()

    def get_result(self):
        from ._coo import COO

        if self.args is None:
            return NotImplemented

        if self._dense_result:
            args = [a.todense() if isinstance(a, COO) else a for a in self.args]
            return self.func(*args, **self.kwargs)

        if any(s == 0 for s in self.shape):
            data = np.empty((0,), dtype=self.fill_value.dtype)
            coords = np.empty((0, len(self.shape)), dtype=np.intp)
            return COO(
                coords,
                data,
                shape=self.shape,
                has_duplicates=False,
                fill_value=self.fill_value,
            )

        data_list = []
        coords_list = []

        for mask in itertools.product(
            *[[True, False] if isinstance(arg, COO) else [None] for arg in self.args]
        ):
            if not any(mask):
                continue

            r = self._get_func_coords_data(mask)

            if r is not None:
                coords_list.append(r[0])
                data_list.append(r[1])

        # Concatenate matches and mismatches
        data = (
            np.concatenate(data_list)
            if len(data_list)
            else np.empty((0,), dtype=self.fill_value.dtype)
        )
        coords = (
            np.concatenate(coords_list, axis=1)
            if len(coords_list)
            else np.empty((0, len(self.shape)), dtype=np.intp)
        )

        return COO(
            coords,
            data,
            shape=self.shape,
            has_duplicates=False,
            fill_value=self.fill_value,
        ).asformat(self.out_type)

    def _get_fill_value(self):
        """
        A function that finds and returns the fill-value.

        Raises
        ------
        ValueError
            If the fill-value is inconsistent.
        """
        from ._coo import COO

        zero_args = tuple(
            arg.fill_value[...] if isinstance(arg, COO) else arg for arg in self.args
        )

        # Some elemwise functions require a dtype argument, some abhorr it.
        try:
            fill_value_array = self.func(
                *np.broadcast_arrays(*zero_args), dtype=self.dtype, **self.kwargs
            )
        except TypeError:
            fill_value_array = self.func(
                *np.broadcast_arrays(*zero_args), **self.kwargs
            )

        try:
            fill_value = fill_value_array[(0,) * fill_value_array.ndim]
        except IndexError:
            zero_args = tuple(
                arg.fill_value if isinstance(arg, COO) else _zero_of_dtype(arg.dtype)
                for arg in self.args
            )
            fill_value = self.func(*zero_args, **self.kwargs)[()]

        equivalent_fv = equivalent(fill_value, fill_value_array).all()
        if not equivalent_fv and self.shape != self.ndarray_shape:
            raise ValueError(
                "Performing a mixed sparse-dense operation that would result in a dense array. "
                "Please make sure that func(sparse_fill_values, ndarrays) is a constant array."
            )
        elif not equivalent_fv:
            self._dense_result = True

        # Store dtype separately if needed.
        if self.dtype is not None:
            fill_value = fill_value.astype(self.dtype)

        self.fill_value = fill_value
        self.dtype = self.fill_value.dtype

    def _check_broadcast(self):
        """
        Checks if adding the ndarrays changes the broadcast shape.

        Raises
        ------
        ValueError
            If the check fails.
        """
        from ._coo import COO

        full_shape = _get_nary_broadcast_shape(*tuple(arg.shape for arg in self.args))
        non_ndarray_shape = _get_nary_broadcast_shape(
            *tuple(arg.shape for arg in self.args if isinstance(arg, COO))
        )
        ndarray_shape = _get_nary_broadcast_shape(
            *tuple(arg.shape for arg in self.args if isinstance(arg, np.ndarray))
        )

        self.shape = full_shape
        self.ndarray_shape = ndarray_shape
        self.non_ndarray_shape = non_ndarray_shape

    def _get_func_coords_data(self, mask):
        """
        Gets the coords/data for a certain mask

        Parameters
        ----------
        mask : tuple[Union[bool, NoneType]]
            The mask determining whether to match or unmatch.

        Returns
        -------
        None or tuple
            The coords/data tuple for the given mask.
        """
        from ._coo import COO

        matched_args = [arg for arg, m in zip(self.args, mask) if m is not None and m]
        unmatched_args = [
            arg for arg, m in zip(self.args, mask) if m is not None and not m
        ]
        ndarray_args = [arg for arg, m in zip(self.args, mask) if m is None]

        matched_broadcast_shape = _get_nary_broadcast_shape(
            *tuple(arg.shape for arg in itertools.chain(matched_args, ndarray_args))
        )

        matched_arrays = self._match_coo(
            *matched_args, cache=self.cache, broadcast_shape=matched_broadcast_shape
        )

        func_args = []

        m_arg = 0
        for arg, m in zip(self.args, mask):
            if m is None:
                func_args.append(
                    np.broadcast_to(arg, matched_broadcast_shape)[
                        tuple(matched_arrays[0].coords)
                    ]
                )
                continue

            if m:
                func_args.append(matched_arrays[m_arg].data)
                m_arg += 1
            else:
                func_args.append(arg.fill_value)

        # Try our best to preserve the output dtype.
        try:
            func_data = self.func(*func_args, dtype=self.dtype, **self.kwargs)
        except TypeError:
            try:
                func_args = np.broadcast_arrays(*func_args)
                out = np.empty(func_args[0].shape, dtype=self.dtype)
                func_data = self.func(*func_args, out=out, **self.kwargs)
            except TypeError:
                func_data = self.func(*func_args, **self.kwargs).astype(self.dtype)

        unmatched_mask = ~equivalent(func_data, self.fill_value)

        if not unmatched_mask.any():
            return None

        func_coords = matched_arrays[0].coords[:, unmatched_mask]
        func_data = func_data[unmatched_mask]

        if matched_arrays[0].shape != self.shape:
            params = _get_broadcast_parameters(matched_arrays[0].shape, self.shape)
            func_coords, func_data = _get_expanded_coords_data(
                func_coords, func_data, params, self.shape
            )

        if all(m is None or m for m in mask):
            return func_coords, func_data

        # Not really sorted but we need the sortedness.
        func_array = COO(
            func_coords, func_data, self.shape, has_duplicates=False, sorted=True
        )

        unmatched_mask = np.ones(func_array.nnz, dtype=np.bool_)

        for arg in unmatched_args:
            matched_idx = self._match_coo(func_array, arg, return_midx=True)[0]
            unmatched_mask[matched_idx] = False

        coords = np.asarray(func_array.coords[:, unmatched_mask], order="C")
        data = np.asarray(func_array.data[unmatched_mask], order="C")

        return coords, data

    @staticmethod
    def _match_coo(*args, **kwargs):
        """
        Matches the coordinates for any number of input :obj:`COO` arrays.
        Equivalent to "sparse" broadcasting for all arrays.

        Parameters
        ----------
        *args : Tuple[COO]
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
        from ._coo import COO
        from ._coo.common import linear_loc

        cache = kwargs.pop("cache", None)
        return_midx = kwargs.pop("return_midx", False)
        broadcast_shape = kwargs.pop("broadcast_shape", None)

        if kwargs:
            raise ValueError("Unknown kwargs: {}".format(kwargs.keys()))

        if return_midx and (len(args) != 2 or cache is not None):
            raise NotImplementedError(
                "Matching indices only supported for two args, and no cache."
            )

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
            params = [
                _get_broadcast_parameters(arg.shape, current_shape) for arg in cargs
            ]
            reduced_params = [all(p) for p in zip(*params)]
            reduced_shape = _get_reduced_shape(
                arg2.shape, _rev_idx(reduced_params, arg2.ndim)
            )

            reduced_coords = [
                _get_reduced_coords(arg.coords, _rev_idx(reduced_params, arg.ndim))
                for arg in cargs
            ]

            linear = [linear_loc(rc, reduced_shape) for rc in reduced_coords]
            sorted_idx = [np.argsort(idx) for idx in linear]
            linear = [idx[s] for idx, s in zip(linear, sorted_idx)]
            matched_idx = _match_arrays(*linear)

            if return_midx:
                matched_idx = [
                    sidx[midx] for sidx, midx in zip(sorted_idx, matched_idx)
                ]
                return matched_idx

            coords = [arg.coords[:, s] for arg, s in zip(cargs, sorted_idx)]
            mcoords = [c[:, idx] for c, idx in zip(coords, matched_idx)]
            mcoords = _get_matching_coords(mcoords, params)
            mdata = [arg.data[sorted_idx[0]][matched_idx[0]] for arg in matched_arrays]
            mdata.append(arg2.data[sorted_idx[1]][matched_idx[1]])
            # The coords aren't truly sorted, but we don't need them, so it's
            # best to avoid the extra cost.
            matched_arrays = [
                COO(mcoords, md, shape=current_shape, sorted=True, has_duplicates=False)
                for md in mdata
            ]

            if cache is not None:
                cache[key] = matched_arrays

        if broadcast_shape is not None and matched_arrays[0].shape != broadcast_shape:
            params = _get_broadcast_parameters(matched_arrays[0].shape, broadcast_shape)
            coords, idx = _get_expanded_coords_data(
                matched_arrays[0].coords,
                np.arange(matched_arrays[0].nnz),
                params,
                broadcast_shape,
            )

            matched_arrays = [
                COO(
                    coords,
                    arr.data[idx],
                    shape=broadcast_shape,
                    sorted=True,
                    has_duplicates=False,
                )
                for arr in matched_arrays
            ]

        return matched_arrays


def _rev_idx(arg, idx):
    if idx == 0:
        return arg[len(arg) :]

    return arg[-idx:]
