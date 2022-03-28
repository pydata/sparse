from numbers import Rational
from hypothesis import strategies as st
from hypothesis.strategies import composite
from hypothesis.extra.numpy import (
    array_shapes,
    broadcastable_shapes,
    mutually_broadcastable_shapes,
)
import numpy as np
import sparse
import scipy.sparse

from sparse._utils import random_value_array


# Code copied from numpy.side_tricks module.
# To be used till Numpy is bumped up
def _broadcast_shape(*args):
    """
    Utility function broadcast_shapes()
    """
    # use the old-iterator because np.nditer does not handle size 0 arrays
    # consistently
    b = np.broadcast(*args[:32])
    # unfortunately, it cannot handle 32 or more arguments directly
    for pos in range(32, len(args), 31):
        # ironically, np.broadcast does not properly handle np.broadcast
        # objects (it treats them as scalars)
        # use broadcasting to avoid allocating the full array
        b = broadcast_to(0, b.shape)
        b = np.broadcast(b, *args[pos : (pos + 31)])
    return b.shape


def broadcast_shapes(*args):
    arrays = [np.empty(x, dtype=[]) for x in args]
    return _broadcast_shape(*arrays)


@composite
def gen_shape_data(draw):
    shape = draw(array_shapes(max_dims=3, min_side=5))
    data = draw(
        st.dictionaries(keys=st.integers(min_value=0, max_value=4), values=st.floats())
    )
    return shape, data


@composite
def gen_notimpl_err(draw):
    n = draw(st.integers(min_value=2, max_value=3))
    shape = draw(array_shapes(min_dims=n, max_dims=n, min_side=5, max_side=10))
    data = array_shapes(min_dims=3, max_dims=3, min_side=1, max_side=4)
    density = draw(st.floats(min_value=0, max_value=1))
    indices = draw(st.lists(st.lists(data), min_size=n - 1, max_size=n - 1).map(tuple),)
    return shape, density, indices


@composite
def gen_getitem_index_err(draw):
    n = draw(st.integers(min_value=2, max_value=2))
    shape = draw(array_shapes(min_dims=n, max_dims=n, min_side=5, max_side=10))
    density = draw(st.floats(min_value=0, max_value=1))
    if n == 2:
        indices = draw(
            st.tuples(
                st.lists(st.integers(min_value=1, max_value=3), min_size=1, max_size=2),
                st.lists(st.integers(min_value=1, max_value=3), min_size=3, max_size=4),
            )
        )
    else:
        indices = draw(
            st.tuples(
                st.lists(st.integers(min_value=1, max_value=3), min_size=1, max_size=2),
                st.lists(st.integers(min_value=1, max_value=3), min_size=1, max_size=2),
                st.lists(st.integers(min_value=1, max_value=3), min_size=3, max_size=4),
            )
        )

    return shape, density, indices


@composite
def gen_setitem_val_err(draw):
    shape = draw(array_shapes(min_dims=2, max_dims=2, min_side=4, max_side=6))
    index = draw(
        st.tuples(
            st.lists(st.integers(min_value=1, max_value=3), min_size=2, max_size=2),
            st.lists(st.integers(min_value=1, max_value=3), min_size=2, max_size=2),
        )
    )
    value_shape = draw(array_shapes(min_dims=4, max_dims=6, min_side=3, max_side=5),)

    return shape, index, value_shape


@composite
def gen_transpose(draw):
    a = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=2, max_size=4))
    if len(a) == 2:
        b = draw(st.sampled_from([(1, 0), (0, 1)]))
    elif len(a) == 3:
        b = draw(st.sampled_from([(0, 2, 1), (2, 0, 1), (1, 2, 0)]))
    else:
        b = draw(st.sampled_from([(0, 3, 2, 1), (1, 0, 3, 2), (3, 2, 0, 1)]))

    s = draw(gen_sparse_random(a, density=0.5, format="gcxs"))

    return s, b


@composite
def gen_sparse_random(draw, shape, **kwargs):
    seed = draw(st.integers(min_value=0, max_value=100))
    return sparse.random(shape, random_state=seed, **kwargs)


@composite
def gen_reductions(draw, function=False):
    reduction = draw(
        st.sampled_from(
            ["sum", "mean", "prod", "max", "min", "std", "var", "any", "all"]
        )
    )
    kwargs = {}
    if reduction not in {"max", "min", "any", "all"}:
        dtype = draw(st.sampled_from([np.float32, np.float64, None]))
        kwargs["dtype"] = dtype
    kwargs["axis"] = draw(st.sampled_from([None, 0, 1, 2, (0, 2), -3, (1, -1)]))
    kwargs["keepdims"] = draw(st.sampled_from([True, False]))
    if function:
        reduction = getattr(np, reduction)

    return reduction, kwargs


@composite
def gen_broadcast_shape(draw, num_shapes=2):
    shape_1, shape_2 = draw(
        mutually_broadcastable_shapes(
            num_shapes=num_shapes, max_dims=2, min_side=2, max_side=5
        )
    ).input_shapes

    return shape_1, shape_2


@composite
def gen_matmul_warning(draw):
    a = draw(
        st.sampled_from(
            [
                sparse.GCXS.from_numpy(
                    np.random.choice(
                        [0, np.nan, 2], size=[100, 100], p=[0.99, 0.001, 0.009]
                    )
                ),
                sparse.COO.from_numpy(
                    np.random.choice(
                        [0, np.nan, 2], size=[100, 100], p=[0.99, 0.001, 0.009]
                    )
                ),
                sparse.GCXS.from_numpy(
                    np.random.choice(
                        [0, np.nan, 2], size=[100, 100], p=[0.99, 0.001, 0.009]
                    )
                ),
                np.random.choice(
                    [0, np.nan, 2], size=[100, 100], p=[0.99, 0.001, 0.009]
                ),
            ]
        )
    )
    if not isinstance(a, np.ndarray):
        b = draw(
            st.sampled_from(
                [sparse.random((100, 100), density=0.01), scipy.sparse.random(100, 100)]
            )
        )
    else:
        b = draw(st.sampled_from([sparse.random((100, 100), density=0.01)]))

    return a, b


@composite
def gen_broadcast_shape_dot(draw, max_dims=4):
    a_shape, b_shape = draw(
        mutually_broadcastable_shapes(
            signature=np.matmul.signature, max_dims=2, min_side=2, max_side=5
        )
    ).input_shapes
    if len(a_shape) != 1 and len(b_shape) != 1:
        a_shape = draw(array_shapes(min_side=2, max_side=5, max_dims=2)) + a_shape
        b_shape = draw(array_shapes(min_side=2, max_side=5, max_dims=2)) + b_shape

    return tuple(a_shape), tuple(b_shape)


@composite
def gen_broadcast_to(draw):
    shape1 = draw(
        st.lists(st.integers(min_value=2, max_value=5), min_size=2, max_size=3).map(
            tuple
        )
    )
    shape2 = draw(broadcastable_shapes(shape1, min_dims=2, max_dims=3))

    return shape1, shape2


@composite
def gen_getitem(draw):
    shape = draw(
        st.tuples(
            st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=5)
        )
    )
    density = draw(st.floats(min_value=0, max_value=1))
    indices = draw(st.tuples(st.slices(shape[0]), st.slices(shape[1])))

    return shape, density, indices


@composite
def gen_setitem(draw):
    a = draw(st.integers(min_value=2, max_value=5))
    shape = draw(array_shapes(min_dims=2, max_dims=2, min_side=a))
    index = draw(array_shapes(min_dims=2, max_dims=2, max_side=a - 1))

    return shape, index


@composite
def gen_reshape(draw):
    x = draw(st.integers(min_value=1, max_value=10))
    y = draw(st.integers(min_value=1, max_value=10))
    a = draw(st.tuples(x, y))
    b = draw(st.tuples(y, x))

    return a, b


@composite
def gen_stack(draw):
    shape = draw(st.sampled_from([(5,), (2, 3, 4), (5, 2)]))
    axis = draw(st.sampled_from([0, 1, -1]))
    xx = draw(gen_sparse_random(shape, format="gcxs"))
    yy = draw(gen_sparse_random(shape, format="gcxs"))
    zz = draw(gen_sparse_random(shape, format="gcxs"))

    return shape, axis, xx, yy, zz


@composite
def gen_flatten(draw):
    in_shape = draw(st.sampled_from([(5, 5), 62, (3, 3, 3)]))
    s = draw(gen_sparse_random(in_shape, format="gcxs", density=0.5))

    return s


@composite
def gen_pad_valid(draw):
    pad_width = draw(
        st.sampled_from([2, (2, 1), ((2), (1)), ((1, 2), (4, 5), (7, 8)),])
    )
    constant_values = draw(st.sampled_from([0, 1, 150, np.nan]))
    y = draw(
        gen_sparse_random(
            (50, 50, 3), density=0.15, fill_value=constant_values, format="gcxs"
        )
    )

    return pad_width, constant_values, y


@composite
def gen_pad_invalid(draw):
    pad_width = draw(st.sampled_from([((2, 1), (5, 7))]))
    constant_values = draw(st.sampled_from([150, 2, (1, 2)]))
    fill_value = draw(st.floats(min_value=0, max_value=10))
    y = draw(
        gen_sparse_random(
            (50, 50, 3), density=0.15, format="gcxs", fill_value=fill_value
        )
    )

    return pad_width, constant_values, y


@composite
def gen_advanced_indexing(draw):
    index = draw(
        st.sampled_from(
            [
                ([1, 0], 0),
                (1, [0, 2]),
                (0, [1, 0], 0),
                (1, [2, 0], 0),
                ([True, False], slice(1, None), slice(-2, None)),
                (slice(1, None), slice(-2, None), [True, False, True, False]),
                ([1, 0],),
                (Ellipsis, [2, 1, 3]),
                (slice(None), [2, 1, 2]),
                (1, [2, 0, 1]),
            ]
        )
    )
    compressed_axes = draw(st.sampled_from([(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]))
    s = draw(
        gen_sparse_random(
            (2, 3, 4), density=0.5, format="gcxs", compressed_axes=compressed_axes
        )
    )

    return index, s


@composite
def gen_random_seed(draw):
    n = draw(st.integers(min_value=0, max_value=99))
    return n


@composite
def gen_matmul_shapes(draw):
    return draw(
        mutually_broadcastable_shapes(
            signature=np.matmul.signature, max_dims=4, min_side=2, max_side=5
        )
    ).input_shapes


@composite
def gen_sparse_random_slicing(draw, shapes, **kwargs):
    seed = draw(st.integers(min_value=0, max_value=100))
    compressed_axes = draw(st.sampled_from([(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]))
    return sparse.random(
        shapes, random_state=seed, compressed_axes=compressed_axes, **kwargs
    )


@composite
def gen_sparse_random_from(draw, shapes, **kwargs):
    seed = draw(st.integers(min_value=0, max_value=100))
    source_type = draw(st.sampled_from(["gcxs", "coo"]))
    return sparse.random(shapes, random_state=seed, format=source_type, **kwargs)


@composite
def gen_sparse_random_scipy(draw, shapes, **kwargs):
    seed = draw(st.integers(min_value=0, max_value=100))
    cls_str = draw(st.sampled_from(["coo", "dok", "csr", "csc", "gcxs"]))
    return sparse.random(shapes, random_state=seed, format=cls_str, **kwargs)


FORMATS = [
    sparse.COO,
    sparse.DOK,
    sparse.GCXS,
    sparse._compressed.CSC,
    sparse._compressed.CSR,
]


@composite
def gen_sparse_random_conversion(draw, shapes, **kwargs):
    seed = draw(st.integers(min_value=0, max_value=100))
    cls_str = draw(st.sampled_from(FORMATS))
    return sparse.random(shapes, random_state=seed, format=cls_str, **kwargs)


@composite
def gen_sparse_random_nan_reduction(draw, shapes, **kwargs):
    fraction = draw(st.sampled_from([0.25, 0.5, 0.75, 1.0]))
    return sparse.random(
        shapes, data_rvs=random_value_array(np.nan, fraction), **kwargs
    )


@composite
def gen_sparse_random_kron_a(draw, **kwargs):
    seed = draw(st.integers(min_value=0, max_value=100))
    ndim = draw(st.sampled_from([1, 2, 3]))
    shapes = (2, 3, 4)[:ndim]
    return sparse.random(shapes, **kwargs)


@composite
def gen_sparse_random_kron_b(draw, **kwargs):
    seed = draw(st.integers(min_value=0, max_value=100))
    ndim = draw(st.sampled_from([1, 2, 3]))
    shapes = (5, 6, 7)[:ndim]
    return sparse.random(shapes, **kwargs)


@composite
def gen_sparse_random_three(draw):
    shape = draw(
        st.sampled_from(
            [
                [(2,), (3, 2), (4, 3, 2)],
                [(3,), (2, 3), (2, 2, 3)],
                [(2,), (2, 2), (2, 2, 2)],
                [(4,), (4, 4), (4, 4, 4)],
                [(4,), (4, 4), (4, 4, 4)],
                [(4,), (4, 4), (4, 4, 4)],
                [(1, 1, 2), (1, 3, 1), (4, 1, 1)],
                [(2,), (2, 1), (2, 1, 1)],
                [(3,), (), (2, 3)],
                [(4, 4), (), ()],
            ]
        )
    )
    seed = draw(st.integers(min_value=0, max_value=100))
    a = sparse.random(shape[0], density=0.5, random_state=seed).astype(np.bool_)
    b = sparse.random(shape[1], density=0.5, random_state=seed)
    c = sparse.random(shape[2], density=0.5, random_state=seed)

    return a, b, c


@composite
def gen_sparse_random_outer(draw):
    seed = draw(st.integers(min_value=0, max_value=100))
    shape = draw(st.sampled_from([(2,), (2, 3), (2, 3, 4)]))
    return sparse.random(shape, density=0.5)


@composite
def gen_sparse_random_getitem_single(draw):
    seed = draw(st.integers(min_value=0, max_value=100))
    shape = draw(st.sampled_from([(2,), (2, 3), (2, 3, 4)]))
    density = draw(st.sampled_from([0.1, 0.3, 0.5, 0.7]))
    return sparse.random(shape, density=density, random_state=seed, format="dok")


@composite
def gen_sparse_random_pad_invalid(draw, shape, **kwargs):
    seed = draw(st.integers(min_value=0, max_value=100))
    fill_value = draw(st.floats(min_value=0, max_value=10))
    return sparse.random(shape, random_state=seed, fill_value=fill_value, **kwargs)


@composite
def gen_sparse_random_elemwise(draw, shape, **kwargs):
    seed = draw(st.integers(min_value=0, max_value=100))
    format = draw(st.sampled_from([sparse.COO, sparse.GCXS, sparse.DOK]))
    return sparse.random(shape, random_state=seed, format=format, **kwargs), format


@composite
def gen_sparse_random_elemwise_mixed(draw, shape, **kwargs):
    seed = draw(st.integers(min_value=0, max_value=100))
    format = draw(st.sampled_from([sparse.COO, sparse.GCXS, sparse.DOK]))
    return sparse.random(shape, random_state=seed, format=format, **kwargs)


@composite
def gen_sparse_random_elemwise_trinary(draw, **kwargs):
    seed = draw(st.integers(min_value=0, max_value=100))
    formats = draw(
        st.sampled_from(
            [
                [sparse.COO, sparse.COO, sparse.COO],
                [sparse.GCXS, sparse.GCXS, sparse.GCXS],
                [sparse.COO, sparse.GCXS, sparse.GCXS],
            ]
        )
    )
    shape = draw(st.sampled_from([(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]))
    shape1, shape2, shape3 = draw(
        mutually_broadcastable_shapes(num_shapes=3, base_shape=shape)
    ).input_shapes
    return (
        sparse.random(shape1, random_state=seed, format=formats[0], **kwargs),
        sparse.random(shape2, random_state=seed, format=formats[1], **kwargs),
        sparse.random(shape3, random_state=seed, format=formats[2], **kwargs),
    )


@composite
def gen_sparse_random_elemwise_trinary_broadcasting(draw, **kwargs):
    seed = draw(st.integers(min_value=0, max_value=100))
    shape = draw(
        st.sampled_from(
            [
                [(2,), (3, 2), (4, 3, 2)],
                [(3,), (2, 3), (2, 2, 3)],
                [(2,), (2, 2), (2, 2, 2)],
                [(4,), (4, 4), (4, 4, 4)],
                [(4,), (4, 4), (4, 4, 4)],
                [(4,), (4, 4), (4, 4, 4)],
                [(1, 1, 2), (1, 3, 1), (4, 1, 1)],
                [(2,), (2, 1), (2, 1, 1)],
            ]
        )
    )
    args = [sparse.random(s, random_state=seed, **kwargs) for s in shape]
    return args


@composite
def gen_broadcast_shape2(draw, num_shapes=2):
    shape_1, shape_2 = draw(
        mutually_broadcastable_shapes(
            num_shapes=num_shapes, min_dims=1, max_dims=2, min_side=2, max_side=5
        )
    ).input_shapes

    if len(shape_1) < len(shape_2):
        shape_1, shape_2 = shape_2, shape_1

    return shape_1, shape_2


@composite
def gen_sparse_random_elemwise_binary(draw, **kwargs):
    shape_1, shape_2 = draw(
        mutually_broadcastable_shapes(
            num_shapes=2, min_dims=1, max_dims=2, min_side=2, max_side=5
        )
    ).input_shapes

    format = draw(st.sampled_from([sparse.COO, sparse.GCXS, sparse.DOK]))

    xs = sparse.random(shape_1, format=format, **kwargs)
    ys = sparse.random(shape_2, format=format, **kwargs)

    return xs, ys
