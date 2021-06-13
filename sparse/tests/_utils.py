from hypothesis import given, strategies as st
from hypothesis.strategies import data, composite
from hypothesis.extra.numpy import array_shapes, basic_indices, broadcastable_shapes
import numpy as np
import sparse
import scipy.sparse


@composite
def gen_shape_data(draw):
    shape = draw(array_shapes(max_dims=3, min_side=5))
    data = draw(
        st.dictionaries(keys=st.integers(min_value=0, max_value=4), values=st.floats())
    )
    return shape, data


@composite
def gen_getitem_notimpl_err(draw):
    n = draw(st.integers(min_value=2, max_value=3))
    shape = draw(array_shapes(min_dims=n, max_dims=n, min_side=5, max_side=10))
    data = array_shapes(min_dims=3, max_dims=3, max_side=4)
    density = draw(st.floats(min_value=0, max_value=1))
    indices = draw(
        st.lists(st.lists(data), min_size=n - 1, max_size=n - 1).map(tuple),
    )
    return shape, density, indices


@composite
def gen_getitem_index_err(draw):
    n = draw(st.integers(min_value=1, max_value=5))
    shape = draw(array_shapes(max_dims=3, min_side=5))
    density = draw(st.floats(min_value=0, max_value=1))
    indices = draw(
        st.lists(
            st.lists(st.integers(min_value=1), min_size=n, max_size=n),
            min_size=2,
            max_size=5,
        ).map(tuple)
    )
    return shape, density, indices


@composite
def gen_setitem_notimpl_err(draw):
    shape = draw(array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=3))
    index = draw(
        st.lists(
            st.lists(st.integers(min_value=0, max_value=1), min_size=2, max_size=2)
        ).map(tuple)
    )

    return shape, index


@composite
def gen_setitem_val_err(draw):
    shape = draw(array_shapes(max_dims=2, min_side=5))
    index = draw(
        st.lists(
            st.lists(st.integers(), min_size=2, max_size=2), min_size=2, max_size=2
        ).map(tuple)
    )
    value_shape = draw(
        st.one_of(
            array_shapes(max_dims=2, min_side=1, max_side=3),
            array_shapes(max_dims=1, min_side=1, max_side=3),
        )
    )

    return shape, index, value_shape


@composite
def gen_transpose(draw):
    a = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=2, max_size=4))
    if len(a) is 2:
        b = draw(st.sampled_from([(1, 0), (0, 1)]))
    elif len(a) is 3:
        b = draw(st.sampled_from([(0, 2, 1), (2, 0, 1), (1, 2, 0)]))
    else:
        b = draw(st.sampled_from([(0, 3, 2, 1), (1, 0, 3, 2), (3, 2, 0, 1)]))

    return a, b


@composite
def gen_reductions(draw):
    reduction = (draw(st.sampled_from(["sum", "mean", "prod", "max", "std", "var"])),)
    kwargs = (draw(st.sampled_from([{"dtype": np.float32}, {}])),)
    axis = (draw(st.sampled_from([None, 0, 1, 2, (0, 2), -3, (1, -1)])),)
    keepdims = draw(st.sampled_from([True, False]))

    return reduction, kwargs, axis, keepdims


@composite
def gen_broadcast_shape(draw):
    shape1 = draw(
        st.lists(st.integers(min_value=1, max_value=5), min_size=2, max_size=3).map(
            tuple
        )
    )
    shape2 = draw(broadcastable_shapes(shape1))

    return shape1, shape2


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
    b = draw(
        st.sampled_from(
            [sparse.random((100, 100), density=0.01), scipy.sparse.random(100, 100)]
        )
    )

    return a, b
