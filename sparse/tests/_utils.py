from hypothesis import given, strategies as st
from hypothesis.strategies import data, composite
from hypothesis.extra.numpy import array_shapes, basic_indices


@composite
def gen_shape_data(draw):
    shape = draw(array_shapes(max_dims=3, min_side=5))
    data = draw(
        st.dictionaries(keys=st.integers(min_value=0, max_value=4), values=st.floats())
    )
    return shape, data


@composite
def gen_getitem_notimpl_err(draw):
    shape = draw(array_shapes(max_dims=3, min_side=5))
    density = draw(st.floats(min_value=0, max_value=1))
    indices = draw(
        st.one_of(
            st.lists(
                st.lists(st.integers(), min_size=4, max_size=5), min_size=4, max_size=5
            ).map(tuple),
            st.lists(st.lists(st.integers(), min_size=4, max_size=5), max_size=3).map(
                tuple
            ),
        )
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
