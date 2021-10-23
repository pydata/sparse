import sparse
import pytest
from sparse._utils import assert_eq
from hypothesis import settings, given, strategies as st
from _utils import gen_sparse_random_conversion


FORMATS = [
    sparse.COO,
    sparse.DOK,
    sparse.GCXS,
    sparse._compressed.CSC,
    sparse._compressed.CSR,
]


@settings(deadline=None)
@given(
    format=st.sampled_from(FORMATS),
    x=gen_sparse_random_conversion((10, 10), density=0.5, fill_value=0.5),
)
def test_conversion(format, x):
    y = x.asformat(format)
    assert_eq(x, y)


def test_extra_kwargs():
    x = sparse.full((2, 2), 1, format="gcxs", compressed_axes=[1])
    y = sparse.full_like(x, 1)

    assert_eq(x, y)
