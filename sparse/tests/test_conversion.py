import sparse
import pytest
from sparse._utils import assert_eq
from hypothesis import settings, given, strategies as st
from _utils import gen_sparse_random_conversion


FORMATS_ND = [
    sparse.COO,
    sparse.DOK,
    sparse.GCXS,
]

FORMATS_2D = [
    sparse._compressed.CSC,
    sparse._compressed.CSR,
]

FORMATS = FORMATS_2D + FORMATS_ND


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


@pytest.mark.parametrize("format1", FORMATS_ND)
@pytest.mark.parametrize("format2", FORMATS_ND)
def test_conversion_scalar(format1, format2):
    x = sparse.random((), format=format1, fill_value=0.5)
    y = x.asformat(format2)
    assert_eq(x, y)
