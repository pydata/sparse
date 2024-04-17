import sparse
from sparse.numba_backend._utils import assert_eq

import pytest

FORMATS_ND = [
    sparse.COO,
    sparse.DOK,
    sparse.GCXS,
]

FORMATS_2D = [
    sparse.numba_backend._compressed.CSC,
    sparse.numba_backend._compressed.CSR,
]

FORMATS = FORMATS_2D + FORMATS_ND


@pytest.mark.parametrize("format1", FORMATS)
@pytest.mark.parametrize("format2", FORMATS)
def test_conversion(format1, format2):
    x = sparse.random((10, 10), density=0.5, format=format1, fill_value=0.5)
    y = x.asformat(format2)
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
