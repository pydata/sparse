import sparse
from sparse.numba_backend._utils import assert_eq

import pytest

import numpy as np
import scipy.sparse as sps

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


def test_non_canonical_conversion():
    """
    Regression test for gh-602.

    Adapted from https://github.com/LiberTEM/sparseconverter/blob/4cfc0ee2ad4c37b07742db8f3643bcbd858a4e85/src/sparseconverter/__init__.py#L154-L183
    """
    data = np.array((2.0, 1.0, 3.0, 3.0, 1.0))
    indices = np.array((1, 0, 0, 1, 1), dtype=int)
    indptr = np.array((0, 2, 5), dtype=int)

    x = sps.csr_matrix((data, indices, indptr), shape=(2, 2))
    ref = np.array(((1.0, 2.0), (3.0, 4.0)))

    gcxs_check = sparse.GCXS(x)
    assert np.all(gcxs_check[:1].todense() == ref[:1]) and np.all(gcxs_check[1:].todense() == ref[1:])
