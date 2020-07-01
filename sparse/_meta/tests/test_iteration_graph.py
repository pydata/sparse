import pytest

from sparse._meta.iteration_graph import Access, IterationGraph
from sparse._meta.format import Format, Tensor

S = slice(None)
N = None


@pytest.mark.parametrize(
    "ndim, numpy_idx, ndim_new, idxs",
    [
        (1, (N, S), 2, {1: 0}),
        (1, (S,), 1, {0: 0}),
        (1, (), 1, {0: 0}),
        (1, (S, N), 2, {0: 0}),
        (3, (S, N), 4, {0: 0, 2: 1, 3: 2}),
    ],
)
def test_access(ndim, numpy_idx, ndim_new, idxs):
    a = Access.from_numpy_notation(numpy_idx, ndim=ndim)
    assert a == Access(idxs, ndim=ndim_new)


@pytest.mark.parametrize(
    "ndim, idxs, axes, idxs_new",
    [
        (1, {0: 0}, (0,), {0: 0}),
        (2, {0: 0}, (1, 0), {1: 0}),
        (3, {1: 0, 0: 1, 2: 2}, None, {1: 0, 2: 1, 0: 2}),
    ],
)
def test_access_transpose(ndim, idxs, axes, idxs_new):
    a = Access(idxs, ndim=ndim)
    assert a.transpose(axes) == Access(idxs_new, ndim=ndim)
