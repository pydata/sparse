import sparse

import pytest


@pytest.mark.parametrize("side", [16, 32, 48])
@pytest.mark.parametrize("step", [1, -1, 2])
def test_gcxs_sparse_basic_slicing(benchmark, side, step):
    """Keep stored entries fixed while the logical column grid grows."""
    x = sparse.COO(
        [[1, 2, 3, 4], [1, 2, 3, 1], [2, 1, 3, 1], [3, 1, 2, 1], [1, 2, 3, 1]],
        [5, 10, 2, 1],
        shape=(8, side, side, side, side),
    ).asformat("gcxs")
    key = (1, slice(None, None, step), slice(None), slice(None), slice(None))
    x[key]  # Numba compilation

    @benchmark
    def bench():
        x[key]
