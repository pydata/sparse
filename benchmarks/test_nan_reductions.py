import sparse

import pytest

import numpy as np


@pytest.fixture(params=[0.0, np.nan], ids=["zero-fill", "nan-fill"])
def nan_reduction_array(request, rng):
    # Shape and density from the workload reported in gh-908.
    return sparse.random((100, 100, 100), density=0.1, fill_value=request.param, random_state=rng)


@pytest.mark.parametrize("operation", ["replace_nan", "nansum"])
def test_nan_reduction(benchmark, nan_reduction_array, operation):
    array = nan_reduction_array

    def compute():
        if operation == "replace_nan":
            return sparse.where(np.isnan(array), 0, array)
        return sparse.nansum(array, axis=0)

    compute()

    @benchmark
    def bench():
        compute()


@pytest.mark.parametrize("operation", ["sum", "multiply", "scaled_sum"])
def test_nary_coordinate_union(benchmark, nan_reduction_array, operation):
    array = nan_reduction_array
    operands = (array, array.transpose((1, 2, 0)), array.transpose((2, 0, 1)))
    weights = np.linspace(0.5, 1.5, array.shape[-1])

    def compute():
        if operation == "sum":
            return sparse.elemwise(lambda a, b, c: a + b + c, *operands)
        if operation == "multiply":
            return sparse.elemwise(lambda a, b, c: a * b * c, *operands)
        return sparse.elemwise(lambda a, b, c, w: (a + b + c) * w, *operands, weights)

    compute()

    @benchmark
    def bench():
        compute()
