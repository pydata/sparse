from sparse.numba_backend._compressed import convert
from sparse.numba_backend._utils import assert_eq

import pytest
from numba.typed import List

import numpy as np


def make_inds(shape):
    return [np.arange(1, a - 1) for a in shape]


def make_increments(shape):
    inds = make_inds(shape)
    shape_bins = convert.transform_shape(np.asarray(shape))
    return List([inds[i] * shape_bins[i] for i in range(len(shape))])


@pytest.mark.parametrize(
    "shape, expected_subsample, subsample",
    [
        [(5, 6, 7, 8, 9), np.array([3610, 6892, 10338]), 1000],
        [(13, 12, 12, 9, 7), np.array([9899, 34441, 60635, 86703]), 10000],
        [
            (12, 15, 7, 14, 9),
            np.array([14248, 36806, 61382, 85956, 110532, 135106]),
            10000,
        ],
        [(9, 9, 12, 7, 12), np.array([10177, 34369, 60577]), 10000],
    ],
)
def test_convert_to_flat(shape, expected_subsample, subsample):
    inds = make_inds(shape)
    dtype = inds[0].dtype

    assert_eq(
        convert.convert_to_flat(inds, shape, dtype)[::subsample],
        expected_subsample.astype(dtype),
    )


@pytest.mark.parametrize(
    "shape, expected_subsample, subsample",
    [
        [(5, 6, 7, 8, 9), np.array([3610, 6892, 10338]), 1000],
        [(13, 12, 12, 9, 7), np.array([9899, 34441, 60635, 86703]), 10000],
        [
            (12, 15, 7, 14, 9),
            np.array([14248, 36806, 61382, 85956, 110532, 135106]),
            10000,
        ],
        [(9, 9, 12, 7, 12), np.array([10177, 34369, 60577]), 10000],
    ],
)
def test_compute_flat(shape, expected_subsample, subsample):
    increments = make_increments(shape)
    dtype = increments[0].dtype
    operations = np.prod([inc.shape[0] for inc in increments[:-1]], dtype=dtype)
    cols = np.tile(increments[-1], operations)

    assert_eq(
        convert.compute_flat(increments, cols, operations)[::subsample],
        expected_subsample.astype(dtype),
    )


@pytest.mark.parametrize(
    "shape, expected_shape",
    [
        [(5, 6, 7, 8, 9), np.array([3024, 504, 72, 9, 1])],
        [(13, 12, 12, 9, 7), np.array([9072, 756, 63, 7, 1])],
        [(12, 15, 7, 14, 9), np.array([13230, 882, 126, 9, 1])],
        [
            (18, 5, 12, 14, 9, 11, 8, 14),
            np.array([9313920, 1862784, 155232, 11088, 1232, 112, 14, 1]),
        ],
        [
            (11, 6, 13, 11, 17, 7, 15),
            np.array([1531530, 255255, 19635, 1785, 105, 15, 1]),
        ],
        [(9, 9, 12, 7, 12), np.array([9072, 1008, 84, 12, 1])],
    ],
)
def test_transform_shape(shape, expected_shape):
    assert_eq(convert.transform_shape(np.asarray(shape)), expected_shape, compare_dtype=False)
