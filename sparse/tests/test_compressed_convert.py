import sparse
import pytest
import numpy as np

import sparse._compressed.convert as convert
from sparse._utils import assert_eq


# @pytest.mark.parametrize(
# )
def test_convert_to_flat():
    ...


# @pytest.mark.parametrize(
# )
def test_compute_flat():
    ...
    # assert_eq(x[index], s[index])


@pytest.mark.parametrize(
    "shape, expected_shape",
    [
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
    assert_eq(convert.transform_shape(np.asarray(shape)), expected_shape)
