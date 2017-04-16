import pytest

import random
import numpy as np
from sparse import COO


x = np.empty(shape=(2, 3, 4), dtype=np.float32)
for i in range(10):
    x[random.randint(0, x.shape[0] - 1),
      random.randint(0, x.shape[1] - 1),
      random.randint(0, x.shape[2] - 1)] = random.randint(0, 100)
y = COO.from_numpy(x)


@pytest.mark.parametrize('axis', [None, 0, 1, 2, (0, 2)])
def test_reductions(axis):
    xx = x.sum(axis=axis)
    yy = y.sum(axis=axis)
    assert xx.shape == yy.shape
    assert xx.dtype == yy.dtype
    assert np.allclose(xx, yy)


@pytest.mark.parametrize('axis', [None, (1, 2, 0), (2, 1, 0), (0, 1, 2)])
def test_transpose(axis):
    xx = x.transpose(axis)
    yy = y.transpose(axis)
    assert xx.shape == yy.shape
    assert xx.dtype == yy.dtype
    assert np.allclose(xx, yy)
