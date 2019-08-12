import sparse
from sparse.utils import assert_eq
import os
import pytest
np = pytest.importorskip('numpy', minversion='1.16')


env_name = "NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"


@pytest.mark.skipif(env_name not in os.environ or os.environ[env_name] != "1",
                    reason=env_name + " undefined or disabled")
@pytest.mark.parametrize('func', [
    lambda x: np.dot(x, x),
    lambda x: np.mean(x),
    lambda x: np.std(x),
    lambda x: np.var(x),
    lambda x: np.sum(x),
    lambda x: np.sum(x, axis=0),
    lambda x: np.stack([x, x]),
    lambda x: np.tensordot(x, x),
    lambda x: np.transpose(x),
    lambda x: np.where(x > np.transpose(x), x, np.transpose(x)),
    lambda x: np.matmul(x, x)
])
def test_array_function(func):
    x = sparse.random((50, 50), density=.25)
    y = x.todense()
    xx = func(x)
    yy = func(y)
    assert_eq(xx, yy)
