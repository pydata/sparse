import os

import numpy as np

AUTO_DENSIFY = bool(int(os.environ.get("SPARSE_AUTO_DENSIFY", "0")))
WARN_ON_TOO_DENSE = bool(int(os.environ.get("SPARSE_WARN_ON_TOO_DENSE", "0")))


def _is_nep18_enabled():
    class A:
        def __array_function__(self, *args, **kwargs):
            return True

    try:
        return np.concatenate([A()])
    except ValueError:
        return False


NEP18_ENABLED = _is_nep18_enabled()
