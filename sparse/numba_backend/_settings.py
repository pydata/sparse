import os

import numpy as np

AUTO_DENSIFY = bool(int(os.environ.get("SPARSE_AUTO_DENSIFY", "0")))
WARN_ON_TOO_DENSE = bool(int(os.environ.get("SPARSE_WARN_ON_TOO_DENSE", "0")))
IS_NUMPY2 = np.lib.NumpyVersion(np.__version__) >= "2.0.0a1"


def _is_nep18_enabled():
    class A:
        def __array_function__(self, *args, **kwargs):
            return True

    try:
        return np.concatenate([A()])
    except ValueError:
        return False


NEP18_ENABLED = _is_nep18_enabled()


class ArrayNamespaceInfo:
    def __init__(self):
        self.np_info = np.__array_namespace_info__()

    def capabilities(self):
        np_capabilities = self.np_info.capabilities()
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
            "max dimensions": np_capabilities.get("max dimensions", 64) - 1,
        }

    def default_device(self):
        return self.np_info.default_device()

    def default_dtypes(self, *, device=None):
        return self.np_info.default_dtypes(device=device)

    def devices(self):
        return self.np_info.devices()

    def dtypes(self, *, device=None, kind=None):
        return self.np_info.dtypes(device=device, kind=kind)


def __array_namespace_info__() -> ArrayNamespaceInfo:
    return ArrayNamespaceInfo()
