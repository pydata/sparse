def __getattr__(attr_name):
    from sparse.pydata_backend import _utils

    ret = getattr(_utils, attr_name, None)
    if ret is None:
        raise AttributeError(f"module 'sparse._compressed' has no attribute {attr_name}")
    return ret
