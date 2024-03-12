def __getattr__(attr_name):
    from sparse.pydata_backend import _settings

    ret = getattr(_settings, attr_name, None)
    if ret is None:
        raise AttributeError(f"module 'sparse._settings' has no attribute {attr_name}")
    return ret
