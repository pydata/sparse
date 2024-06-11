import numpy as np

from ._common import _check_device

__all__ = [
    "capabilities",
    "default_device",
    "default_dtypes",
    "devices",
    "dtypes",
]

_CAPABILITIES = {
    "boolean indexing": True,
    "data-dependent shapes": True,
}

_DEFAULT_DTYPES = {
    "cpu": {
        "real floating": np.dtype(np.float64),
        "complex floating": np.dtype(np.complex128),
        "integral": np.dtype(np.int64),
        "indexing": np.dtype(np.int64),
    }
}


def _get_dtypes_with_prefix(prefix: str):
    out = set()
    for a in np.__all__:
        if not a.startswith(prefix):
            continue
        try:
            dt = np.dtype(getattr(np, a))
            out.add(dt)
        except (ValueError, TypeError, AttributeError):
            pass
    return sorted(out)


_DTYPES = {
    "cpu": {
        "bool": [np.bool_],
        "signed integer": _get_dtypes_with_prefix("int"),
        "unsigned integer": _get_dtypes_with_prefix("uint"),
        "real floating": _get_dtypes_with_prefix("float"),
        "complex floating": _get_dtypes_with_prefix("complex"),
    }
}

for _dtdict in _DTYPES.values():
    _dtdict["integral"] = _dtdict["signed integer"] + _dtdict["unsigned integer"]
    _dtdict["numeric"] = _dtdict["integral"] + _dtdict["real floating"] + _dtdict["complex floating"]

del _dtdict


def capabilities():
    return _CAPABILITIES


def default_device():
    return "cpu"


@_check_device
def default_dtypes(*, device=None):
    if device is None:
        device = default_device()
    return _DEFAULT_DTYPES[device]


def devices():
    return ["cpu"]


@_check_device
def dtypes(*, device=None, kind=None):
    if device is None:
        device = default_device()

    device_dtypes = _DTYPES[device]

    if kind is None:
        return device_dtypes

    if isinstance(kind, str):
        return device_dtypes[kind]

    out = {}

    for k in kind:
        out[k] = device_dtypes[k]

    return out
