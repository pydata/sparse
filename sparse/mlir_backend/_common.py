import abc
import ctypes
import functools
import weakref

from mlir import ir


class MlirType(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_mlir_type(cls) -> ir.Type: ...


def fn_cache(f, maxsize: int | None = None):
    return functools.wraps(f)(functools.lru_cache(maxsize=maxsize)(f))


def _hold_self_ref_in_ret(fn):
    @functools.wraps(fn)
    def wrapped(self, *a, **kw):
        ret = fn(self, *a, **kw)
        _take_owneship(ret, self)
        return ret

    return wrapped


def _take_owneship(owner, obj):
    ptr = ctypes.py_object(obj)
    ctypes.pythonapi.Py_IncRef(ptr)

    def finalizer(ptr):
        ctypes.pythonapi.Py_DecRef(ptr)

    weakref.finalize(owner, finalizer, ptr)
