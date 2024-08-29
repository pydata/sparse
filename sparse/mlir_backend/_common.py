import abc
import functools

from mlir import ir


class MlirType(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_mlir_type(cls) -> ir.Type: ...


def fn_cache(f, maxsize: int | None = None):
    return functools.wraps(f)(functools.lru_cache(maxsize=maxsize)(f))
