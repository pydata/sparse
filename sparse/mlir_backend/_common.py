import abc

from mlir import ir


class MlirType(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_mlir_type(cls) -> ir.Type: ...
