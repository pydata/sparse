import numpy as np

from .levels import StorageFormat


class Array:
    def __init__(self, *, storage, shape: tuple[int, ...]) -> None:
        storage_rank = storage.get_storage_format().rank
        if len(shape) != storage_rank:
            raise ValueError(f"Mismatched rank, `{storage_rank=}`, `{shape=}`")

        self._storage = storage
        self._shape = shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self):
        return self._storage.get_storage_format().dtype

    @property
    def format(self) -> StorageFormat:
        return self._storage.get_storage_format()

    def _get_mlir_type(self):
        return self.format._get_mlir_type(shape=self.shape)

    def _to_module_arg(self):
        return self._storage.to_module_arg()

    def copy(self):
        from ._conversions import from_constituent_arrays

        arrs = tuple(arr.copy() for arr in self.get_constituent_arrays())
        return from_constituent_arrays(format=self.format, arrays=arrs, shape=self.shape)

    def get_constituent_arrays(self) -> tuple[np.ndarray, ...]:
        return self._storage.get_constituent_arrays()
