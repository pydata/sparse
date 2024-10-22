import dataclasses

from ._common import _hold_ref, numpy_to_ranked_memref, ranked_memref_to_numpy
from ._levels import StorageFormat


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

    def _get_storage_format(self) -> StorageFormat:
        return self._storage.get_storage_format()

    def _get_mlir_type(self):
        return self._get_storage_format()._get_mlir_type(shape=self.shape)

    def _to_module_arg(self):
        return self._storage.to_module_arg()

    def copy(self):
        storage_format: StorageFormat = dataclasses.replace(self._get_storage_format(), owns_memory=False)

        fields = self._storage.get__fields_()
        arrs = [ranked_memref_to_numpy(f).copy() for f in fields]
        memrefs = [numpy_to_ranked_memref(arr) for arr in arrs]
        arr = Array(storage=storage_format._get_ctypes_type()(*memrefs), shape=self.shape)
        for carr in arrs:
            _hold_ref(arr, carr)
