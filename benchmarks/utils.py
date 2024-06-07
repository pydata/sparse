import sparse

from asv_runner.benchmarks.mark import SkipNotImplemented


def skip_if_finch():
    if sparse.backend == sparse.BackendType.Finch:
        raise SkipNotImplemented("Finch backend is skipped.")
