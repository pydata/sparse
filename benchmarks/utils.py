import os
import time
from collections.abc import Callable, Iterable
from typing import Any

CI_MODE = bool(int(os.getenv("CI_MODE", default="0")))


def benchmark(
    func: Callable,
    args: Iterable[Any],
    info: str,
    iters: int,
) -> object:
    # Compile
    result = func(*args)

    if CI_MODE:
        print("CI mode - skipping benchmark")
        return result

    # Benchmark
    print(info)
    start = time.time()
    for _ in range(iters):
        func(*args)
    elapsed = time.time() - start
    print(f"Took {elapsed / iters} s.\n")

    return result
