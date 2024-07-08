import os
import time
from collections.abc import Callable, Iterable
from typing import Any

CI_MODE = os.getenv("CI_MODE", default=False)


def benchmark(func: Callable, args: Iterable[Any], info: str, iters: int):
    if CI_MODE:
        print("CI mode - skipping benchmark")
        return

    print(info)
    start = time.time()
    for _ in range(iters):
        func(*args)
    elapsed = time.time() - start
    print(f"Took {elapsed / iters} s.\n")
