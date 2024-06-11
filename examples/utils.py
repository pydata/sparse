import time
from collections.abc import Callable, Iterable
from typing import Any


def benchmark(func: Callable, args: Iterable[Any], info: str, iters: int):
    print(info)
    start = time.time()
    for _ in range(iters):
        func(*args)
    elapsed = time.time() - start
    print(f"Took {elapsed / iters} s.\n")
