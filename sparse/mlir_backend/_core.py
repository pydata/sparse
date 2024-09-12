import ctypes
import ctypes.util
import os
import pathlib

from mlir.ir import Context
from mlir.passmanager import PassManager

DEBUG = bool(int(os.environ.get("DEBUG", "0")))
CWD = pathlib.Path(".")

MLIR_C_RUNNER_UTILS = ctypes.util.find_library("mlir_c_runner_utils")
libc = ctypes.CDLL(ctypes.util.find_library("c")) if os.name != "nt" else ctypes.cdll.msvcrt
libc.free.argtypes = [ctypes.c_void_p]
libc.free.restype = None

# TODO: remove global state
ctx = Context()

pm = PassManager.parse(
    "builtin.module(sparse-assembler{direct-out=true}, sparsifier{create-sparse-deallocs=1})",
    context=ctx,
)
