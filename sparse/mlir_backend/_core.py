import ctypes
import ctypes.util
import os
import pathlib
import sys

from mlir_finch.ir import Context
from mlir_finch.passmanager import PassManager

DEBUG = bool(int(os.environ.get("DEBUG", "0")))
CWD = pathlib.Path(".")

LD_ENV_PATH = f"{sys.prefix}/lib/python3.10/site-packages/lib"

if "LD_LIBRARY_PATH" in os.environ:
    os.environ["LD_LIBRARY_PATH"] = f"{LD_ENV_PATH}:{os.environ['LD_LIBRARY_PATH']}"
else:
    os.environ["LD_LIBRARY_PATH"] = LD_ENV_PATH

MLIR_C_RUNNER_UTILS = ctypes.util.find_library("mlir_c_runner_utils")
if os.name == "posix":
    MLIR_C_RUNNER_UTILS = f"{LD_ENV_PATH}/{MLIR_C_RUNNER_UTILS}"
libc = ctypes.CDLL(ctypes.util.find_library("c")) if os.name != "nt" else ctypes.cdll.msvcrt
libc.free.argtypes = [ctypes.c_void_p]
libc.free.restype = None

# TODO: remove global state
ctx = Context()

pm = PassManager.parse(
    """
    builtin.module(
        sparse-assembler{direct-out=true},
        sparsifier{create-sparse-deallocs=1 enable-runtime-library=false}
    )
    """,
    context=ctx,
)
