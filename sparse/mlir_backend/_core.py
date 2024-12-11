import ctypes
import ctypes.util
import os
import pathlib
import sys

from mlir_finch.ir import Context
from mlir_finch.passmanager import PassManager

DEBUG = bool(int(os.environ.get("DEBUG", "0")))
CWD = pathlib.Path(".")

finch_lib_path = f"{sys.prefix}/lib/python3.{sys.version_info.minor}/site-packages/lib"

ld_library_path = os.environ.get("LD_LIBRARY_PATH")
ld_library_path = f"{finch_lib_path}:{ld_library_path}" if ld_library_path is None else finch_lib_path
os.environ["LD_LIBRARY_PATH"] = ld_library_path

MLIR_C_RUNNER_UTILS = ctypes.util.find_library("mlir_c_runner_utils")
if os.name == "posix" and MLIR_C_RUNNER_UTILS is not None:
    MLIR_C_RUNNER_UTILS = f"{finch_lib_path}/{MLIR_C_RUNNER_UTILS}"

SHARED_LIBS = []
if MLIR_C_RUNNER_UTILS is not None:
    SHARED_LIBS.append(MLIR_C_RUNNER_UTILS)

libc = ctypes.CDLL(ctypes.util.find_library("c")) if os.name != "nt" else ctypes.cdll.msvcrt
libc.free.argtypes = [ctypes.c_void_p]
libc.free.restype = None

SHARED_LIBS = []
if DEBUG:
    SHARED_LIBS.append(MLIR_C_RUNNER_UTILS)

OPT_LEVEL = 0 if DEBUG else 2

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
