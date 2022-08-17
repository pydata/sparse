import sys
import platform

is_64bits = sys.maxsize > 2**32


def pytest_cmdline_preparse(args):
    if platform.system() != "Windows" and not is_64bits:
        args.append("--doctest-modules")
