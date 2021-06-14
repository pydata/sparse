import platform
import numpy.distutils.system_info as sysinfo


def pytest_cmdline_preparse(args):
    if platform.system() != "Windows" and sysinfo.platform_bits != 32:
        args.append("--doctest-modules")
