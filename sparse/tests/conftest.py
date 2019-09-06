import platform


def pytest_cmdline_preparse(args):
    if platform.system() != "Windows":
        args.append("--doctest-modules")
