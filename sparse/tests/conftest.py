import platform
import sys

import pytest

is_64bits = sys.maxsize > 2**32


def pytest_cmdline_preparse(args):
    if platform.system() != "Windows" and not is_64bits:
        args.append("--doctest-modules")


@pytest.fixture(scope="session")
def rng():
    from sparse._utils import default_rng

    return default_rng
