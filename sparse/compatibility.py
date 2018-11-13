# flake8: noqa F401
import sys

if sys.version_info < (2, 7):
    raise ImportError('Need at least Python 2.7.')

if sys.version_info[0] == 3 and sys.version_info[1] < 5:
    raise ImportError('Need at least Python 3.5 if using Python 3.')

if sys.version_info[0] >= 3:
    from itertools import zip_longest
    from builtins import int, range, zip
else:
    from itertools import izip_longest as zip_longest
    from itertools import izip as zip

    int = int
    range = xrange
