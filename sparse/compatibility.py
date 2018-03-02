# flake8: noqa F401
import sys

cur_version = sys.version_info

if cur_version < (2, 7):
    raise ImportError('Need at least Python 2.7.')

if cur_version[0] == 3 and cur_version[1] < 5:
    raise ImportError('Need at least Python 3.5 if using Python 3.')

if cur_version[0] == 3:
    from itertools import zip_longest
    from builtins import int, range, zip
else:
    from itertools import izip_longest as zip_longest
    from itertools import izip as zip

    int = long
    range = xrange
