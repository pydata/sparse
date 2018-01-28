# flake8: noqa F401

try:
    from itertools import izip_longest as zip_longest
except ImportError:
    from itertools import zip_longest

try:
    int = long
except NameError:
    from builtins import int

try:
    range = xrange
except NameError:
    from builtins import range

try:
    from itertools import izip as zip
except ImportError:
    from builtins import zip
