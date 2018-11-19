from .coo import *
from .dok import DOK
from .sparse_array import SparseArray
from .utils import random
from .io import save_npz, load_npz

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import os

_AUTO_DENSIFICATION_ENABLED = bool(int(os.environ.get('SPARSE_AUTO_DENSIFY', '1')))

del os