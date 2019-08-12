from .core import COO, as_coo
from .umath import elemwise
from .common import (tensordot, dot, matmul, concatenate, stack, triu, tril, where,
                     nansum, nanmean, nanprod, nanmin, nanmax, nanreduce, roll,
                     eye, full, full_like, zeros, zeros_like, ones, ones_like,
                     kron, argwhere, isposinf, isneginf)

# Allow applying np.result_type to a COO through __array_function__
from numpy.core._multiarray_umath import result_type

__all__ = ['COO', 'as_coo', 'elemwise', 'tensordot', 'dot', 'matmul', 'concatenate', 'stack', 'triu', 'tril', 'where', 'nansum', 'nanmean',
           'nanprod', 'nanmin', 'nanmax', 'nanreduce', 'roll', 'eye', 'full', 'full_like', 'zeros', 'zeros_like', 'ones', 'ones_like', 'kron', 'argwhere',
           'isposinf', 'isneginf', 'result_type']
