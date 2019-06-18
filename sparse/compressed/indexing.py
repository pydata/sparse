import numpy as np
import numba
import copy
from numbers import Integral
from collections.abc import Iterable 
from ..slicing import normalize_index
from ..coo.common import linear_loc
from ..coo.core import COO
from functools import reduce
from operator import mul
from .convert import convert_prep,uncompress_dimension


def getitem(x,key):
    """ 
    
    
    """
    from .compressed import CSD

    key = list(normalize_index(key,x.shape))
        
    if all(isinstance(k, int) for k in key): # indexing for a single element
        key = np.array(key)[x.axis_order] # reordering the input
        ind = np.ravel_multi_index(key,x.reordered_shape)
        row, col = np.unravel_index(ind,x.compressed_shape)
        current_row = x.indices[x.indptr[row]:x.indptr[row+1]]
        item = np.searchsorted(current_row,col)
        if not (item >= current_row.size or current_row[item] != col):
            item += x.indptr[row]
            return x.data[item]
        return x.fill_value

    shape = []
    compressed_inds = 0
    uncompressed_inds = 0
    reordered_key = copy.copy(key)
    for i,ind in enumerate(key):
        if isinstance(ind,Integral):
            continue
        elif isinstance(ind,slice):
            shape.append(len(range(ind.start,ind.stop,ind.step)))
            if i in x.compressed_axes:
                compressed_inds += 1
            else:
                uncompressed_inds += 1
        elif isinstance(ind, Iterable):
            shape.append(len(ind))
            if i in x.compressed_axes:
                compressed_inds += 1
            else:
                uncompressed_inds += 1
        reordered_key[x.axis_order[i]] = key[i] # reorder the key

    for i,ind in enumerate(reordered_key):
        if isinstance(ind,Integral):
            reordered_key[i] = [ind]
        elif isinstance(ind,slice):
            reordered_key[i] = np.arange(ind.start,ind.stop,ind.step)

    shape = tuple(shape)

    # this is all temporary and currently doesn't work
    rows,cols = convert_prep(reordered_key,x.reordered_shape,x.axisptr)
    reordered_shape = np.array(shape)[x.axis_order] 
    indptr = np.empty(np.prod(reordered_shape[x.axisptr])+1,dtype=np.intp)
    indptr[0] = 0
    arg = get_array_selection(x.data,x.indices,x.indptr,indptr,rows,cols) 
    return CSD(arg,shape=shape,compressed_axes=x.compressed_axes,fill_value=x.fill_value)


@numba.jit(nopython=True,nogil=True)
def get_array_selection(arr_data,arr_indices,arr_indptr,indptr,row,col):
    """
    This is a very general algorithm to be used when more optimized methods don't apply. 
    It performs a binary search for each of the requested elements. 
    Consequently it roughly scales by O(n log nnz per row) where n is the number of requested elements and
    nnz per row is the number of nonzero elements in that row.
    """
    indices = []
    ind_list = []
    for i,r in enumerate(row):
        inds = []
        current_row = arr_indices[arr_indptr[r]:arr_indptr[r+1]]
        if len(current_row) == 0:
            indptr[i+1] = indptr[i]
            continue
        for c in range(len(col)):
            s = np.searchsorted(current_row,col[c]) 
            if not (s >= current_row.size or current_row[s] != col[c]):
                s += arr_indptr[r]
                inds.append(s)
                indices.append(c)
        ind_list.extend(inds)
        indptr[i+1] = indptr[i] + len(inds)
    ind_list = np.array(ind_list,dtype=np.int64)
    indices = np.array(indices) 
    data = arr_data[ind_list]
    return (data,indices,indptr)


