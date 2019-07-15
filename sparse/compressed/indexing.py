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
from .convert import convert_to_flat,uncompress_dimension


def getitem(x,key):
    """ 
    
    
    """
    from .compressed import GXCS

    key = list(normalize_index(key,x.shape))
    
    # return a single element
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
    compressed_inds = np.zeros(len(x.shape),dtype=np.bool)
    uncompressed_inds = np.zeros(len(x.shape),dtype=np.bool)
    shape_key = np.zeros(len(x.shape),dtype=np.intp)
    
    count = 0
    for i,ind in enumerate(key):
        if isinstance(ind,Integral):
            continue
        elif isinstance(ind,slice):
            shape_key[i] = count
            shape.append(len(range(ind.start,ind.stop,ind.step)))
            if i in x.compressed_axes:
                compressed_inds[i] = True
            else:
                uncompressed_inds[i] = True
        elif isinstance(ind, Iterable):
            shape_key[i] = count
            shape.append(len(ind))
            if i in x.compressed_axes:
                compressed_inds[i] = True
            else:
                uncompressed_inds[i] = True
        
        count += 1

    reordered_key = [key[i] for i in x.axis_order]

    # find starts and ends of rows
    a = x.indptr[:-1].reshape(x.reordered_shape[:x.axisptr])
    b = x.indptr[1:].reshape(x.reordered_shape[:x.axisptr])
    starts, ends = a[reordered_key[:x.axisptr]].flatten(), b[reordered_key[:x.axisptr]].flatten()

    # prepare for converting to flat indices
    for i,ind in enumerate(reordered_key):
        if isinstance(ind,Integral):
            reordered_key[i] = [ind]
        elif isinstance(ind,slice):
            reordered_key[i] = np.arange(ind.start,ind.stop,ind.step)
    
    shape = np.array(shape)
    

    cols = convert_to_flat(reordered_key,x.reordered_shape,x.axisptr)
    
    if np.any(compressed_inds):
        compressed_axes = shape_key[compressed_inds]
        
        if len(compressed_axes)==1:
            row_size = shape[compressed_axes]
        else:
            row_size = np.prod(shape[compressed_axes])     
        
    else: # only uncompressed axes     
        compressed_axes = (0,) # defaults to 0
        row_size = 1 # this doesn't matter
        
    if not np.any(uncompressed_inds): # only indexing compressed axes
        compressed_axes = (0,) # defaults to 0
        row_size = starts.size
        
    indptr = np.empty(row_size+1,dtype=np.intp)
    indptr[0] = 0
    arg = get_array_selection(x.data,x.indices,indptr,starts,ends,cols)
    shape = tuple(shape)
    
    data,indices,indptr = arg
    size = np.prod(shape[1:])

    if not np.any(uncompressed_inds): # only indexing compressed axes
        uncompressed = uncompress_dimension(indptr)
        indices = uncompressed % size
        indptr = np.empty(shape[0]+1,dtype=np.intp)
        np.cumsum(np.bincount(uncompressed//size,minlength=shape[0]),out=indptr[1:])
    if not np.any(compressed_inds):
        uncompressed = indices // size
        indptr = np.empty(shape[0]+1,dtype=np.intp)
        np.cumsum(np.bincount(uncompressed,minlength=shape[0]),out=indptr[1:])
        indices = indices % size
    
    arg = (data,indices,indptr)
    
    return GXCS(arg,shape=shape,compressed_axes=compressed_axes,fill_value=x.fill_value)


@numba.jit(nopython=True,nogil=True)
def get_array_selection(arr_data,arr_indices,indptr,starts,ends,col):
    """
    This is a very general algorithm to be used when more optimized methods don't apply. 
    It performs a binary search for each of the requested elements. 
    Consequently it roughly scales by O(n log nnz per row) where n is the number of requested elements and
    nnz per row is the number of nonzero elements in that row.
    """
    indices = []
    ind_list = []
    for i,(start,end) in enumerate(zip(starts,ends)):
        inds = []
        current_row = arr_indices[start:end]
        if len(current_row) == 0:
            indptr[i+1] = indptr[i]
            continue
        for c in range(len(col)):
            s = np.searchsorted(current_row,col[c]) 
            if not (s >= current_row.size or current_row[s] != col[c]):
                s += start
                inds.append(s)
                indices.append(c)
        ind_list.extend(inds)
        indptr[i+1] = indptr[i] + len(inds)
    ind_list = np.array(ind_list,dtype=np.int64)
    indices = np.array(indices) 
    data = arr_data[ind_list]
    return (data,indices,indptr)

