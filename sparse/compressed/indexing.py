import numpy as np 
from ..slicing import normalize_index
from ..coo.common import linear_loc
from ..coo.core import COO
from functools import reduce
from operator import mul
from .convert import convert_prep,uncompress_dimension
from .csr_indexing import csr_row_array_col_array, csr_full_col_slices, csr_partial_col_slices
from .csc_indexing import csc_row_array_col_array

def getitem(x,key):
    """ 
    Because half of the dimensions represent the rows and the other half represent the columns 
    of the underlying csr/csc matrix, there are issues related to getting the correct coordinates
    of the new matrix. For example, if x has a shape of (10,10,10,10) and we want to access x[:5,0,:8,:6]
    than the resulting shape will be (40,6). However, the axis we're drawing :8 from originally stored columns
    and now it's storing rows. As a result we need to transform some of the resulting coordinates. This is what the 
    manage_csr_output and manage_csc_output functions are for. I'm still working some issues out on the CSC end, but CSR
    should be mostly correct except for when trying to index a 1d array.
    
    """

    key = list(normalize_index(key,x.shape))
        
    if all(isinstance(k, int) for k in key):
        inds = np.ravel_multi_index(key,x.shape)
        row, col = np.unravel_index(inds,x.compressed_shape)
        if x.format is 'CSR':
            item = np.searchsorted(x.indices[x.indptr[row]:x.indptr[row+1]],col) + x.indptr[row]
            if x.indices[item]==col:
                return x.data[item]
        else:
            item = np.searchsorted(x.indices[x.indptr[col]:x.indptr[col+1]],row) + x.indptr[col]
            if x.indices[item]==row:
                return x.data[item]
        return x.fill_value

    midpoint = int(len(x.shape)//2) # in this case midpoint means first col axis
    midpoint = midpoint + 1 if len(x.shape) % 2 == 1 else midpoint
    if len(x.shape)==3:
        midpoint = 2
    shape = []
    row_inds = 0
    col_inds = 0
    
    for i,ind in enumerate(key):
        if isinstance(ind,int):
            continue
        elif isinstance(ind,slice):
            shape.append(len(range(ind.start,ind.stop,ind.step)))
            if i >= midpoint:
                col_inds += 1
            else:
                row_inds += 1
        elif isinstance(ind, np.ndarray):
            shape.append(ind.shape[0])
            if i >= midpoint:
                col_inds += 1
            else:
                row_inds += 1
        
    shape = tuple(shape)

    
    for k in range(len(key)):
        if isinstance(key[k],int):
            key[k] = [key[k]]
        elif isinstance(key[k],slice):
            key[k] = range(key[k].start,key[k].stop,key[k].step)
    
    if len(x.shape)==2: # no need for coordinate conversion if x is already 2d
        rows,cols = np.array(key[0]),key[1]
    else:
        rows,cols = convert_prep(key,x.shape)
        
    
    
    if x.format is 'CSR':
        indptr = np.zeros(len(rows)+1,dtype=int)
        if check_full_slices(key,x.shape,midpoint): # check for row optimized methods
            arg = csr_full_col_slices(x.data,x.indices,x.indptr,indptr,rows)
        elif len(x.shape)==2 and isinstance(cols,range) and cols.step==1:
            arg = csr_partial_col_slices(x.data,x.indices,x.indptr,indptr,rows,cols.start,cols.stop)
        else:    
            arg = csr_row_array_col_array(x.data,x.indices,x.indptr,indptr,rows,cols)
        arg = manage_csr_output(x,arg,shape,row_inds,col_inds)
    
    else: # still working on the CSC-specific indexing routines
        arg = csc_row_array_col_array(x,rows,cols)
        arg = manage_csc_output(x,arg,shape,row_inds,col_inds)
    return x.__class__(arg,shape=shape,fill_value=x.fill_value)


def check_full_slices(key,shape,midpoint):
    i = 0
    for k,s in zip(key[midpoint:],shape[midpoint:]):
        if isinstance(k,range):
            if k.start==0 and k.stop==s and k.step==1:
                i+=1
    if i==len(key[midpoint:]):
        return True
    return False

def manage_csr_output(x,arg,shape,row_inds,col_inds):
    sl = len(shape)
    row_size = np.prod(shape[:sl//2+1]) if sl%2==1 else np.prod(shape[:sl//2])
    col_size = np.prod(shape[sl//2+1:]) if sl%2==1 else np.prod(shape[sl//2:])
    midpoint = int(len(shape)//2)
    data,indices,indptr = arg
    
    if col_inds==0:
        uncompressed = uncompress_dimension(indptr,indices)
        indices = uncompressed%shape[-1]
        indptr = np.zeros(row_size+1,dtype=int)
        np.cumsum(np.bincount(uncompressed//shape[-1], minlength=row_size), out=indptr[1:])
        arg = (data,indices,indptr)
    elif row_inds==0:
        indptr = np.zeros(row_size+1,dtype=int)
        np.cumsum(np.bincount(indices//shape[-1], minlength=row_size), out=indptr[1:])
        indices = indices % shape[-1]
        arg = (data,indices,indptr)
    elif col_inds > row_inds: # r + c + c ##########################################
        uncompressed = indices//shape[-1]
        for i in range(1,len(indptr)-1):
            uncompressed[indptr[i]:] += shape[midpoint]
        indptr = np.zeros(row_size+1,dtype=int)
        np.cumsum(np.bincount(uncompressed, minlength=row_size), out=indptr[1:])
        indices = indices % shape[-1]
        arg = (data,indices,indptr)
    elif row_inds - col_inds > 1: # r + r + r + c
        uncompressed = uncompress_dimension(indptr,indices) 
        indptr = np.zeros(row_size+1,dtype=int)
        np.cumsum(np.bincount(uncompressed // shape[midpoint], minlength=row_size), out=indptr[1:])
        uncompressed = (uncompressed % shape[midpoint]) * shape[-1]
        for i in range(len(indptr)-1):
            indices[indptr[i]:indptr[i+1]] = uncompressed[indptr[i]:indptr[i+1]] + indices[indptr[i]:indptr[i+1]]
        arg = (data,indices,indptr)
    return arg



def manage_csc_output(x,arg,shape,row_inds,col_inds):
    sl = len(shape)
    row_size = np.prod(shape[:sl//2+1]) if sl%2==1 else np.prod(shape[:sl//2])
    col_size = np.prod(shape[sl//2+1:]) if sl%2==1 else np.prod(shape[sl//2:])
    midpoint = int(len(shape)//2)
    data,indices,indptr = arg
    if col_inds==0:
        order = np.argsort(indices%shape[-1])
        indptr = np.zeros(col_size+1,dtype=int)
        np.cumsum(np.bincount(indices[order] % shape[-1], minlength=col_size), out=indptr[1:])
        indices = indices[order]//shape[-1]
        data = data[order]
        arg = (data,indices,indptr)
        
    elif row_inds==0:
        uncompressed = uncompress_dimension(indptr,indices)
        order = np.argsort(uncompressed % shape[-1])
        indices = uncompressed[order]//shape[-1]
        indptr = np.zeros(col_size+1,dtype=int)
        np.cumsum(np.bincount(uncompressed % shape[-1],minlength=col_size),out=indptr[1:])
        data = data[order]
        arg = (data,indices,indptr)
    elif col_inds > row_inds: # r + c + c ##########################################
        raise NotImplementedError('The data returns in the wrong order. Still working on this issue')
        uncompressed = uncompress_dimension(indptr,indices)
        coords = COO(np.vstack((indices,uncompressed)),data=data).reshape((row_size,col_size)).coords
        uncompressed = None
        indptr = np.zeros(col_size+1,dtype=int)
        np.cumsum(np.bincount(coords[1],minlength=col_size),out=indptr[1:])
        linear = linear_loc(coords[[1,0]],(col_size,row_size))
        order = np.argsort(linear)
        indices = coords[0,order]
        #data = data[order]
        arg = (data,indices,indptr)
    
    elif row_inds - col_inds > 1: # r + r + r + c
        raise NotImplementedError('This is not finished yet')
    #    print('multiple row interactions')
    #    uncompressed = uncompress_dimension(indptr,indices) 
    #    indptr = compress_dimension(uncompressed // shape[midpoint],np.empty(col_size+1,dtype=int))
    #    uncompressed = (uncompressed % shape[midpoint]) * shape[-1]
    #    for i in range(len(indptr)-1):
    #        indices[indptr[i]:indptr[i+1]] = uncompressed[indptr[i]:indptr[i+1]] + indices[indptr[i]:indptr[i+1]]
    #    arg = (data,indices,indptr)
    return arg
