import numpy as np 
from functools import reduce
from operator import mul
import numba

def convert_prep(inds,shape):

    inds = [np.array(ind) for ind in inds]
    midpoint = len(shape)//2
    midpoint = midpoint + 1 if len(shape)==3 else midpoint
    row_shapes = np.array(shape[:midpoint])
    col_shapes = np.array(shape[midpoint:])
    row_idx_size = np.prod([ind.size for ind in inds[:midpoint]])
    col_idx_size = np.prod([ind.size for ind in inds[midpoint:]])
    row_inds = inds[:midpoint]
    col_inds = inds[midpoint:]

    rows = np.empty(row_idx_size,dtype=int)
    row_operations = np.prod([ind.size for ind in inds[:midpoint-1]])
    print(row_operations) 
    row_key_vals = [int(row_inds[i][0]) for i in range(len(row_inds[:-1]))]
    j = np.zeros(len(row_shapes)-1,dtype=int)
    print('row j: ', j)
    rows = convert_to_2d(row_inds,row_key_vals,row_shapes,row_operations,rows,j)

    if len(col_shapes)==1:
        cols = col_inds[0]
    else:
        cols = np.empty(col_idx_size,dtype=int)
        col_operations = np.prod([ind.size for ind in inds[midpoint:-1]])
        col_key_vals = [int(col_inds[i][0]) for i in range(len(col_inds[:-1]))]
        j = np.zeros(len(col_shapes)-1,dtype=int)
        print('col j: ', j)
        cols = convert_to_2d(col_inds,col_key_vals,col_shapes,col_operations,cols,j)
        print('motherfucker I am back')
    print('rows: ',rows)
    print('cols: ',cols)
    return rows,cols

#@numba.jit(nopython=True,nogil=True)
def convert_to_2d(inds,key_vals,shape,operations,indices,j):
    
    pos = len(key_vals)-1
    print(pos)
    increment = 0
    for i in range(operations):
        if key_vals[pos] == inds[pos][-1]:
            key_vals[pos] = inds[pos][0]
            j[pos] = 0
            pos -= 1
            j[pos] += 1
        key_vals[pos] = inds[pos][j[pos]]
        pos = len(key_vals)-1
        j[pos] +=1
        add = 0
        for count,val in enumerate(key_vals):
            other = np.prod(shape[count:-1])
            add += val * other
        indices[increment:increment+len(inds[-1])] = inds[-1] + add
        increment += len(inds[-1])
    
    return indices

def uncompress_dimension(indptr,indices):
    """converts an index pointer array into an array of coordinates"""
    uncompressed = np.empty(indices.shape[0],dtype=np.intp)
    position = 0
    for i in range(len(indptr)-1):
        inds = indices[indptr[i]:indptr[i+1]].shape[0] 
        uncompressed[np.arange(position,inds + position)] = i
        position += inds
    return uncompressed

