import numpy as np 
from functools import reduce
from operator import mul
import numba

def convert_prep(inds,shape,axisptr):

    inds = [np.array(ind) for ind in inds]
    row_shapes = np.array(shape[:axisptr])
    col_shapes = np.array(shape[axisptr:])
    row_idx_size = np.prod([ind.size for ind in inds[:axisptr]])
    col_idx_size = np.prod([ind.size for ind in inds[axisptr:]])
    row_inds = inds[:axisptr]
    col_inds = inds[axisptr:]
    print(col_inds) 

    if len(row_shapes)==1:
        rows = row_inds[0]
    else:
        rows = np.empty(row_idx_size,dtype=np.intp)
        row_operations = np.prod([ind.size for ind in inds[:axisptr-1]])
        row_key_vals = [int(row_inds[i][0]) for i in range(len(row_inds[:-1]))]
        positions = np.zeros(len(row_shapes)-1,dtype=int)
        rows = convert_to_2d(row_inds,row_key_vals,row_shapes,row_operations,rows,positions)
    if len(col_shapes)==1:
        cols = col_inds[0]
    else:
        cols = np.empty(col_idx_size,dtype=int)
        col_operations = np.prod([ind.size for ind in inds[axisptr:-1]])
        col_key_vals = [int(col_inds[i][0]) for i in range(len(col_inds[:-1]))]
        positions = np.zeros(len(col_shapes)-1,dtype=int)
        cols = convert_to_2d(col_inds,col_key_vals,col_shapes,col_operations,cols,positions)
    return rows,cols

def convert_to_2d(inds,key_vals,shape,operations,indices,positions):
    
    pos = len(key_vals)-1
    increment = 0
    for i in range(operations):
        if key_vals[pos] == inds[pos][-1]:
            key_vals[pos] = inds[pos][0]
            positions[pos] = 0
            pos -= 1
            positions[pos] += 1
        key_vals[pos] = inds[pos][positions[pos]]
        pos = len(key_vals)-1
        positions[pos] +=1
        add = 0
        for count,val in enumerate(key_vals):
            other = np.prod(shape[count:-1])
            add += val * other
        indices[increment:increment+len(inds[-1])] = inds[-1] + add
        increment += len(inds[-1])
    
    return indices

def uncompress_dimension(indptr):
    """converts an index pointer array into an array of coordinates"""
    uncompressed = np.empty(indptr[-1],dtype=np.intp)
    for i in range(len(indptr)-1):
        uncompressed[indptr[i]:indptr[i+1]] = i
    return uncompressed

