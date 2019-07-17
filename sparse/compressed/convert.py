import numpy as np 
import numba

def convert_to_flat(inds,shape,axisptr):

    inds = [np.array(ind) for ind in inds]
    col_shapes = np.array(shape[axisptr:])
    col_idx_size = np.prod([ind.size for ind in inds[axisptr:]])
    col_inds = inds[axisptr:]
    if len(col_inds)==1:
        return col_inds[0]
    cols = np.empty(col_idx_size,dtype=int)
    col_operations = np.prod([ind.size for ind in inds[axisptr:-1]]) if len(inds[axisptr:]) > 1 else 1
    col_key_vals = [int(col_inds[i][0]) for i in range(len(col_inds[:-1]))] if len(col_inds) > 1 else [int(col_inds[0][0])]
    positions = np.zeros(len(col_shapes)-1,dtype=int)
    cols = convert_to_2d(col_inds,col_key_vals,col_shapes,col_operations,cols,positions)
    return cols

numba.jit(nopython=True,nogil=True)
def convert_to_2d(inds,key_vals,shape,operations,indices,positions):
    
    pos = len(key_vals)-1
    increment = 0
        
    for i in range(operations):
        if i != 0 and key_vals[pos] == inds[pos][-1]:
            key_vals[pos] = inds[pos][0]
            positions[pos] = 0
            pos -= 1
            positions[pos] += 1
        key_vals[pos] = inds[pos][positions[pos]]
        pos = len(key_vals)-1
        positions[pos] +=1
        linearized = np.ravel_multi_index(key_vals+[inds[-1][0]],shape)
        indices[increment:increment+len(inds[-1])] = inds[-1] + linearized
        increment += len(inds[-1])
    
    return indices

@numba.jit(nopython=True,nogil=True)
def uncompress_dimension(indptr):
    """converts an index pointer array into an array of coordinates"""
    uncompressed = np.empty(indptr[-1],dtype=np.intp)
    for i in range(len(indptr)-1):
        uncompressed[indptr[i]:indptr[i+1]] = i
    return uncompressed

