import numpy as np 
import numba



@numba.jit(nopython=True,nogil=True)
def csr_row_array_col_array(arr_data,arr_indices,arr_indptr,indptr,row,col):
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


@numba.jit(nopython=True,nogil=True)
def csr_full_col_slices(arr_data,arr_indices,arr_indptr,indptr,row):
    """
    This algorithm is used for when all column dimensions are full slices with a step of one.
    It might be worth it to make two passes over the array and use static arrays instead of lists. 
    """
    indices = []
    data = []
    for i,r in enumerate(row,1):
        indices.extend(arr_indices[arr_indptr[r]:arr_indptr[r+1]])
        data.extend(arr_data[arr_indptr[r]:arr_indptr[r+1]])
        indptr[i] = indptr[i-1] + len(arr_indices[arr_indptr[r]:arr_indptr[r+1]])
    data = np.array(data)
    indices = np.array(indices)
    return (data,indices,indptr)

@numba.jit(nopython=True,nogil=True)
def csr_partial_col_slices(arr_data,arr_indices,arr_indptr,indptr,row,col_start,col_stop):
    """
    This algorithm is used for partial column slices with a step of one. It is currently only used for 2d arrays.
    """
    indices = []
    data = []
    for i,r in enumerate(row,1):
        start = np.searchsorted(arr_indices[arr_indptr[r]:arr_indptr[r+1]],col_start) + arr_indptr[r]
        stop = np.searchsorted(arr_indices[arr_indptr[r]:arr_indptr[r+1]],col_stop) + arr_indptr[r]
        inds = arr_indices[start:stop] - col_stop
        indices.extend(inds)
        data.extend(arr_data[start:stop])
        indptr[i] = indptr[i-1] + inds.size
    data = np.array(data)
    indices = np.array(indices)
    return (data,indices,indptr)


