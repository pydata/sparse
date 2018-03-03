import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from numpy cimport uint64_t

cdef extern from "<cstring>" namespace "std":
    void* memcpy (void* destination, const void* source, size_t num)


def cywhere(np.ndarray a not None, np.ndarray b not None):
    a = a.astype(dtype=np.uint64)
    b = b.astype(dtype=np.uint64)

    a_ind, b_ind = cywhere_wrapped(a, b)

    a_ind = a_ind.astype(np.min_scalar_type(a.shape[0]))
    b_ind = b_ind.astype(np.min_scalar_type(b.shape[0]))

    return a_ind, b_ind

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cywhere_wrapped(uint64_t[:] a, uint64_t[:] b):
    cdef vector[size_t] a_ind = vector[size_t]()
    cdef vector[size_t] b_ind = vector[size_t]()
    cdef size_t na = a.shape[0], nb = b.shape[0]
    cdef size_t ia, ib = 0
    cdef uint64_t j
    cdef uint64_t match = 0

    with nogil:
        for ia in range(na):
            j = a[ia]
            if j == b[match]:
                ib = match
            while ib < nb and j >= b[ib]:
                if j == b[ib]:
                    a_ind.push_back(ia)
                    b_ind.push_back(ib)
                    if b[match] < b[ib]:
                        match = ib

                ib += 1

    return vector_to_np(a_ind), vector_to_np(b_ind)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector_to_np(vector[size_t]& vec):
    cdef np.ndarray[uint64_t] arr = np.empty(vec.size(), dtype=np.uint64)

    cdef void* arr_p = <void*> arr.data
    cdef void* vec_p = <void*> &vec[0]

    memcpy(arr_p, vec_p, sizeof(uint64_t) * vec.size())

    return arr
