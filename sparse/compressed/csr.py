import numpy as np 
from sparse.coo.common import linear_loc
from .compressed import compressed
from .convert import uncompress_dimension

class CSR(compressed):

    format = 'CSR'

 
    def tocsr(self):
        return self

    def tocsc(self):
        uncompressed = uncompress_dimension(self.indptr,self.indices)
        coords = np.vstack((uncompressed,self.indices))
        linear = linear_loc(coords[[1,0]],(self.compressed_shape[1],self.compressed_shape[0]))
        order = np.argsort(linear)
        coords = coords[:,order]
        indptr = np.zeros(self.compressed_shape[1]+1,dtype=int)
        np.cumsum(np.bincount(coords[1],minlength=self.compressed_shape[1]), out=indptr[1:])
        data = self.data[order]
        from .csc import CSC
        return CSC((data,coords[0],indptr),shape=self.shape,fill_value=self.fill_value)