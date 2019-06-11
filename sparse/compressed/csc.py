import numpy as np 
from sparse.coo.common import linear_loc
from .compressed import compressed
from .convert import uncompress_dimension

class CSC(compressed):

    format = 'CSC'

    def tocsc(self):
        return self
    
    def tocsr(self):
        uncompressed = uncompress_dimension(self.indptr,self.indices)
        coords = np.vstack((self.indices,uncompressed))
        linear = linear_loc(coords,self.compressed_shape)
        order = np.argsort(linear)
        coords = coords[:,order]
        data = self.data[order]
        indptr = np.zeros(self.compressed_shape[0]+1,dtype=int)
        np.cumsum(np.bincount(coords[0],minlength=self.compressed_shape[0]),out=indptr[1:])
        from .csr import CSR
        return CSR((data,coords[1],indptr),shape=self.shape,fill_value=self.fill_value)
