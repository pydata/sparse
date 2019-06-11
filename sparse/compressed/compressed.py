import numpy as np
from functools import reduce
from operator import mul
import scipy.sparse as ss

from ..sparse_array import SparseArray
from ..coo.common import linear_loc
from ..utils import normalize_axis, equivalent, check_zero_fill_value, _zero_of_dtype
from ..coo.core import COO 
from .convert import uncompress_dimension
from .indexing import getitem



def _from_coo(x,format):
    midpoint = int(len(x.shape) // 2)
    midpoint = midpoint + 1 if len(x.shape) % 2 == 1 else midpoint # where do col axes start
    if len(x.shape)==3:
        midpoint = 2
    row_size = int(np.prod(x.shape[:midpoint]))
    col_size = int(np.prod(x.shape[midpoint:]))
    coords = x.reshape((row_size,col_size)).coords

    if format is 'CSR':
        indptr = np.zeros(row_size+1,dtype=int)
        np.cumsum(np.bincount(coords[0], minlength=row_size), out=indptr[1:])
        indices = coords[1]
        data = x.data
    else: 
        linear = linear_loc(coords[[1,0]],(col_size,row_size))
        order = np.argsort(linear)
        coords = coords[:,order]
        indptr = np.zeros(col_size+1,dtype=int)
        np.cumsum(np.bincount(coords[1], minlength=col_size), out=indptr[1:])
        indices = coords[0]
        data = x.data[order]
    return (data,indices,indptr), x.shape, x.fill_value

class compressed(SparseArray):

    def __init__(self,arg,shape=None,fill_value=0):

        if isinstance(arg,np.ndarray):
            arg, shape, fill_value = _from_coo(COO(arg),self.format)

        elif isinstance(arg,COO):
            arg, shape, fill_value = _from_coo(arg,self.format)

        if isinstance(arg,tuple):
            data,indices,indptr = arg
            self.data = data
            self.indices = indices
            self.indptr = indptr
            self.shape = shape
            sl = len(shape)
            row_size = int(np.prod(shape[:sl//2+1]) if sl%2==1 else np.prod(shape[:sl//2]))
            col_size = int(np.prod(shape[sl//2+1:]) if sl%2==1 else np.prod(shape[sl//2:]))
            self.compressed_shape = (row_size,col_size)
            self.fill_value = fill_value
            self.dtype = self.data.dtype

    @classmethod
    def from_numpy(cls,x,fill_value=0):
        coo = COO(x,fill_value=fill_value)
        return cls.from_coo(coo)


    @classmethod
    def from_coo(cls,x):
        arg, shape, fill_value = _from_coo(x,cls.format)
        return cls(arg,shape=shape,fill_value=fill_value)

    @classmethod
    def from_scipy_sparse(cls,x):
        if cls.format is 'CSR':
            x = x.asformat('csr')
            return cls((x.data,x.indices,x.indptr),shape=x.shape)
        else:
            x = x.asformat('csc')
            return cls((x.data,x.indices,x.indptr),shape=x.shape)
        

    @classmethod
    def from_iter(cls,x,shape=None,fill_value=None):
        return cls.from_coo(COO.from_iter(x,shape,fill_value))

    @property
    def nnz(self):
        return self.data.shape[0]

    @property
    def nbytes(self):
        return self.data.nbytes + self.indices.nbytes + self.indptr.nbytes
    
    @property
    def density(self):
        return self.nnz / reduce(mul,self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def __str__(self):
        return '<{}: shape={}, dtype={}, nnz={}, fill_value={}>'.format(self.format,self.shape,self.dtype,self.nnz,self.fill_value)

    __repr__ = __str__      


    __getitem__ = getitem

    def tocoo(self):
        uncompressed = uncompress_dimension(self.indptr,self.indices)
        coords = np.vstack((uncompressed,self.indices)) if self.format is 'CSR' else np.vstack((self.indices,uncompressed))
        return COO(coords,self.data,shape=self.compressed_shape,fill_value=self.fill_value).reshape(self.shape) 

    def todense(self):       
        return self.tocoo().todense()
    

    def todok(self):

        from ..dok import DOK 
        return DOK.from_coo(self.tocoo()) # probably a temporary solution


    def to_scipy_sparse(self):
        """
        Converts this :obj:`CSR` or `CSC` object into a :obj:`scipy.sparse.csr_matrix` or `scipy.sparse.csc_matrix`.
        Returns
        -------
        :obj:`scipy.sparse.csr_matrix` or `scipy.sparse.csc_matrix`
            The converted Scipy sparse matrix.
        Raises
        ------
        ValueError
            If the array is not two-dimensional.
        ValueError
            If all the array doesn't zero fill-values.
        """
        
        check_zero_fill_value(self)

        if self.ndim != 2:
            raise ValueError("Can only convert a 2-dimensional array to a Scipy sparse matrix.")

        if self.format is 'CSR':
            return ss.csr_matrix((self.data,self.indices,self.indptr),shape=self.shape)
        else:
            return ss.csc_matrix((self.data,self.indices,self.indptr),shape=self.shape)

    
    def asformat(self,format):
        """
        Convert this sparse array to a given format.
        Parameters
        ----------
        format : str
            A format string.
        Returns
        -------
        out : SparseArray
            The converted array.
        Raises
        ------
        NotImplementedError
            If the format isn't supported.
        """

        if format is 'coo':
            return self.tocoo()
        elif format is 'csc':
            return self.tocsc()
        elif format is 'csr':
            return self.tocsr()
        elif format is 'dok':
            return self.todok()
        
        raise NotImplementedError('The given format is not supported.')


    def maybe_densify(self, max_size=1000, min_density=0.25):
        """
        Converts this :obj:`CSR` or `CSC` array to a :obj:`numpy.ndarray` if not too
        costly.
        Parameters
        ----------
        max_size : int
            Maximum number of elements in output
        min_density : float
            Minimum density of output
        Returns
        -------
        numpy.ndarray
            The dense array.
        Raises
        -------
        ValueError
            If the returned array would be too large.
        """

        if self.size <= max_size or self.density >= min_density:
            return self.todense()
        else:
            raise ValueError("Operation would require converting "
                            "large sparse array to dense")
    

    def reshape(self,shape, order='C'):
        """
        Returns a new :obj:`CSR` or `CSC` array that is a reshaped version of this array.
        Parameters
        ----------
        shape : tuple[int]
            The desired shape of the output array.
        Returns
        -------
        CSR or CSC
            The reshaped output array.
        See Also
        --------
        numpy.ndarray.reshape : The equivalent Numpy function.
        sparse.COO.reshape: The equivalent COO function.
        Notes
        -----
        The :code:`order` parameter is provided just for compatibility with
        Numpy and isn't actually supported.
        
        """


        if order not in {'C', None}:
            raise NotImplementedError("The 'order' parameter is not supported")
        if any(d == -1 for d in shape):
            extra = int(self.size /
                    np.prod([d for d in shape if d != -1]))
            shape = tuple([d if d != -1 else extra for d in shape])

        if self.shape==shape:
            return self
        
        if self.size != reduce(mul,shape,1):
            raise ValueError('cannot reshape array of size {} into shape {}'.format(self.size,shape))
        
        midpoint = int(len(shape)//2)
        midpoint = midpoint + 1 if len(shape) % 2 == 1 else midpoint
        row_size = np.prod(shape[:midpoint])
        col_size = np.prod(shape[midpoint:])
        uncompressed = uncompress_dimension(self.indptr,self.indices)
        coords = np.vstack((uncompressed,self.indices)) if self.format is "CSR" else np.vstack((self.indices,uncompressed))
        reshaped_coords = COO(coords,self.data,shape=self.compressed_shape).reshape((row_size,col_size)).coords

        if self.format is 'CSR':
            indptr = np.zeros(row_size+1,dtype=int)
            np.cumsum(np.bincount(reshaped_coords[0], minlength=row_size), out=indptr[1:])
            indices = reshaped_coords[1]
        else:
            indptr = np.zeros(col_size+1,dtype=int)
            np.cumsum(np.bincount(reshaped_coords[1], minlength=col_size), out=indptr[1:])
            indices = reshaped_coords[0]
        
        return self.__class__((self.data,indices,indptr),shape=shape,fill_value=self.fill_value)
            
        
        

    def resize(self, *args, refcheck=True):
        """
        This method changes the shape and size of an array in-place.
        
        Parameters
        ----------
        args : tuple, or series of integers
            The desired shape of the output array.
        
        See Also
        --------
        numpy.ndarray.resize : The equivalent Numpy function.
        sparse.COO.resize : The equivalent COO function.

        """
        
        if len(args) == 1 and isinstance(args[0], tuple):
            shape = args[0]
        elif all(isinstance(arg, int) for arg in args):
            shape = tuple(args)
        else:
            raise ValueError('Invalid input')

       
        if any(d < 0 for d in shape):
            raise ValueError('negative dimensions not allowed')
        
        if self.shape==shape:
            return
        
        
        midpoint = int(len(shape)//2)
        midpoint = midpoint + 1 if len(shape) % 2 == 1 else midpoint
        row_size = np.prod(shape[:midpoint])
        col_size = np.prod(shape[midpoint:])
        uncompressed = uncompress_dimension(self.indptr,self.indices)
        coords = np.vstack((uncompressed,self.indices)) if self.format is "CSR" else np.vstack((self.indices,uncompressed))
        resized = COO(coords,self.data,shape=self.compressed_shape).resize((row_size,col_size))
        resized_coords = resized.coords
        self.data = resized.data
        self.shape = shape

        if self.format is 'CSR':
            self.indptr = np.zeros(row_size+1,dtype=int)
            np.cumsum(np.bincount(resized_coords[0], minlength=row_size), out=self.indptr[1:])
            self.indices = resized_coords[1]
        else:
            self.indptr = np.zeros(col_size+1,dtype=int)
            np.cumsum(np.bincount(resized_coords[1], minlength=col_size), out=self.indptr[1:])
            self.indices = resized_coords[0]
        
