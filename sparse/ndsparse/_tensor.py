import pytaco as pt
import sparse
import scipy.sparse
import numpy as np
from .._sparse_array import SparseArray
from numpy.lib.mixins import NDArrayOperatorsMixin

mode = pt.compressed


class tensor(SparseArray, NDArrayOperatorsMixin):
    """
    A sparse tensor object which wraps around the Pytaco tensor class.
    Initializes a zero tensor of a given shape.

    Parameter
    ----------
    shape : An integer or an iterable object. Denotes the shape of the tensor.

    Methods
    --------
    todense
    to_scipy_csc
    to_scipy_csr
    insert
    order
    transpose
    remove_explicit_zeros
    """

    def __init__(self, shape):

        if isinstance(shape, int) or isinstance(shape, float):
            self.shape = shape

        elif isinstance(shape, str):
            raise ValueError("String is not a valid input")

        elif hasattr(shape, "__iter__"):
            self.shape = shape

        else:
            raise ValueError("Not an iterable object")

        self._tensor = pt.tensor(shape)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise ValueError("Invalid call")
        print(inputs)
        if len(inputs) == 1:
            (inputs,) = inputs
            inputs = inputs.todense()
            return solve(ufunc, *inputs)
        elif len(inputs) == 2:
            input1, input2 = inputs
            input1 = input1.todense()
            input2 = input2.todense()
            inputs = (input1, input2)
            return solve(ufunc, *inputs)
        else:
            raise ValueError("Invalid number of arguments")

    def todense(self):
        """
        Densifies the tensor
        """
        return np.asarray(self._tensor.to_dense())

    def to_scipy_csc(self):
        """
        Convert tensor into scipy sparse array
        """
        return pt.to_sp_csc(self._tensor)

    def to_scipy_csr(self):
        """
        Convert tensor into scipy sparse array
        """
        return pt.to_sp_csr(self._tensor)

    def insert(self, coord, val):
        """
        Inserts value into the tensor
        """
        self._tensor.insert(coord, val)

    def order(self):
        """
        Returns the order of the tensor
        """
        return self._tensor.order()

    def transpose(self, new_ordering, new_format=None, name=None):
        """
        Returns the transpose of the tensor
        """
        return self._tensor.transpose(new_ordering, new_format, name)

    def T():
        """
        Transpose with shape = [::-1]
        """
        return self._tensor.T()

    def __array__(self):
        return self.todense()

    def __getitem__(self, index):
        return self._tensor[index]

    def __setitem__(self, index, val):
        self._tensor[index] = val

    def remove_explicit_zeros(self):
        """
        Return a tensor with no zeros
        """
        return pt.remove_explicit_zeros(self._tensor)


def from_scipy_sparse(arr):
    """
    Construct a tensor from scipy sparse array
    """
    if isinstance(arr, scipy.sparse.csc_matrix):
        return pt.from_sp_csc(arr)
    elif isinstance(arr, scipy.sparse.csr_matrix):
        return pt.from_sp_csc(arr)
    else:
        raise TypeError("Input is not a scipy csc/csr matrix")


def from_array(arr):
    """
    Converts a numpy ndarray into a sparse tensor.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("The input must be a Numpy array")
    else:
        return pt.as_tensor(arr)


class Solve:
    def __init__(self, ufunc, *args, **kwargs):
        """
        Ufunc solver for __array_ufunc__

        Parameters
        -----------

        ufunc : The numpy universal function
        *args : Tensor/ndarray/SparseArray
        **kwargs : Extra arguments
        """
        import scipy.sparse

        arg_converted = []
        for arg in args:
            if isinstance(arg, pt.tensor):
                arg_converted.append(np.asarray(arg.to_dense()))
            elif isinstance(arg, SparseArray):
                arg_converted.append(np.asarray(arg.todense()))
            elif isinstance(arg, np.ndarray):
                arg_converted.append(arg)
            elif isinstance(arg, sparse.tensor):
                arg_converted.append(arg.todense())
            elif isinstance(arg, scipy.sparse.spmatrix):
                arg_converted.append(arg.toarray())
            else:
                raise TypeError("Incompatible type")

        self.args = arg_converted
        self.func = ufunc
        self.kwargs = kwargs

    def solution(self):
        """
        Applies ufuncs to the given arguments
        """
        return self.func(*self.args, **self.kwargs)


def solve(func, *args, **kwargs):
    """
    Function to use the Solve class
    """
    return Solve(func, *args, **kwargs).solution()
