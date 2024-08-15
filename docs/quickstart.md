# Getting Started

## Install

If you haven't already, install the `sparse` library

```bash
pip install sparse
```

## Create

To start, lets construct a sparse [`sparse.COO`][] array from a [`numpy.ndarray`][]:

```python

import numpy as np
import sparse

x = np.random.random((100, 100, 100))
x[x < 0.9] = 0  # fill most of the array with zeros

s = sparse.COO(x)  # convert to sparse array
```

These store the same information and support many of the same operations,
but the sparse version takes up less space in memory

```python
>>> x.nbytes
8000000
>>> s.nbytes
1102706
>>> s
<COO: shape=(100, 100, 100), dtype=float64, nnz=100246, fill_value=0.0>
```

For more efficient ways to construct sparse arrays,
see documentation on [Construct sparse arrays][construct-sparse-arrays].

## Compute

Many of the normal Numpy operations work on [`sparse.COO`][] objects just like on [`numpy.ndarray`][] objects.
This includes arithmetic, [`numpy.ufunc`][] operations, or functions like tensordot and transpose.

```python
>>> np.sin(s) + s.T * 1
<COO: shape=(100, 100, 100), dtype=float64, nnz=189601, fill_value=0.0>
```

However, operations which map zero elements to nonzero will usually change the fill-value
instead of raising an error.

```python
>>> y = s + 5
<COO: shape=(100, 100, 100), dtype=float64, nnz=100246, fill_value=5.0>
```

However, if you're sure you want to convert a sparse array to a dense one,
you can use the ``todense`` method (which will result in a [`numpy.ndarray`][]):

```python
y = s.todense() + 5
```

For more operations see the [operations][operators]
or the [API reference page](../../api/).
