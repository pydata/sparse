# Construct Sparse Arrays

## From coordinates and data

You can construct [`sparse.COO`][] arrays from coordinates and value data.

The `cords` parameter contains the indices where the data is nonzero,
and the `data` parameter contains the data corresponding to those indices.
For example, the following code will generate a $5 \times 5$ diagonal
matrix:

```python

>>> import sparse

>>> coords = [[0, 1, 2, 3, 4],
...           [0, 1, 2, 3, 4]]
>>> data = [10, 20, 30, 40, 50]
>>> s = sparse.COO(coords, data, shape=(5, 5))
>>> s
<COO: shape=(5, 5), dtype=int64, nnz=5, fill_value=0>
     0    1    2    3    4
  ┌                         ┐
0 │ 10                      │
1 │      20                 │
2 │           30            │
3 │                40       │
4 │                     50  │
  └                         ┘
```

In general `coords` should be a `(ndim, nnz)` shaped
array. Each row of `coords` contains one dimension of the
desired sparse array, and each column contains the index
corresponding to that nonzero element. `data` contains
the nonzero elements of the array corresponding to the indices
in `coords`. Its shape should be `(nnz,)`.

If `data` is the same across all the coordinates, it can be passed
in as a scalar. For example, the following produces the $4 \times 4$
identity matrix:

```python

>>> import sparse

>>> coords = [[0, 1, 2, 3],
...           [0, 1, 2, 3]]
>>> data = 1
>>> s = sparse.COO(coords, data, shape=(4, 4))
>>> s
<COO: shape=(4, 4), dtype=int64, nnz=4, fill_value=0>
     0    1    2    3
  ┌                    ┐
0 │  1                 │
1 │       1            │
2 │            1       │
3 │                 1  │
  └                    ┘
```

You can, and should, pass in [`numpy.ndarray`][] objects for
`coords` and `data`.

In this case, the shape of the resulting array was determined from
the maximum index in each dimension. If the array extends beyond
the maximum index in `coords`, you should supply a shape
explicitly. For example, if we did the following without the
`shape` keyword argument, it would result in a
$4 \times 5$ matrix, but maybe we wanted one that was actually
$5 \times 5$.

```python

>>> coords = [[0, 3, 2, 1], [4, 1, 2, 0]]
>>> data = [1, 4, 2, 1]
>>> s = COO(coords, data, shape=(5, 5))
>>> s
<COO: shape=(5, 5), dtype=int64, nnz=4, fill_value=0>
     0    1    2    3    4
  ┌                         ┐
0 │                      1  │
1 │  1                      │
2 │            2            │
3 │       4                 │
4 │                         │
  └                         ┘
```

[`sparse.COO`][] arrays support arbitrary fill values. Fill values are the "default"
value, or value to not store. This can be given a value other than zero. For
example, the following builds a (bad) representation of a $2 \times 2$
identity matrix. Note that not all operations are supported for operations
with nonzero fill values.

```python

>>> coords = [[0, 1], [1, 0]]
>>> data = [0, 0]
>>> s = COO(coords, data, fill_value=1)
>>> s
<COO: shape=(2, 2), dtype=int64, nnz=2, fill_value=1>
     0    1
  ┌          ┐
0 │       0  │
1 │  0       │
  └          ┘
```

## From [`scipy.sparse.spmatrix`][]

To construct [`sparse.COO`][] array from [spmatrix][scipy.sparse.spmatrix]
objects, you can use the [`sparse.COO.from_scipy_sparse`][] method. As an
example, if `x` is a [scipy.sparse.spmatrix][], you can
do the following to get an equivalent [`sparse.COO`][] array:

```python

s = COO.from_scipy_sparse(x)
```

## From [Numpy arrays][`numpy.ndarray`]

To construct [`sparse.COO`][] arrays from [`numpy.ndarray`][]
objects, you can use the [`sparse.COO.from_numpy`][] method. As an
example, if `x` is a [`numpy.ndarray`][], you can
do the following to get an equivalent [`sparse.COO`][] array:

```python

s = COO.from_numpy(x)
```

## Generating random [`sparse.COO`][] objects

The [`sparse.random`][] method can be used to create random
[`sparse.COO`][] arrays. For example, the following will generate
a $10 \times 10$ matrix with $10$ nonzero entries,
each in the interval $[0, 1)$.

```python

s = sparse.random((10, 10), density=0.1)
```

Building [`sparse.COO`][] Arrays from [`sparse.DOK`][] Arrays

It's possible to build [`sparse.COO`][] arrays from [`sparse.DOK`][] arrays, if it is not
easy to construct the `coords` and `data` in a simple way. [`sparse.DOK`][]
arrays provide a simple builder interface to build [`sparse.COO`][] arrays, but at
this time, they can do little else.

You can get started by defining the shape (and optionally, datatype) of the
[`sparse.DOK`][] array. If you do not specify a dtype, it is inferred from the value
dictionary or is set to `dtype('float64')` if that is not present.

```python

s = DOK((6, 5, 2))
s2 = DOK((2, 3, 4), dtype=np.uint8)
```

After this, you can build the array by assigning arrays or scalars to elements
or slices of the original array. Broadcasting rules are followed.

```python

s[1:3, 3:1:-1] = [[6, 5]]
```

DOK arrays also support fancy indexing assignment if and only if all dimensions are indexed.

```python

s[[0, 2], [2, 1], [0, 1]] = 5
s[[0, 3], [0, 4], [0, 1]] = [1, 5]
```

Alongside indexing assignment and retrieval, [`sparse.DOK`][] arrays support any arbitrary broadcasting function
to any number of arguments where the arguments can be [`sparse.SparseArray`][] objects, [`scipy.sparse.spmatrix`][]
objects, or [`numpy.ndarray`][].

```python

x = sparse.random((10, 10), 0.5, format="dok")
y = sparse.random((10, 10), 0.5, format="dok")
sparse.elemwise(np.add, x, y)
```

[`sparse.DOK`][] arrays also support standard ufuncs and operators, including comparison operators,
in combination with other objects implementing the *numpy* *ndarray.\__array_ufunc\__* method. For example,
the following code will perform elementwise equality comparison on the two arrays
and return a new boolean [`sparse.DOK`][] array.

```python

x = sparse.random((10, 10), 0.5, format="dok")
y = np.random.random((10, 10))
x == y
```

[`sparse.DOK`][] arrays are returned from elemwise functions and standard ufuncs if and only if all
[`sparse.SparseArray`][] objects are [`sparse.DOK`][] arrays. Otherwise, a [`sparse.COO`][] array or dense array are returned.

At the end, you can convert the [`sparse.DOK`][] array to a [`sparse.COO`][] arrays.

```python

s3 = COO(s)
```

In addition, it is possible to access single elements and slices of the [`sparse.DOK`][] array
using normal Numpy indexing, as well as fancy indexing if and only if all dimensions are indexed.
Slicing and fancy indexing will always return a new DOK array.

```python

s[1, 2, 1]  # 5
s[5, 1, 1]  # 0
s[[0, 3], [0, 4], [0, 1]] # <DOK: shape=(2,), dtype=float64, nnz=2, fill_value=0.0>
```

## Converting [`sparse.COO`][] objects to other Formats

[`sparse.COO`][] arrays can be converted to [Numpy arrays][numpy.ndarray],
or to some [spmatrix][scipy.sparse.spmatrix] subclasses via the following
methods:

* [`sparse.COO.todense`][]: Converts to a [`numpy.ndarray`][] unconditionally.
* [`sparse.COO.maybe_densify`][]: Converts to a [`numpy.ndarray`][] based on
   certain constraints.
* [`sparse.COO.to_scipy_sparse`][]: Converts to a [`scipy.sparse.coo_matrix`][] if
   the array is two dimensional.
* [`sparse.COO.tocsr`][]: Converts to a [`scipy.sparse.csr_matrix`][] if
   the array is two dimensional.
* [`sparse.COO.tocsc`][]: Converts to a [`scipy.sparse.csc_matrix`][] if
   the array is two dimensional.
