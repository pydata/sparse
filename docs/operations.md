# Operations on [`sparse.COO`][] and [`sparse.GCXS`][] arrays

## Operators

[`sparse.COO`][] and [`sparse.GCXS`][] objects support a number of operations. They interact with scalars,
[`sparse.COO`][] and [`sparse.GCXS`][] objects,
[scipy.sparse.spmatrix][] objects, all following standard Python and Numpy
conventions.

For example, the following Numpy expression produces equivalent
results for both Numpy arrays, COO arrays, or a mix of the two:

```python

np.log(X.dot(beta.T) + 1)
```

However some operations are not supported, like operations that
implicitly cause dense structures, or numpy functions that are not
yet implemented for sparse arrays.

```python

np.linalg.cholesky(x)  # sparse cholesky not implemented
```

This page describes those valid operations, and their limitations.

**[`sparse.elemwise`][]**

This function allows you to apply any arbitrary broadcasting function to any number of arguments
where the arguments can be [`sparse.SparseArray`][] objects or [`scipy.sparse.spmatrix`][] objects.
For example, the following will add two arrays:

```python

sparse.elemwise(np.add, x, y)
```

!!! warning

    Previously, [`sparse.elemwise`][] was a method of the [`sparse.COO`][] class. Now,
    it has been moved to the [sparse][] module.


**Auto-Densification**

Operations that would result in dense matrices, such as
operations with [Numpy arrays][`numpy.ndarray`]
raises a [ValueError][]. For example, the following will raise a
[ValueError][] if `x` is a [`numpy.ndarray`][]:

```python

x + y
```

However, all of the following are valid operations.

```python

x + 0
x != y
x + y
x == 5
5 * x
x / 7.3
x != 0
x == 0
~x
x + 5
```

We also support operations with a nonzero fill value. These are operations
that map zero values to nonzero values, such as `x + 1` or `~x`.
In these cases, they will produce an output with a fill value of `1` or `True`,
assuming the original array has a fill value of `0` or `False` respectively.

If densification is needed, it must be explicit. In other words, you must call
[`sparse.SparseArray.todense`][] on the [`sparse.SparseArray`][] object. If both operands are [`sparse.SparseArray`][],
both must be densified.

**Operations with NumPy arrays**

In certain situations, operations with NumPy arrays are also supported. For example,
the following will work if `x` is [`sparse.COO`][] and `y` is a NumPy array:

```python

x * y
```

The following conditions must be met when performing element-wise operations with
NumPy arrays:

* The operation must produce a consistent fill-values. In other words, the resulting
  array must also be sparse.
* Operating on the NumPy arrays must not increase the size when broadcasting the arrays.

## Operations with [`scipy.sparse.spmatrix`][]

Certain operations with [`scipy.sparse.spmatrix`][] are also supported.
For example, the following are all allowed if `y` is a [`scipy.sparse.spmatrix`][]:

```python

x + y
x - y
x * y
x > y
x < y
```

In general, operating on a [`scipy.sparse.spmatrix`][] is the same as operating
on [`sparse.COO`][] or [`sparse.GCXS`][], as long as it is to the right of the operator.

!!! note

    Results are not guaranteed if `x` is a [scipy.sparse.spmatrix][].
    For this reason, we recommend that all Scipy sparse matrices should be explicitly
    converted to [`sparse.COO`][] or [`sparse.GCXS`][] before any operations.


## Broadcasting

All binary operators support [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).
This means that (under certain conditions) you can perform binary operations
on arrays with unequal shape. Namely, when the shape is missing a dimension,
or when a dimension is `1`. For example, performing a binary operation
on two `COO` arrays with shapes `(4,)` and `(5, 1)` yields
an object of shape `(5, 4)`. The same happens with arrays of shape
`(1, 4)` and `(5, 1)`. However, `(4, 1)` and `(5, 1)`
will raise a [`ValueError`][].If densification is needed,


## Element-wise Operations

[`sparse.COO`][] and [`sparse.GCXS`][] arrays support a variety of element-wise operations. However, as
with operators, operations that map zero to a nonzero value are not supported.

To illustrate, the following are all possible, and will produce another
[`sparse.SparseArray`][]:

```python

np.abs(x)
np.sin(x)
np.sqrt(x)
np.conj(x)
np.expm1(x)
np.log1p(x)
np.exp(x)
np.cos(x)
np.log(x)
```

As above, in the last three cases, an array with a nonzero fill value will be produced.

Notice that you can apply any unary or binary [`sparse.COO`][]
arrays, and [`numpy.ndarray`][] objects and scalars and it will work so
long as the result is not dense. When applying to [`numpy.ndarray`][] objects,
we check that operating on the array with zero would always produce a zero.


## Reductions

[`sparse.COO`][] and [`sparse.GCXS`][] objects support a number of reductions. However, not all important
reductions are currently implemented (help welcome!). All of the following
currently work:

```python

x.sum(axis=1)
np.max(x)
np.min(x, axis=(0, 2))
x.prod()
```

[`sparse.SparseArray.reduce`][]

This method can take an arbitrary [`numpy.ufunc`][] and performs a
reduction using that method. For example, the following will perform
a sum:

```python

x.reduce(np.add, axis=1)
```

!!! note

    This library currently performs reductions by grouping together all
    coordinates along the supplied axes and reducing those. Then, if the
    number in a group is deficient, it reduces an extra time with zero.
    As a result, if reductions can change by adding multiple zeros to
    it, this method won't be accurate. However, it works in most cases.

**Partial List of Supported Reductions**

Although any binary [`numpy.ufunc`][] should work for reductions, when calling
in the form `x.reduction()`, the following reductions are supported:

* [`sparse.COO.sum`][]
* [`sparse.COO.max`][]
* [`sparse.COO.min`][]
* [`sparse.COO.prod`][]


## Indexing

[`sparse.COO`][] and [`sparse.GCXS`][] arrays can be [indexed](https://numpy.org/doc/stable/user/basics.indexing.html)
just like regular [`numpy.ndarray`][] objects. They support integer, slice and boolean indexing.
However, currently, numpy advanced indexing is not properly supported. This
means that all of the following work like in Numpy, except that they will produce
[`sparse.SparseArray`][] arrays rather than [`numpy.ndarray`][] objects, and will produce
scalars where expected. Assume that `z.shape` is `(5, 6, 7)`

```python

z[0]
z[1, 3]
z[1, 4, 3]
z[:3, :2, 3]
z[::-1, 1, 3]
z[-1]
```

All of the following will raise an `IndexError`, like in Numpy 1.13 and later.

```python

z[6]
z[3, 6]
z[1, 4, 8]
z[-6]
```

**Advanced Indexing**

Advanced indexing (indexing arrays with other arrays) is supported, but only for indexing
with a *single array*. Indexing a single array with multiple arrays is not supported at
this time. As above, if `z.shape` is `(5, 6, 7)`, all of the following will
work like NumPy:

```python

z[[0, 1, 2]]
z[1, [3]]
z[1, 4, [3, 6]]
z[:3, :2, [1, 5]]
```


**Package Configuration**

By default, when performing something like `np.array(COO)`, we do not allow the
array to be converted into a dense one and it raise a [`RuntimeError`][].
To prevent this, set the environment variable `SPARSE_AUTO_DENSIFY` to `1`.

If it is desired to raise a warning if creating a sparse array that takes no less
memory than an equivalent desne array, set the environment variable
`SPARSE_WARN_ON_TOO_DENSE` to `1`.


## Other Operations

[`sparse.COO`][] and [`sparse.GCXS`][] arrays support a number of other common operations. Among them are
[`sparse.dot`][], [`sparse.tensordot`][] [`sparse.einsum`][], [`sparse.concatenate`][]
and [`sparse.stack`][], [`sparse.COO.transpose`][] and [`sparse.COO.reshape`][].
You can view the full list on the [API reference page](../../api/).

!!! note

    Some operations require zero fill-values (such as [`sparse.COO.nonzero`][])
    and others (such as [`sparse.concatenate`][]) require that all inputs have consistent fill-values.
    For details, check the API reference.
