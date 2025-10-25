# Migration to the Finch Julia backend
To switch to the Finch Julia backend, set the environment variable `SPARSE_BACKEND="Finch"`, then continue using.

While this is largely compatible with the Array API, support for some functions may not be present, and API compatibility isn't strictly preserved with the default (Numba) backend.

However, the new backend has a large performance benefit over the default backend. Below, you will find a table of common invocations, with their equivalents in the Finch Julia backend. The most common change is a standard API for construction of arrays.

| Numba Backend<br>(`SPARSE_BACKEND="Numba"`) | Finch Julia Backend<br>(`SPARSE_BACKEND="Finch"`)  | Notes |
|---------------------------------------------|----------------------------------------------------|-------|
| `sparse.COO.from_numpy(arr, fill_value=fv)`<br>`sparse.COO.from_scipy(arr)`<br>`sparse.COO(x)` | `sparse.asarray(x, format="coo", [fill_value=fv])` | Doesn't support pulling out individual arrays |
| `sparse.GCXS.from_numpy(arr, fill_value=fv)`<br>`sparse.GCXS.from_scipy(arr)`<br>`sparse.GCXS(x)` | `sparse.asarray(x, format="csf", [fill_value=fv])` | Format might not be a 1:1 match |
| `sparse.DOK.from_numpy(arr, fill_value=fv)`<br>`sparse.DOK.from_scipy(arr)`<br>`sparse.DOK(x)` | `sparse.asarray(x, format="dok", [fill_value=fv])` | Format might not be a 1:1 match |

Most things work as expected, with the following exceptions, which aren't defined yet for Finch:

* `sparse.broadcast_to`
* `sparse.solve`
* Statistical functions: `mean`, `std`, `var`
* `sparse.isdtype`
* `sparse.reshape`
* Some elementwise functions
* Manipulation functions: `concat`, `expand_dims`, `squeeze`, `flip`, `roll`, `stack`
* `arg*` functions: `argmin`, `argmax`
* Sorting functions: `sort`, `argsort`

IEEE-754 compliance is hard to maintain with sparse arrays in general. This is now even more true of the Julia backend, which trades off performance for IEEE-754 compatibility.
