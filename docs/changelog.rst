Changelog
=========

* :feature:`124` (via :issue:`182`) Allow mixed :code:`ndarray`-:obj:`COO` operations if the result is sparse.
* :feature:`179` (via :issue:`180`) Allow specifying a fill-value when converting from NumPy arrays.
* :feature:`175` Added :code:`COO.any` and :code:`COO.all` methods.
* :feature:`172` Indexing for :code:`COO` now accepts a single one-dimensional array index.
* :feature:`165` The fill-value can now be something other than zero or :code:`False`.
* :feature:`160` Added a :code:`sparse.roll` function.
* :feature:`159` Numba code now releases the GIL. This leads to better multi-threaded performance in Dask.
* :bug:`158` A number of bugs occurred, so to resolve them, :code:`COO.coords.dtype` is always :code:`np.int64`.
  :code:`COO`, therefore, uses more memory than before.
* :feature:`153` (via :issue:`154`) Add support for saving and loading :code:`COO` files from disk
* :feature:`145` (via :issue:`148`) Support :code:`COO.nonzero` and :code:`np.argwhere`
* :feature:`80` (via :issue:`146`) Allow faux in-place operations
* :support:`141` :code:`COO` is now always canonical
* :feature:`128` Improve indexing performance
* :feature:`127` Improve element-wise performance
* :feature:`117` (via :issue:`118`) Reductions now support a negative axis.
* :bug:`107` (via :issue:`108`) Match behaviour of :code:`ufunc.reduce` from NumPy
* :release:`0.3.1 <2018-04-12>`
* :bug:`138` Fix packaging error.
* :release:`0.3.0 <2018-02-22>`
* :feature:`102` Add NaN-skipping aggregations
* :feature:`102` Add equivalent to :code:`np.where`
* :feature:`98` N-input universal functions now work
* :feature:`96` Make :code:`dot` more consistent with NumPy
* :support:`92` Create a base class :code:`SparseArray`
* :support:`90` Minimum NumPy version is now 1.13
* :bug:`93` (via :issue:`94`) Fix a bug where setting a :code:`DOK` element to zero did nothing.
* :release:`0.2.0 <2018-01-25>`
* :feature:`87` Support faster :code:`np.array(COO)`
* :feature:`85` Add :code:`DOK` type
* :bug:`82` (via :issue:`83`) Fix sum for large arrays
* :feature:`69` Support :code:`.size` and :code:`.density`
* :support:`43` Documentation added for the package
* :support:`70` Minimum required SciPy version is now 0.19
* :feature:`68` :code:`len(COO)` now works
* :feature:`67` :code:`scalar op COO` now works for all operators
* :bug:`61` Validate axes for :code:`.transpose()`
* :feature:`57` Extend indexing support
* :feature:`41` Add :code:`random` function for generating random sparse arrays
* :feature:`55` :code:`COO(COO)` now copies the original object
* :feature:`49` NumPy universal functions and reductions now work on :code:`COO` arrays
* :bug:`32` (via :issue:`51`) Fix concatenate and stack for large arrays
* :bug:`47` (via :issue:`48`) Fix :code:`nnz` for scalars
* :feature:`46` Support more operators and remove all special cases
* :feature:`40` Add support for :code:`triu` and :code:`tril`
* :feature:`37` Add support for Ellipsis (:code:`...`) and :code:`None` when indexing
* :feature:`38` Add support for bitwise bindary operations like :code:`&` and :code:`|`
* :feature:`35` Support broadcasting in element-wise operations
