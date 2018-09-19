Changelog
=========

Upcoming Release
----------------

* Added :code:`order` parameter to :code:`COO.reshape` to make it work with
  :code:`np.reshape` (:pr:`193`).
* Added :code:`COO.mean` and :code:`sparse.nanmean` (:pr:`190`).
* Added :code:`sparse.full` and :code:`sparse.full_like` (:pr:`189`).
* Added :code:`COO.clip` method (:pr:`185`).
* Added :code:`COO.copy` method, and changed pickle of :code:`COO` to not
  include its cache (:pr:`184`).
* Added :code:`sparse.eye`, :code:`sparse.zeros`, :code:`sparse.zeros_like`,
  :code:`sparse.ones`, and :code:`sparse.ones_like` (:pr:`183`).

0.4.1 / 2018-09-12
------------------

* Allow mixed :code:`ndarray`-:code:`COO` operations if the result is sparse
  (:issue:`124`, via :pr:`182`).
* Allow specifying a fill-value when converting from NumPy arrays
  (:issue:`179`, via :pr:`180`).
* Added :code:`COO.any` and :code:`COO.all` methods (:pr:`175`).
* Indexing for :code:`COO` now accepts a single one-dimensional array index
  (:pr:`172`).
* The fill-value can now be something other than zero or :code:`False`
  (:pr:`165`).
* Added a :code:`sparse.roll` function (:pr:`160`).
* Numba code now releases the GIL. This leads to better multi-threaded
  performance in Dask (:pr:`159`).
* A number of bugs occurred, so to resolve them, :code:`COO.coords.dtype` is
  always :code:`np.int64`.  :code:`COO`, therefore, uses more memory than
  before (:pr:`158`).
* Add support for saving and loading :code:`COO` files from disk (:issue:`153`,
  via :pr:`154`).
* Support :code:`COO.nonzero` and :code:`np.argwhere` (:issue:`145`, via
  :pr:`148`).
* Allow faux in-place operations (:issue:`80`, via :pr:`146`).
* :code:`COO` is now always canonical (:pr:`141`).
* Improve indexing performance (:pr:`128`).
* Improve element-wise performance (:pr:`127`).
* Reductions now support a negative axis (:issue:`117`, via :pr:`118`).
* Match behaviour of :code:`ufunc.reduce` from NumPy (:issue:`107`, via
  :pr:`108`).

0.3.1 / 2018-04-12
------------------

* Fix packaging error (:pr:`138`).

0.3.0 / 2018-02-22
------------------

* Add NaN-skipping aggregations (:pr:`102`).
* Add equivalent to :code:`np.where` (:pr:`102`).
* N-input universal functions now work (:pr:`98`).
* Make :code:`dot` more consistent with NumPy (:pr:`96`).
* Create a base class :code:`SparseArray` (:pr:`92`).
* Minimum NumPy version is now 1.13 (:pr:`90`).
* Fix a bug where setting a :code:`DOK` element to zero did nothing
  (:issue:`93`, via :pr:`94`).

0.2.0 / 2018-01-25
------------------

* Support faster :code:`np.array(COO)` (:pr:`87`).
* Add :code:`DOK` type (:pr:`85`).
* Fix sum for large arrays (:issue:`82`, via :pr:`83`).
* Support :code:`.size` and :code:`.density` (:pr:`69`).
* Documentation added for the package (:pr:`43`).
* Minimum required SciPy version is now 0.19 (:pr:`70`).
* :code:`len(COO)` now works (:pr:`68`).
* :code:`scalar op COO` now works for all operators (:pr:`67`).
* Validate axes for :code:`.transpose()` (:pr:`61`).
* Extend indexing support (:pr:`57`).
* Add :code:`random` function for generating random sparse arrays (:pr:`41`).
* :code:`COO(COO)` now copies the original object (:pr:`55`).
* NumPy universal functions and reductions now work on :code:`COO` arrays
  (:pr:`49`).
* Fix concatenate and stack for large arrays (:issue:`32`, via :pr:`51`).
* Fix :code:`nnz` for scalars (:issue:`47`, via :pr:`48`).
* Support more operators and remove all special cases (:pr:`46`).
* Add support for :code:`triu` and :code:`tril` (:pr:`40`).
* Add support for Ellipsis (:code:`...`) and :code:`None` when indexing
  (:pr:`37`).
* Add support for bitwise bindary operations like :code:`&` and :code:`|`
  (:pr:`38`).
* Support broadcasting in element-wise operations (:pr:`35`).
