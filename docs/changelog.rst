Changelog
=========

.. currentmodule:: sparse

0.9.1 / 2020-01-23
------------------

* Fixed a bug where indexing with an empty list could lead
  to issues. (:issue:`281`, :pr:`282`)
* Change code formatter to black. (:pr:`284`)
* Add the :obj:`diagonal` and :obj:`diagonalize` functions.
  (:issue:`288`, :pr:`289`, thanks :ghuser:`pettni`)
* Add HTML repr for notebooks. (:pr:`283`, thanks :ghuser:`daletovar`)
* Avoid making copy of ``coords`` when making a new :obj:`COO`
  array.
* Add stack and concatenate for GCXS. (:issue:`301`, :pr:`303`, thanks
  :ghuser:`daletovar`).
* Fix issue where functions dispatching to an attribute access wouldn't
  work with ``__array_function__``. (:issue:`308`, :pr:`309`).
* Add partial support for constructing and mirroring :obj:`COO` objects to
  Numba.

0.8.0 / 2019-08-26
------------------

This release switches to Numba's new typed lists, a lot of
back-end work with the CI infrastructure, so Linux, macOS
and Windows are officially tested. It also includes bug fixes.

It also adds in-progress, not yet public support for the GCXS
format, which is a generalisation of CSR/CSC. (huge thanks to
:ghuser:`daletovar`)

* Fixed a bug where an array with size == 1 and nnz == 0
  could not be broadcast. (:issue:`242`, :pr:`243`)
* Add ``std`` and ``var``. (:pr:`244`)
* Move to Azure Pipelines with CI for Windows, macOS and
  Linux. (:pr:`245`, :pr:`246`, :pr:`247`, :pr:`248`)
* Add ``resize``, and change ``reshape`` so it raises a
  ``ValueError`` on shapes that don't correspond to the
  same size. (:issue:`241`, :issue:`250`, :pr:`256`
  thanks, :ghuser:`daletovar`)
* Add ``isposinf`` and ``isneginf``. (:issue:`252`, :pr:`253`)
* Fix ``tensordot`` when nnz = 0. (:issue:`255`, :pr:`256`)
* Modifications to ``__array_function__`` to allow for sparse
  XArrays. (:pr:`261`, thanks :ghuser:`nvictus`)
* Add not-yet-public support for GCXS. (:pr:`258`, thanks :ghuser:`daletovar`)
* Improvements to ``__array_function__``. (:pr:`267`, :pr:`272`, thanks
  :ghuser:`crusaderky`)
* Convert all Numba lists to typed lists. (:pr:`264`)
* Why write code when it exists elsewhere? (:pr:`277`)
* Fix some element-wise operations with scalars. (:pr:`278`)
* Private modules should be private, and tests should be in the package.
  (:pr:`280`)


0.7.0 / 2019-03-14
------------------

This is a release that adds compatibility with NumPy's new
``__array_function__`` protocol, for details refer to
`NEP-18 <http://www.numpy.org/neps/nep-0018-array-function-protocol.html#coercion-to-a-numpy-array-as-a-catch-all-fallback>`_.

The other big change is that we dropped compatibility with
Python 2. Users on Python 2 should use version 0.6.0.

There are also some bug-fixes relating to fill-values.

This was mainly a contributor-driven release.

The full list of changes can be found below:

* Fixed a bug where going between :obj:`sparse.DOK` and
  :obj:`sparse.COO` caused fill-values to be lost.
  (:issue:`225`, :pr:`226`).
* Fixed warning for a matrix that was incorrectly considered
  too dense. (:issue:`228`, :pr:`229`)
* Fixed some warnings in Python 3.7, the fix was needed.
  in preparation for Python 3.8. (:pr:`233`, thanks :ghuser:`nils-werner`)
* Drop support for Python 2.7 (:issue:`234`, :pr:`235`, thanks
  :ghuser:`hugovk`)
* Clearer error messages (:issue:`230`, :issue:`231`, :pr:`232`)
* Restructure requirements.txt files. (:pr:`236`)
* Support fill-value in reductions in specific cases. (:issue:`237`, :pr:`238`)
* Add ``__array_function__`` support. (:pr:`239`, thanks, :ghuser:`pentschev`)
* Cleaner code! (:pr:`240`)

0.6.0 / 2018-12-19
------------------

This release breaks backward-compatibility. Previously, if arrays were fed into
NumPy functions, an attempt would be made to densify the array and apply the NumPy
function. This was unintended behaviour in most cases, with the array filling up
memory before raising a ``MemoryError`` if the array was too large.

We have now changed this behaviour so that a ``RuntimeError`` is now raised if
an attempt is made to automatically densify an array. To densify, use the explicit
``.todense()`` method.

* Fixed a bug where ``np.matrix`` could sometimes fail to
  convert to a ``COO``. (:issue:`199`, :pr:`200`).
* Make sure that ``sparse @ sparse`` returns a sparse array. (:issue:`201`, :pr:`203`)
* Bring ``operator.matmul`` behaviour in line with NumPy for ``ndim > 2``.
  (:issue:`202`, :pr:`204`, :pr:`217`)
* Make sure ``dtype`` is preserved with the ``out`` kwarg. (:issue:`205`, :pr:`206`)
* Fix integer overflow in ``reduce`` on Windows. (:issue:`207`, :pr:`208`)
* Disallow auto-densification. (:issue:`218`, :pr:`220`)
* Add auto-densification configuration, and a configurable warning for checking
  if the array is too dense. (:pr:`210`, :pr:`213`)
* Add pruning of fill-values to COO constructor. (:pr:`221`)

0.5.0 / 2018-10-12
------------------

* Added :code:`COO.real`, :code:`COO.imag`, and :code:`COO.conj` (:pr:`196`).
* Added :code:`sparse.kron` function (:pr:`194`, :pr:`195`).
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
