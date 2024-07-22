Changelog
=========

.. currentmodule:: sparse

0.15.1 / 2024-01-10
-------------------
* Fix regression where with XArray by supporting all API functions via the Array API standard. (:pr:`622` thanks :ghuser:`hameerabbasi`)

0.15.0 / 2024-01-09
-------------------
* Fix regression where :obj:`DeprecationWarning`s were being fired unexpectedly. (:pr:`581` thanks :ghuser:`hameerabbasi`)
* Extended :obj:`sparse.einsum` support (:pr:`579` thanks :ghuser:`HadrienNU`)
* General code clean-up (:pr:`586` thanks :ghuser:`MHRasmy`, :pr:`598` thanks :ghuser:`jamestwebber`)
* Bug fixes with respect to NumPy compatibility  (:pr:`598` thanks :ghuser:`hameerabbasi`, :pr:`609` thanks :ghuser:`Illviljan`, :pr:`620` thanks :ghuser:`mtsokol`)
* Bug fixes with respect to GCXS (:pr:`611` thanks :ghuser:`EuGig`, :pr:`601` thanks :ghuser:`jamestwebber`)
* `Array API standard <https://data-apis.org/array-api/latest/>`_ support (:pr:`612`, :pr:`613`, :pr:`614`, :pr:`615`, :pr:`619`, :pr:`620` thanks :ghuser:`mtsokol`)
* ``matrepr`` support for display of sparse data (:pr:`605`, :pr:`606` thanks :ghuser:`alugowski`).
* Larger code clean-up with Ruff formatter and linter (:pr:`617`, :pr:`621` thanks :ghuser:`hameerabbasi`)
* Packaging and maintenance (:pr:`616`, :commit:`b5954e68d3d6e35a62f7401d1d4fb84ea04414dd`, :commit:`dda93d3ea9521881c721c3ba875c769c9c5a79d4` thanks :ghuser:`hameerabbasi`)


0.14.0 / 2023-02-24
-------------------
* :obj:`sparse.einsum` support (:pr:`564` thanks
  :ghuser:`jcmgray`)
* Some bug-fixes (:pr:`524`, :pr:`527`, :pr:`555` thanks :ghuser:`hameerabbasi`, :pr:`569`, thanks :ghuser:`jamestwebber`, :pr:`534`, thanks :ghuser:`sarveshbhatnagar`)
* Some performance improvements (:pr:`570`, thanks :ghuser:`jamestwebber`, :pr:`540`, thanks :ghuser:`smldub`).
* Miscellaneous maintenance fixes.

0.13.0 / 2021-08-28
-------------------

* GCXS improvements and changes. (:pr:`448`, :pr:`450`, :pr:`455`, thanks
  :ghuser:`sayandip18`).
* Maintainence fixes (:pr:`462`, :pr:`466`, :commit:`1ccb85da581be65a0345b399e00fd3c325700d95`,
  :commit:`5547b4e92dc8d61492e9dc10ba00175c1a6637fa`
  :commit:`00c0e5514de2aab8b9a0be16b5da470b091d9eb9`, :commit:`fcd3020dd08c7022a44f709173fe23969d3e8f7c`,
  thanks :ghuser:`hameerabbasi`)
* :obj:`sparse.DOK.from_scipy_sparse` method (:pr:`464`, :issue:`463`, thanks
  :ghuser:`hameerabbasi`).
* Black re-formatting (:pr:`471`, :pr:`484`, thanks :ghuser:`GenevieveBuckley`, :ghuser:`sayandip18`)
* Add :obj:`sparse.pad` (:pr:`474`, :issue:`438`, thanks :ghuser:`H4R5H1T-007`)
* Switch to GitHub Actions (:compare:`5547b4e92dc8d61492e9dc10ba00175c1a6637fa..a332f22c96a96e5ab9b4384342df67e8f3966f85`)
* Fix a number of bugs in format conversion. (:pr:`504`, :issue:`503`, thanks
  :ghuser:`hameerabbasi`)
* Fix bug in :obj:`sparse.matmul` for higher-dimensional arrays. (:pr:`508`,
  :issue:`506`, thanks :ghuser:`sayandip18`).
* Fix scalar conversion to COO (:issue:`510`, :pr:`511`, thanks :ghuser:`hameerabbasi`)
* Fix OOB memory accesses (:issue:`515`, :commit:`1e24a7e29786e888dee4c02153309986ae4b5dde`
  thanks :ghuser:`hameerabbasi`)
* Fixes element-wise ops with scalar COO array. (:issue:`505`, :commit:`5211441ec685233657ab7156f99eb67e660cee86`,
  thanks :ghuser:`hameerabbasi`)
* Fix scalar broadcast_to with ``nnz==0``. (:issue:`513`, :commit:`bfabaa0805e811884e79c4bdbfd14316986d65e4`,
  thanks :ghuser:`hameerabbasi`)
* Add order parameter to ``{zero, ones, full}[_like]``. (:issue:`514`, :commit:`37de1d0141c4375962ecdf18337c2dd0f667b60c`,
  thanks :ghuser:`hameerabbasi`)
* Fix tensordot typing bugs. (:issue:`493`, :issue:`499`, :commit:`37de1d0141c4375962ecdf18337c2dd0f667b60c`,
  thanks :ghuser:`hameerabbasi`).

0.12.0 / 2021-03-19
-------------------

There are a number of large changes in this release. For example, we have implemented the
:obj:`GCXS` type, and its specializations :obj:`CSR` and :obj:`CSC`. We plan on gradually improving
the performance of these.

* A number of :obj:`GCXS` fixes and additions (:pr:`409`, :pr:`407`, :pr:`414`,
  :pr:`417`, :pr:`419` thanks :ghuser:`daletovar`)
* Ability to change the index dtype for better storage characteristics. (:pr:`441`,
  thanks :ghuser:`daletovar`)
* Some work on :obj:`DOK` arrays to bring them closer to the other formats (:pr:`435`,
  :pr:`437`, :pr:`439`, :pr:`440`, thanks :ghuser:`DragaDoncila`)
* :obj:`CSR` and :obj:`CSC` specializations of :obj:`GCXS` (:pr:`442`, thanks :ghuser:`ivirshup`)
  For now, this is experimental undocumented API, and subject to change.
* Fix a number of bugs (:pr:`407`, :issue:`406`)
* Add ``nnz`` parameter to :obj:`sparse.random` (:pr:`410`, thanks :ghuser:`emilmelnikov`)

0.11.2 / 2020-09-04
-------------------

* Fix :obj:`TypingError` on :obj:`sparse.dot` with complex dtypes. (:issue:`403`, :pr:`404`)

0.11.1 / 2020-08-31
-------------------

* Fix :obj:`ValueError` on :obj:`sparse.dot` with extremely small values. (:issue:`398`, :pr:`399`)

0.11.0 / 2020-08-18
-------------------

* Improve the performance of :obj:`sparse.dot`. (:issue:`331`, :pr:`389`, thanks :ghuser:`daletovar`)
* Added the :obj:`COO.swapaxes` method. (:pr:`344`, thanks :ghuser:`lueckem`)
* Added multi-axis 1-D indexing support. (:pr:`343`, thanks :ghuser:`mikeymezher`)
* Fix :obj:`outer` for arrays that weren't one-dimensional. (:issue:`346`, :pr:`347`)
* Add ``casting`` kwarg to :obj:`COO.astype`. (:issue:`391`, :pr:`392`)
* Fix for :obj:`COO` constructor accepting invalid inputs. (:issue:`385`, :pr:`386`)

0.10.0 / 2020-05-13
-------------------

* Fixed a bug where converting an empty DOK array to COO leads
  to an incorrect dtype. (:issue:`314`, :pr:`315`)
* Change code formatter to black. (:pr:`284`)
* Add :obj:`COO.flatten` and :obj:`sparse.outer`. (:issue:`316`, :pr:`317`).
* Remove broadcasting restriction between sparse arrays and dense arrays.
  (:issue:`306`, :pr:`318`)
* Implement deterministic dask tokenization. (:issue:`300`, :pr:`320`, thanks
  :ghuser:`danielballan`)
* Improve testing around densification (:pr:`321`, thanks
  :ghuser:`danielballan`)
* Simplify Numba extension. (:pr:`324`, thanks :ghuser:`eric-wieser`).
* Respect ``copy=False`` in ``astype`` (:pr:`328`, thanks :ghuser:`eric-wieser`).
* Replace linear_loc with ravel_multi_index, which is 3x faster. (:pr:`330`,
  thanks :ghuser:`eric-wieser`).
* Add error msg to tensordot operation when ``ndim==0`` (:issue:`332`,
  :pr:`333`, thanks :ghuser:`guilhermeleobas`).
* Maintainence fixes for Sphinx 3.0 and Numba 0.49, and dropping support for
  Python 3.5. (:pr:`337`).
* Fixed signature for :obj:`numpy.clip`.

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
