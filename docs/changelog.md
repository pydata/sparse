# Changelog


0.15.1 / 2024-01-10
-------------------
* Fix regression where with XArray by supporting all API functions via the Array API standard. (PR [#622](https://github.com/pydata/sparse/pull/622) thanks [@hameerabbasi](https://github.com/hameerabbasi))

0.15.0 / 2024-01-09
-------------------
* Fix regression where [`DeprecationWarning`][]s were being fired unexpectedly. (PR [#581](https://github.com/pydata/sparse/pull/581) thanks [@hameerabbasi](https://github.com/hameerabbasi))
* Extended [`sparse.einsum`][] support (PR [#579](https://github.com/pydata/sparse/pull/579) thanks [@HadrienNU](https://github.com/HadrienNU))
* General code clean-up (PR [#586](https://github.com/pydata/sparse/pull/586) thanks [@MHRasmy](https://github.com/MHRasmy), PR [#598](https://github.com/pydata/sparse/pull/598) thanks [@jamestwebber](https://github.com/jamestwebber))
* Bug fixes with respect to NumPy compatibility  (PR [#598](https://github.com/pydata/sparse/pull/598) thanks [@hameerabbasi](https://github.com/hameerabbasi), PR [#609](https://github.com/pydata/sparse/pull/609) thanks [@Illviljan](https://github.com/Illviljan), PR [#620](https://github.com/pydata/sparse/pull/620) thanks [@mtsokol](https://github.com/mtsokol))
* Bug fixes with respect to GCXS (PR [#611](https://github.com/pydata/sparse/pull/611) thanks [@EuGig](https://github.com/EuGig), PR [#601](https://github.com/pydata/sparse/pull/601) thanks [@jamestwebber](https://github.com/jamestwebber))
* `Array API standard <https://data-apis.org/array-api/latest/>`_ support (PR [#612](https://github.com/pydata/sparse/pull/612), PR [#613](https://github.com/pydata/sparse/pull/613), PR [#614](https://github.com/pydata/sparse/pull/614), PR [#615](https://github.com/pydata/sparse/pull/615), PR [#619](https://github.com/pydata/sparse/pull/619), PR [#620](https://github.com/pydata/sparse/pull/620) thanks [@mtsokol](https://github.com/mtsokol))
* ``matrepr`` support for display of sparse data (PR [#605](https://github.com/pydata/sparse/pull/605), PR [#606](https://github.com/pydata/sparse/pull/606) thanks [@alugowski](https://github.com/alugowski)).
* Larger code clean-up with Ruff formatter and linter (PR [#617](https://github.com/pydata/sparse/pull/617), PR [#621](https://github.com/pydata/sparse/pull/621) thanks [@hameerabbasi](https://github.com/hameerabbasi))
* Packaging and maintenance (PR [#616](https://github.com/pydata/sparse/pull/616), :commit:`b5954e68d3d6e35a62f7401d1d4fb84ea04414dd`, :commit:`dda93d3ea9521881c721c3ba875c769c9c5a79d4` thanks [@hameerabbasi](https://github.com/hameerabbasi))


0.14.0 / 2023-02-24
-------------------
* [`sparse.einsum`][] support (PR [#564](https://github.com/pydata/sparse/pull/564) thanks
  [@jcmgray](https://github.com/jcmgray))
* Some bug-fixes (PR [#524](https://github.com/pydata/sparse/pull/524), PR [#527](https://github.com/pydata/sparse/pull/527), PR [#555](https://github.com/pydata/sparse/pull/555) thanks [@hameerabbasi](https://github.com/hameerabbasi), PR [#569](https://github.com/pydata/sparse/pull/569), thanks [@jamestwebber](https://github.com/jamestwebber), PR [#534](https://github.com/pydata/sparse/pull/534), thanks [@sarveshbhatnagar](https://github.com/sarveshbhatnagar))
* Some performance improvements (PR [#570](https://github.com/pydata/sparse/pull/570), thanks [@jamestwebber](https://github.com/jamestwebber), PR [#540](https://github.com/pydata/sparse/pull/540), thanks [@smldub](https://github.com/smldub)).
* Miscellaneous maintenance fixes.

0.13.0 / 2021-08-28
-------------------

* [`sparse.GCXS`][] improvements and changes. (PR [#448](https://github.com/pydata/sparse/pull/448), PR [#450](https://github.com/pydata/sparse/pull/450), PR [#455](https://github.com/pydata/sparse/pull/455), thanks
  [@sayandip18](https://github.com/sayandip18)).
* Maintainence fixes (PR [#462](https://github.com/pydata/sparse/pull/462), PR [#466](https://github.com/pydata/sparse/pull/466), :commit:`1ccb85da581be65a0345b399e00fd3c325700d95`,
  :commit:`5547b4e92dc8d61492e9dc10ba00175c1a6637fa`
  :commit:`00c0e5514de2aab8b9a0be16b5da470b091d9eb9`, :commit:`fcd3020dd08c7022a44f709173fe23969d3e8f7c`,
  thanks [@hameerabbasi](https://github.com/hameerabbasi))
* [`sparse.DOK.from_scipy_sparse`][] method (PR [#464](https://github.com/pydata/sparse/pull/464), Issue [#463](https://github.com/pydata/sparse/issues/463), thanks
  [@hameerabbasi](https://github.com/hameerabbasi)).
* Black re-formatting (PR [#471](https://github.com/pydata/sparse/pull/471), PR [#484](https://github.com/pydata/sparse/pull/484), thanks [@GenevieveBuckley](https://github.com/GenevieveBuckley), [@sayandip18](https://github.com/sayandip18))
* Add [`sparse.pad`][] (PR [#474](https://github.com/pydata/sparse/pull/474), Issue [#438](https://github.com/pydata/sparse/issues/438), thanks [@H4R5H1T-007](https://github.com/H4R5H1T-007))
* Switch to GitHub Actions (:compare:`5547b4e92dc8d61492e9dc10ba00175c1a6637fa..a332f22c96a96e5ab9b4384342df67e8f3966f85`)
* Fix a number of bugs in format conversion. (PR [#504](https://github.com/pydata/sparse/pull/504), Issue [#503](https://github.com/pydata/sparse/issues/503), thanks
  [@hameerabbasi](https://github.com/hameerabbasi))
* Fix bug in [`sparse.matmul`][] for higher-dimensional arrays. (PR [#508](https://github.com/pydata/sparse/pull/508),
  Issue [#506](https://github.com/pydata/sparse/issues/506), thanks [@sayandip18](https://github.com/sayandip18)).
* Fix scalar conversion to COO (Issue [#510](https://github.com/pydata/sparse/issues/510), PR [#511](https://github.com/pydata/sparse/pull/511), thanks [@hameerabbasi](https://github.com/hameerabbasi))
* Fix OOB memory accesses (Issue [#515](https://github.com/pydata/sparse/issues/515), :commit:`1e24a7e29786e888dee4c02153309986ae4b5dde`
  thanks [@hameerabbasi](https://github.com/hameerabbasi))
* Fixes element-wise ops with scalar COO array. (Issue [#505](https://github.com/pydata/sparse/issues/505), :commit:`5211441ec685233657ab7156f99eb67e660cee86`,
  thanks [@hameerabbasi](https://github.com/hameerabbasi))
* Fix scalar broadcast_to with ``nnz==0``. (Issue [#513](https://github.com/pydata/sparse/issues/513), :commit:`bfabaa0805e811884e79c4bdbfd14316986d65e4`,
  thanks [@hameerabbasi](https://github.com/hameerabbasi))
* Add order parameter to ``{zero, ones, full}[_like]``. (Issue [#514](https://github.com/pydata/sparse/issues/514), :commit:`37de1d0141c4375962ecdf18337c2dd0f667b60c`,
  thanks [@hameerabbasi](https://github.com/hameerabbasi))
* Fix tensordot typing bugs. (Issue [#493](https://github.com/pydata/sparse/issues/493), Issue [#499](https://github.com/pydata/sparse/issues/499), :commit:`37de1d0141c4375962ecdf18337c2dd0f667b60c`,
  thanks [@hameerabbasi](https://github.com/hameerabbasi)).

0.12.0 / 2021-03-19
-------------------

There are a number of large changes in this release. For example, we have implemented the
[`sparse.GCXS`][] type, and its specializations `CSR` and `CSC`. We plan on gradually improving
the performance of these.

* A number of [`sparse.GCXS`][] fixes and additions (PR [#409](https://github.com/pydata/sparse/pull/409), PR [#407](https://github.com/pydata/sparse/pull/407), PR [#414](https://github.com/pydata/sparse/pull/414),
  PR [#417](https://github.com/pydata/sparse/pull/417), PR [#419](https://github.com/pydata/sparse/pull/419) thanks [@daletovar](https://github.com/daletovar))
* Ability to change the index dtype for better storage characteristics. (PR [#441](https://github.com/pydata/sparse/pull/441),
  thanks [@daletovar](https://github.com/daletovar))
* Some work on [`sparse.DOK`][] arrays to bring them closer to the other formats (PR [#435](https://github.com/pydata/sparse/pull/435),
  PR [#437](https://github.com/pydata/sparse/pull/437), PR [#439](https://github.com/pydata/sparse/pull/439), PR [#440](https://github.com/pydata/sparse/pull/440), thanks [@DragaDoncila](https://github.com/DragaDoncila))
* `CSR` and `CSC` specializations of [`sparse.GCXS`][] (PR [#442](https://github.com/pydata/sparse/pull/442), thanks [@ivirshup](https://github.com/ivirshup))
  For now, this is experimental undocumented API, and subject to change.
* Fix a number of bugs (PR [#407](https://github.com/pydata/sparse/pull/407), Issue [#406](https://github.com/pydata/sparse/issues/406))
* Add `nnz` parameter to [`sparse.random`][] (PR [#410](https://github.com/pydata/sparse/pull/410), thanks [@emilmelnikov](https://github.com/emilmelnikov))

0.11.2 / 2020-09-04
-------------------

* Fix `TypingError` on [`sparse.dot`][] with complex dtypes. (Issue [#403](https://github.com/pydata/sparse/issues/403), PR [#404](https://github.com/pydata/sparse/pull/404))

0.11.1 / 2020-08-31
-------------------

* Fix [`ValueError`][] on [`sparse.dot`][] with extremely small values. (Issue [#398](https://github.com/pydata/sparse/issues/398), PR [#399](https://github.com/pydata/sparse/pull/399))

0.11.0 / 2020-08-18
-------------------

* Improve the performance of [`sparse.dot`][]. (Issue [#331](https://github.com/pydata/sparse/issues/331), PR [#389](https://github.com/pydata/sparse/pull/389), thanks [@daletovar](https://github.com/daletovar))
* Added the [`sparse.COO.swapaxes`][] method. (PR [#344](https://github.com/pydata/sparse/pull/344), thanks [@lueckem](https://github.com/lueckem))
* Added multi-axis 1-D indexing support. (PR [#343](https://github.com/pydata/sparse/pull/343), thanks [@mikeymezher](https://github.com/mikeymezher))
* Fix `outer` for arrays that weren't one-dimensional. (Issue [#346](https://github.com/pydata/sparse/issues/346), PR [#347](https://github.com/pydata/sparse/pull/347))
* Add `casting` kwarg to [`sparse.COO.astype`][]. (Issue [#391](https://github.com/pydata/sparse/issues/391), PR [#392](https://github.com/pydata/sparse/pull/392))
* Fix for [`sparse.COO`][] constructor accepting invalid inputs. (Issue [#385](https://github.com/pydata/sparse/issues/385), PR [#386](https://github.com/pydata/sparse/pull/386))

0.10.0 / 2020-05-13
-------------------

* Fixed a bug where converting an empty DOK array to COO leads
  to an incorrect dtype. (Issue [#314](https://github.com/pydata/sparse/issues/314), PR [#315](https://github.com/pydata/sparse/pull/315))
* Change code formatter to black. (PR [#284](https://github.com/pydata/sparse/pull/284))
* Add [`sparse.COO.flatten`] and `outer`. (Issue [#316](https://github.com/pydata/sparse/issues/316), PR [#317](https://github.com/pydata/sparse/pull/317)).
* Remove broadcasting restriction between sparse arrays and dense arrays.
  (Issue [#306](https://github.com/pydata/sparse/issues/306), PR [#318](https://github.com/pydata/sparse/pull/318))
* Implement deterministic dask tokenization. (Issue [#300](https://github.com/pydata/sparse/issues/300), PR [#320](https://github.com/pydata/sparse/pull/320), thanks
  [@danielballan](https://github.com/danielballan))
* Improve testing around densification (PR [#321](https://github.com/pydata/sparse/pull/321), thanks
  [@danielballan](https://github.com/danielballan))
* Simplify Numba extension. (PR [#324](https://github.com/pydata/sparse/pull/324), thanks [@eric-wieser](https://github.com/eric-wieser)).
* Respect ``copy=False`` in ``astype`` (PR [#328](https://github.com/pydata/sparse/pull/328), thanks [@eric-wieser](https://github.com/eric-wieser)).
* Replace linear_loc with ravel_multi_index, which is 3x faster. (PR [#330](https://github.com/pydata/sparse/pull/330),
  thanks [@eric-wieser](https://github.com/eric-wieser)).
* Add error msg to tensordot operation when ``ndim==0`` (Issue [#332](https://github.com/pydata/sparse/issues/332),
  PR [#333](https://github.com/pydata/sparse/pull/333), thanks [@guilhermeleobas](https://github.com/guilhermeleobas)).
* Maintainence fixes for Sphinx 3.0 and Numba 0.49, and dropping support for
  Python 3.5. (PR [#337](https://github.com/pydata/sparse/pull/337)).
* Fixed signature for [numpy.clip][].

0.9.1 / 2020-01-23
------------------

* Fixed a bug where indexing with an empty list could lead
  to issues. (Issue [#281](https://github.com/pydata/sparse/issues/281), PR [#282](https://github.com/pydata/sparse/pull/282))
* Change code formatter to black. (PR [#284](https://github.com/pydata/sparse/pull/284))
* Add the [`sparse.diagonal`][] and [`sparse.diagonalize`][] functions.
  (Issue [#288](https://github.com/pydata/sparse/issues/288), PR [#289](https://github.com/pydata/sparse/pull/289), thanks [@pettni](https://github.com/pettni))
* Add HTML repr for notebooks. (PR [#283](https://github.com/pydata/sparse/pull/283), thanks [@daletovar](https://github.com/daletovar))
* Avoid making copy of ``coords`` when making a new [`sparse.COO`][]
  array.
* Add stack and concatenate for GCXS. (Issue [#301](https://github.com/pydata/sparse/issues/301), PR [#303](https://github.com/pydata/sparse/pull/303), thanks
  [@daletovar](https://github.com/daletovar)).
* Fix issue where functions dispatching to an attribute access wouldn't
  work with ``__array_function__``. (Issue [#308](https://github.com/pydata/sparse/issues/308), PR [#309](https://github.com/pydata/sparse/pull/309)).
* Add partial support for constructing and mirroring [`sparse.COO`][] objects to
  Numba.

0.8.0 / 2019-08-26
------------------

This release switches to Numba's new typed lists, a lot of
back-end work with the CI infrastructure, so Linux, macOS
and Windows are officially tested. It also includes bug fixes.

It also adds in-progress, not yet public support for the GCXS
format, which is a generalisation of CSR/CSC. (huge thanks to
[@daletovar](https://github.com/daletovar))

* Fixed a bug where an array with size == 1 and nnz == 0
  could not be broadcast. (Issue [#242](https://github.com/pydata/sparse/issues/242), PR [#243](https://github.com/pydata/sparse/pull/243))
* Add ``std`` and ``var``. (PR [#244](https://github.com/pydata/sparse/pull/244))
* Move to Azure Pipelines with CI for Windows, macOS and
  Linux. (PR [#245](https://github.com/pydata/sparse/pull/245), PR [#246](https://github.com/pydata/sparse/pull/246), PR [#247](https://github.com/pydata/sparse/pull/247), PR [#248](https://github.com/pydata/sparse/pull/248))
* Add ``resize``, and change ``reshape`` so it raises a
  ``ValueError`` on shapes that don't correspond to the
  same size. (Issue [#241](https://github.com/pydata/sparse/issues/241), Issue [#250](https://github.com/pydata/sparse/issues/250), PR [#256](https://github.com/pydata/sparse/pull/256)
  thanks, [@daletovar](https://github.com/daletovar))
* Add ``isposinf`` and ``isneginf``. (Issue [#252](https://github.com/pydata/sparse/issues/252), PR [#253](https://github.com/pydata/sparse/pull/253))
* Fix ``tensordot`` when nnz = 0. (Issue [#255](https://github.com/pydata/sparse/issues/255), PR [#256](https://github.com/pydata/sparse/pull/256))
* Modifications to ``__array_function__`` to allow for sparse
  XArrays. (PR [#261](https://github.com/pydata/sparse/pull/261), thanks [@nvictus](https://github.com/nvictus))
* Add not-yet-public support for GCXS. (PR [#258](https://github.com/pydata/sparse/pull/258), thanks [@daletovar](https://github.com/daletovar))
* Improvements to ``__array_function__``. (PR [#267](https://github.com/pydata/sparse/pull/267), PR [#272](https://github.com/pydata/sparse/pull/272), thanks
  [@crusaderky](https://github.com/crusaderky))
* Convert all Numba lists to typed lists. (PR [#264](https://github.com/pydata/sparse/pull/264))
* Why write code when it exists elsewhere? (PR [#277](https://github.com/pydata/sparse/pull/277))
* Fix some element-wise operations with scalars. (PR [#278](https://github.com/pydata/sparse/pull/278))
* Private modules should be private, and tests should be in the package.
  (PR [#280](https://github.com/pydata/sparse/pull/280))


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

* Fixed a bug where going between [`sparse.DOK`][] and
  [`sparse.COO`][] caused fill-values to be lost.
  (Issue [#225](https://github.com/pydata/sparse/issues/225), PR [#226](https://github.com/pydata/sparse/pull/226)).
* Fixed warning for a matrix that was incorrectly considered
  too dense. (Issue [#228](https://github.com/pydata/sparse/issues/228), PR [#229](https://github.com/pydata/sparse/pull/229))
* Fixed some warnings in Python 3.7, the fix was needed.
  in preparation for Python 3.8. (PR [#233](https://github.com/pydata/sparse/pull/233), thanks [@nils-werner](https://github.com/nils-werner))
* Drop support for Python 2.7 (Issue [#234](https://github.com/pydata/sparse/issues/234), PR [#235](https://github.com/pydata/sparse/pull/235), thanks
  [@hugovk](https://github.com/hugovk))
* Clearer error messages (Issue [#230](https://github.com/pydata/sparse/issues/230), Issue [#231](https://github.com/pydata/sparse/issues/231), PR [#232](https://github.com/pydata/sparse/pull/232))
* Restructure requirements.txt files. (PR [#236](https://github.com/pydata/sparse/pull/236))
* Support fill-value in reductions in specific cases. (Issue [#237](https://github.com/pydata/sparse/issues/237), PR [#238](https://github.com/pydata/sparse/pull/238))
* Add ``__array_function__`` support. (PR [#239](https://github.com/pydata/sparse/pull/239), thanks, [@pentschev](https://github.com/pentschev))
* Cleaner code! (PR [#240](https://github.com/pydata/sparse/pull/240))

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
  convert to a ``COO``. (Issue [#199](https://github.com/pydata/sparse/issues/199), PR [#200](https://github.com/pydata/sparse/pull/200)).
* Make sure that ``sparse @ sparse`` returns a sparse array. (Issue [#201](https://github.com/pydata/sparse/issues/201), PR [#203](https://github.com/pydata/sparse/pull/203))
* Bring ``operator.matmul`` behaviour in line with NumPy for ``ndim > 2``.
  (Issue [#202](https://github.com/pydata/sparse/issues/202), PR [#204](https://github.com/pydata/sparse/pull/204), PR [#217](https://github.com/pydata/sparse/pull/217))
* Make sure ``dtype`` is preserved with the ``out`` kwarg. (Issue [#205](https://github.com/pydata/sparse/issues/205), PR [#206](https://github.com/pydata/sparse/pull/206))
* Fix integer overflow in ``reduce`` on Windows. (Issue [#207](https://github.com/pydata/sparse/issues/207), PR [#208](https://github.com/pydata/sparse/pull/208))
* Disallow auto-densification. (Issue [#218](https://github.com/pydata/sparse/issues/218), PR [#220](https://github.com/pydata/sparse/pull/220))
* Add auto-densification configuration, and a configurable warning for checking
  if the array is too dense. (PR [#210](https://github.com/pydata/sparse/pull/210), PR [#213](https://github.com/pydata/sparse/pull/213))
* Add pruning of fill-values to COO constructor. (PR [#221](https://github.com/pydata/sparse/pull/221))

0.5.0 / 2018-10-12
------------------

* Added `COO.real`, `COO.imag`, and `COO.conj` (PR [#196](https://github.com/pydata/sparse/pull/196)).
* Added `sparse.kron` function (PR [#194](https://github.com/pydata/sparse/pull/194), PR [#195](https://github.com/pydata/sparse/pull/195)).
* Added `order` parameter to `COO.reshape` to make it work with
  `np.reshape` (PR [#193](https://github.com/pydata/sparse/pull/193)).
* Added `COO.mean` and `sparse.nanmean` (PR [#190](https://github.com/pydata/sparse/pull/190)).
* Added `sparse.full` and `sparse.full_like` (PR [#189](https://github.com/pydata/sparse/pull/189)).
* Added `COO.clip` method (PR [#185](https://github.com/pydata/sparse/pull/185)).
* Added `COO.copy` method, and changed pickle of `COO` to not
  include its cache (PR [#184](https://github.com/pydata/sparse/pull/184)).
* Added `sparse.eye`, `sparse.zeros`, `sparse.zeros_like`,
  `sparse.ones`, and `sparse.ones_like` (PR [#183](https://github.com/pydata/sparse/pull/183)).

0.4.1 / 2018-09-12
------------------

* Allow mixed `ndarray`-`COO` operations if the result is sparse
  (Issue [#124](https://github.com/pydata/sparse/issues/124), via PR [#182](https://github.com/pydata/sparse/pull/182)).
* Allow specifying a fill-value when converting from NumPy arrays
  (Issue [#179](https://github.com/pydata/sparse/issues/179), via PR [#180](https://github.com/pydata/sparse/pull/180)).
* Added `COO.any` and `COO.all` methods (PR [#175](https://github.com/pydata/sparse/pull/175)).
* Indexing for `COO` now accepts a single one-dimensional array index
  (PR [#172](https://github.com/pydata/sparse/pull/172)).
* The fill-value can now be something other than zero or `False`
  (PR [#165](https://github.com/pydata/sparse/pull/165)).
* Added a `sparse.roll` function (PR [#160](https://github.com/pydata/sparse/pull/160)).
* Numba code now releases the GIL. This leads to better multi-threaded
  performance in Dask (PR [#159](https://github.com/pydata/sparse/pull/159)).
* A number of bugs occurred, so to resolve them, `COO.coords.dtype` is
  always `np.int64`.  `COO`, therefore, uses more memory than
  before (PR [#158](https://github.com/pydata/sparse/pull/158)).
* Add support for saving and loading `COO` files from disk (Issue [#153](https://github.com/pydata/sparse/issues/153),
  via PR [#154](https://github.com/pydata/sparse/pull/154)).
* Support `COO.nonzero` and `np.argwhere` (Issue [#145](https://github.com/pydata/sparse/issues/145), via
  PR [#148](https://github.com/pydata/sparse/pull/148)).
* Allow faux in-place operations (Issue [#80](https://github.com/pydata/sparse/issues/80), via PR [#146](https://github.com/pydata/sparse/pull/146)).
* `COO` is now always canonical (PR [#141](https://github.com/pydata/sparse/pull/141)).
* Improve indexing performance (PR [#128](https://github.com/pydata/sparse/pull/128)).
* Improve element-wise performance (PR [#127](https://github.com/pydata/sparse/pull/127)).
* Reductions now support a negative axis (Issue [#117](https://github.com/pydata/sparse/issues/117), via PR [#118](https://github.com/pydata/sparse/pull/118)).
* Match behaviour of `ufunc.reduce` from NumPy (Issue [#107](https://github.com/pydata/sparse/issues/107), via
  PR [#108](https://github.com/pydata/sparse/pull/108)).

0.3.1 / 2018-04-12
------------------

* Fix packaging error (PR [#138](https://github.com/pydata/sparse/pull/138)).

0.3.0 / 2018-02-22
------------------

* Add NaN-skipping aggregations (PR [#102](https://github.com/pydata/sparse/pull/102)).
* Add equivalent to `np.where` (PR [#102](https://github.com/pydata/sparse/pull/102)).
* N-input universal functions now work (PR [#98](https://github.com/pydata/sparse/pull/98)).
* Make `dot` more consistent with NumPy (PR [#96](https://github.com/pydata/sparse/pull/96)).
* Create a base class `SparseArray` (PR [#92](https://github.com/pydata/sparse/pull/92)).
* Minimum NumPy version is now 1.13 (PR [#90](https://github.com/pydata/sparse/pull/90)).
* Fix a bug where setting a `DOK` element to zero did nothing
  (Issue [#93](https://github.com/pydata/sparse/issues/93), via PR [#94](https://github.com/pydata/sparse/pull/94)).

0.2.0 / 2018-01-25
------------------

* Support faster `np.array(COO)` (PR [#87](https://github.com/pydata/sparse/pull/87)).
* Add `DOK` type (PR [#85](https://github.com/pydata/sparse/pull/85)).
* Fix sum for large arrays (Issue [#82](https://github.com/pydata/sparse/issues/82), via PR [#83](https://github.com/pydata/sparse/pull/83)).
* Support `.size` and `.density` (PR [#69](https://github.com/pydata/sparse/pull/69)).
* Documentation added for the package (PR [#43](https://github.com/pydata/sparse/pull/43)).
* Minimum required SciPy version is now 0.19 (PR [#70](https://github.com/pydata/sparse/pull/70)).
* `len(COO)` now works (PR [#68](https://github.com/pydata/sparse/pull/68)).
* `scalar op COO` now works for all operators (PR [#67](https://github.com/pydata/sparse/pull/67)).
* Validate axes for `.transpose()` (PR [#61](https://github.com/pydata/sparse/pull/61)).
* Extend indexing support (PR [#57](https://github.com/pydata/sparse/pull/57)).
* Add `random` function for generating random sparse arrays (PR [#41](https://github.com/pydata/sparse/pull/41)).
* `COO(COO)` now copies the original object (PR [#55](https://github.com/pydata/sparse/pull/55)).
* NumPy universal functions and reductions now work on `COO` arrays
  (PR [#49](https://github.com/pydata/sparse/pull/49)).
* Fix concatenate and stack for large arrays (Issue [#32](https://github.com/pydata/sparse/issues/32), via PR [#51](https://github.com/pydata/sparse/pull/51)).
* Fix `nnz` for scalars (Issue [#47](https://github.com/pydata/sparse/issues/47), via PR [#48](https://github.com/pydata/sparse/pull/48)).
* Support more operators and remove all special cases (PR [#46](https://github.com/pydata/sparse/pull/46)).
* Add support for `triu` and `tril` (PR [#40](https://github.com/pydata/sparse/pull/40)).
* Add support for Ellipsis (`...`) and `None` when indexing
  (PR [#37](https://github.com/pydata/sparse/pull/37)).
* Add support for bitwise bindary operations like `&` and `|`
  (PR [#38](https://github.com/pydata/sparse/pull/38)).
* Support broadcasting in element-wise operations (PR [#35](https://github.com/pydata/sparse/pull/35)).
