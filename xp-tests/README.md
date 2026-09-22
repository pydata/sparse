# Array API tests

The Numba task loads `numba_masking.py` to parameterize the pinned suite's
`test_getitem_masking` by source type. Its assertions are unchanged:

- `sparse-array` preserves sparse sources, including zero-dimensional arrays,
  and keeps the upstream shape, dtype, and mask strategies. This case runs
  without a skip or xfail.
- `numpy-scalar` isolates a NumPy scalar indexed by a sparse zero-dimensional
  boolean mask. This unsupported combination is the only masking entry in
  `Numba-array-api-xfails.txt`. It fails deterministically even with the small
  example budget used in CI.

The upstream helper sometimes extracts a scalar with `x[()]`. For Sparse,
this switches the source to NumPy's indexing implementation. Separating the
cases keeps that limitation from masking failures on sparse arrays. The
fixture restores the upstream strategies after each test and leaves all
other Array API tests unchanged. No external test checkout is edited.
