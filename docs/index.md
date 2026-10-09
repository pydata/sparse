---
hide:
  - navigation
  - toc
---

# Sparse
This project implements sparse arrays of arbitrary dimension on top of
[`numpy`][] and [`scipy.sparse`][]. [`sparse.COO`][] and [`sparse.DOK`][]
generalize the [`scipy.sparse.coo_matrix`][] and
[`scipy.sparse.dok_matrix`][] layouts. [`sparse.GCXS`][] extends compressed
sparse storage beyond two dimensions: for a two-dimensional array,
compressing rows (`compressed_axes=(0,)`) corresponds to
[`scipy.sparse.csr_matrix`][], while compressing columns
(`compressed_axes=(1,)`) corresponds to [`scipy.sparse.csc_matrix`][].
<br>
<br>
![Sparse](./assets/images/logo.png){width=20%, align=left}
<div class="grid" markdown>

  ![Sparse](./assets/images/conference-room-icon.png){width=10%, align=left}
  <a href="introduction/" style: class="card">Introduction </a>
  { .card }

  ![Sparse](./assets/images/install-software-download-icon.png){width=10%, align=left}
  <a href="install/" style: class="card">Install</a>
  { .card }

  ![Sparse](./assets/images/open-book-icon.png){width=10%, align=left}
  <a href="examples/" class="card">Tutorials</a>
  { .card }

  ![Sparse](./assets/images/check-list-icon.png){width=10%, align=left}
  <a href="how-to-guides/" class="card">How-to guides</a>
  { .card }

  ![Sparse](./assets/images/repair-fix-repairing-icon.png){width=10%, align=left}
  <a href="api/" style: class="card">API</a>
  { .card }

 ![Sparse](./assets/images/group-discussion-icon.png){width=10%, align=left}
  <a href="contributing" style: class="card">Contributing </a>
  { .card }

</div>
