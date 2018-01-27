Sparse
======

Introduction
------------
In many scientific applications, arrays come up that are mostly empty or filled
with zeros. These arrays are aptly named *sparse arrays*. However, it is a matter
of choice as to how these are stored. One may store the full array, i.e., with all
the zeros included. This incurs a significant cost in terms of memory and
performance when working with these arrays.

An alternative way is to store them in a standalone data structure that keeps track
of only the nonzero entries. Often, this improves performance and memory consumption
but most operations on sparse arrays have to be re-written. :obj:`sparse` tries to
provide one such data structure. It isn't the only library that does this. Notably,
:obj:`scipy.sparse` achieves this, along with `pysparse <http://pysparse.sourceforge.net/>`_.

Motivation
----------
So why use :obj:`sparse`? Well, the other libraries mentioned are mostly limited to
two-dimensional arrays. In addition, inter-compatibility with :obj:`numpy` is
hit-or-miss. :obj:`sparse` strives to achieve inter-compatibility with
:obj:`numpy.ndarray`, and provide mostly the same API. It defers to :obj:`scipy.sparse`
when it is convenient to do so, and writes custom implementations of operations where
this isn't possible. It also supports general N-dimensional arrays.

.. toctree::
   :maxdepth: 3
   :hidden:

   install
   quickstart
   construct
   operations
   generated/sparse
   contribute
   changelog
