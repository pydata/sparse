Contributing to sparse
======================

General Guidelines
------------------
sparse is a community-driven project on GitHub. You can find our
`repository on GitHub <https://github.com/mrocklin/sparse>`_. Feel
free to open issues for new features or bugs, or open a pull request
to fix a bug or add a new feature.

If you haven't contributed to open-source before, we recommend you read
`this excellent guide by GitHub on how to contribute to open source
<https://opensource.guide/how-to-contribute/>`_. The guide is long,
so you can gloss over things you're familiar with.

If you're not already familiar with it, we follow the `fork and pull model
<https://help.github.com/articles/about-collaborative-development-models/>`_
on GitHub.

Running/Adding Unit Tests
-------------------------
It is best if all new functionality and/or bug fixes have unit tests added
with each use-case.

Since we support both Python 2.7 and Python 3.5 and newer, it is recommended
to test with at least these two versions before committing your code or opening
a pull request. We use `pytest <https://docs.pytest.org/en/latest/>`_ as our unit
testing framework, with the pytest-cov extension to check code coverage and
pytest-flake8 to check code style. You don't need to configure these extensions
yourself. Once you've configured your environment, you can just :code:`cd` to
the root of your repository and run

.. code-block:: bash

   py.test

Adding/Building the Documentation
---------------------------------
If a feature is stable and relatively finalized, it is time to add it to the
documentation. If you are adding any private/public functions, it is best to
add docstrings, to aid in reviewing code and also for the API reference.

We use `Numpy style docstrings <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
and `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ to document this library.
Sphinx, in turn, uses `reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_
as its markup language for adding code.

We use the `Sphinx Autosummary extension <http://www.sphinx-doc.org/en/stable/ext/autosummary.html>`_
to generate API references. In particular, you may want do look at the :code:`docs/generated`
directory to see how these files look and where to add new functions, classes or modules.
For example, if you add a new function to the :code:`sparse.COO` class, you would open up
:code:`docs/generated/sparse.COO.rst`, and add in the name of the function where appropriate.

To build the documentation, you can :code:`cd` into the :code:`docs` directory
and run

.. code-block:: bash

   sphinx-build -b html . _build/html

After this, you can find an HTML version of the documentation in :code:`docs/_build/html/index.html`.
