Contributing
============

General Guidelines
------------------

sparse is a community-driven project on GitHub. You can find our
`repository on GitHub <https://github.com/pydata/sparse>`_. Feel
free to open issues for new features or bugs, or open a pull request
to fix a bug or add a new feature.

If you haven't contributed to open-source before, we recommend you read
`this excellent guide by GitHub on how to contribute to open source
<https://opensource.guide/how-to-contribute/>`_. The guide is long,
so you can gloss over things you're familiar with.

If you're not already familiar with it, we follow the `fork and pull model
<https://help.github.com/articles/about-collaborative-development-models/>`_
on GitHub.

Filing Issues
-------------

If you find a bug or would like a new feature, you might want to `consider
filing a new issue on GitHub <https://github.com/pydata/sparse/issues>`_. Before
you open a new issue, please make sure of the following:

* This should go without saying, but make sure what you are requesting is within
  the scope of this project.
* The bug/feature is still present/missing on the ``master`` branch on GitHub.
* A similar issue or pull request isn't already open. If one already is, it's better
  to contribute to the discussion there.

Contributing Code
-----------------

This project has a number of requirements for all code contributed.

* We use ``flake8`` to automatically lint the code and maintain code style.
* We use Numpy-style docstrings.
* It's ideal if user-facing API changes or new features have documentation added.
* 100% code coverage is recommended for all new code in any submitted PR. Doctests
  count toward coverage.
* Performance optimizations should have benchmarks added in ``benchmarks``.

Setting up Your Development Environment
---------------------------------------

The following bash script is all you need to set up your development environment,
after forking and cloning the repository:

.. code-block:: bash

   pip install -e .[all]


Running/Adding Unit Tests
-------------------------

It is best if all new functionality and/or bug fixes have unit tests added
with each use-case.

We use `pytest <https://docs.pytest.org/en/latest/>`_ as our unit testing framework,
with the ``pytest-cov`` extension to check code coverage and ``pytest-flake8`` to
check code style. You don't need to configure these extensions yourself. Once you've
configured your environment, you can just ``cd`` to the root of your repository and run

.. code-block:: bash

   pytest --pyargs sparse

This automatically checks code style and functionality, and prints code coverage,
even though it doesn't fail on low coverage.

Unit tests are automatically run on Travis CI for pull requests.

Coverage
--------

The ``pytest`` script automatically reports coverage, both on the terminal for
missing line numbers, and in annotated HTML form in ``htmlcov/index.html``.

Coverage is automatically checked on CodeCov for pull requests.

Adding/Building the Documentation
---------------------------------

If a feature is stable and relatively finalized, it is time to add it to the
documentation. If you are adding any private/public functions, it is best to
add docstrings, to aid in reviewing code and also for the API reference.

We use `Numpy style docstrings <https://numpydoc.readthedocs.io/en/latest/format.html>`_
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

   sphinx-build -W -b html . _build/html

After this, you can find an HTML version of the documentation in :code:`docs/_build/html/index.html`.

Documentation for pull requests is automatically built on CircleCI and can be found in the build
artifacts.

Adding and Running Benchmarks
-----------------------------

We use `Airspeed Velocity <https://asv.readthedocs.io/en/latest/>`_ to run benchmarks. We have it set
up to use ``conda``, but you can edit the configuration locally if you so wish.
