## Contributing

## General Guidelines

sparse is a community-driven project on GitHub. You can find our
[repository on GitHub](https://github.com/pydata/sparse). Feel
free to open issues for new features or bugs, or open a pull request
to fix a bug or add a new feature.

If you haven't contributed to open-source before, we recommend you read
[this excellent guide by GitHub on how to contribute to open source](https://opensource.guide/how-to-contribute). The guide is long,
so you can gloss over things you're familiar with.

If you're not already familiar with it, we follow the [fork and pull model](https://help.github.com/articles/about-collaborative-development-models)
on GitHub.

## Filing Issues

If you find a bug or would like a new feature, you might want to *consider
filing a new issue* on [GitHub](https://github.com/pydata/sparse/issues). Before
you open a new issue, please make sure of the following:

* This should go without saying, but make sure what you are requesting is within
  the scope of this project.
* The bug/feature is still present/missing on the `main` branch on GitHub.
* A similar issue or pull request isn't already open. If one already is, it's better
  to contribute to the discussion there.

## Contributing Code

This project has a number of requirements for all code contributed.

* We use `pre-commit` to automatically lint the code and maintain code style.
* We use Numpy-style docstrings.
* It's ideal if user-facing API changes or new features have documentation added.
* 100% code coverage is recommended for all new code in any submitted PR. Doctests
  count toward coverage.
* Performance optimizations should have benchmarks added in `benchmarks`.

## Setting up Your Development Environment

The following bash script is all you need to set up your development environment,
after forking and cloning the repository:

```bash

pip install -e .[all]
```

## Pull requests

Please adhere to the following guidelines:

1. Start your pull request title with a [conventional commit](https://www.conventionalcommits.org/) tag. This helps us add your contribution to the right section of the changelog. We use "Type" from the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#type).<br>
    TLDR:<br>
    The PR title should start with any of these abbreviations: `build`, `chore`, `ci`, `depr`,
    `docs`, `feat`, `fix`, `perf`, `refactor`, `release`, `test`. Add a `!`at the end, if it is a breaking change. For example `refactor!`.
2. This text will end up in the changelog.
3. Please follow the instructions in the pull request form and submit.

## Running/Adding Unit Tests

It is best if all new functionality and/or bug fixes have unit tests added
with each use-case.

We use [pytest](https://docs.pytest.org/en/latest) as our unit testing framework,
with the `pytest-cov` extension to check code coverage and `pytest-flake8` to
check code style. You don't need to configure these extensions yourself. Once you've
configured your environment, you can just `cd` to the root of your repository and run

```bash
pytest --pyargs sparse
```

This automatically checks code style and functionality, and prints code coverage,
even though it doesn't fail on low coverage.

Unit tests are automatically run on Travis CI for pull requests.

### Advanced

To run the complete set of unit tests run in CI for your platform, run the following
in the repository root:

```bash
ci/setup_env.sh
ACTIVATE_VENV=1 ci/test_all.sh
```

## Coverage

The `pytest` script automatically reports coverage, both on the terminal for
missing line numbers, and in annotated HTML form in `htmlcov/index.html`.

Coverage is automatically checked on CodeCov for pull requests.

## Adding/Building the Documentation

If a feature is stable and relatively finalized, it is time to add it to the
documentation. If you are adding any private/public functions, it is best to
add docstrings, to aid in reviewing code and also for the API reference.

We use [Numpy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material) to document this library.
MkDocs, in turn, uses [Markdown](https://www.markdownguide.org)
as its markup language for adding code.

We use [mkdoctrings](https://mkdocstrings.github.io/recipes) with the
[mkdocs-gen-files plugin](https://oprypin.github.io/mkdocs-gen-files)
to generate API references.

To build the documentation, you can run

```bash

mkdocs build
mkdocs serve
```

After this, you can see a version of the documentation on your local server.

Documentation for each pull requests is automatically built on `Read the Docs`.
It is rebuilt with every new commit to your PR. There will be a link to preview it
from your PR checks area on `GitHub` when ready.


## Adding and Running Benchmarks

We use [`CodSpeed`](https://docs.codspeed.io/) to run benchmarks. They are run in the CI environment
when a pull request is opened. Then the results of the run are sent to `CodSpeed` servers to be analyzed.
When the analysis is done, a report is generated and posted automatically as a comment to the PR.
The report includes a link to `CodSpeed`cloud where you can see the all the results.

If you add benchmarks, they should be written as regular tests to be used with pytest, and use the fixture `benchmark`. Please see the `CodSpeed`documentation for more details.
