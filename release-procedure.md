* Update changelog in docs/changelog.rst and commit.

* Tag commit
  ```bash
  git tag -a x.x.x -m 'Version x.x.x'
  ```

* Push to github
  ```bash
  git push pydata main --tags
  ```

* Upload to PyPI
  ```bash
  git clean -xfd  # remove all files in directory not in repository
  python -m build --wheel --sdist # make packages
  twine upload dist/*  # upload packages
  ```

* Enable the newly-pushed tag for documentation: https://readthedocs.org/projects/sparse-nd/versions/
* Wait for conda-forge to realise that the build is too old and make a PR.
  * Edit and merge that PR.
* Announce the release on:
  * numpy-discussion@python.org
  * python-announce-list@python.org
