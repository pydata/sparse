* Tag commit
  ```bash
  git tag -a x.x.x -m 'Version x.x.x'
  ```

* Push to github
  ```bash
  git push pydata main --tags
  ```
  When you open the PR on GitHub, make sure the title of the PR starts with "release".

* Upload to PyPI
  ```bash
  git clean -xfd  # remove all files in directory not in repository
  python -m build --wheel --sdist # make packages
  twine upload dist/*  # upload packages
  ```

* Update the release drafter:
  Go to https://github.com/pydata/sparse
  Under the “Release" section there are two links: One is the latest release (it has a tag).
  The second one is +<number of releases>. Click on the second one so you can see the release drafter.
  Edit the draft by clicking the "pencil" figure.
  Make sure you have the correct tags. If they are not, you can create one.
  If the markdown page looks correct, click on “Publish release”.
<br>
* Enable the newly-pushed tag for documentation: https://readthedocs.org/projects/sparse-nd/versions/
* Wait for conda-forge to realise that the build is too old and make a PR.
  * Edit and merge that PR.
* Announce the release on:
  * numpy-discussion@python.org
  * python-announce-list@python.org
