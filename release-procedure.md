*   Update changelog in docs/changelog.rst

*   Bump version in sparse/version.py and commit.

*   Tag commit

        git tag -a x.x.x -m 'Version x.x.x'

*   Push to github

        git push pydata master --tags

*   Upload to PyPI

        git clean -xfd  # remove all files in directory not in repository
        python setup.py sdist bdist_wheel --universal  # make packages
        twine upload dist/*  # upload packages
