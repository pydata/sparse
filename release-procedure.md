*   Tag commit

        git tag -a x.x.x -m 'Version x.x.x'

*   Push to github

        git push mrocklin master --tags

*   Upload to PyPI

        git clean -xfd
        python setup.py sdist bdist_wheel --universal
        twine upload dist/*
