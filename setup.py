#!/usr/bin/env python

from setuptools import setup
from sparse._version import __version__

with open('requirements.txt') as f:
    reqs = list(f.read().strip().split('\n'))

with open('README.rst') as f:
    long_desc = f.read()

setup(
    name='sparse',
    version=__version__,
    description='Sparse n-dimensional arrays',
    url='http://github.com/pydata/sparse/',
    maintainer='Hameer Abbasi',
    maintainer_email='hameerabbasi@yahoo.com',
    license='BSD',
    keywords='sparse,numpy,scipy,dask',
    packages=['sparse'],
    long_description=long_desc,
    install_requires=reqs,
    extras_require={
        'tests': [
            'tox',
            'pytest',
            'pytest-cov',
            'pytest-flake8',
            'packaging',
        ],
        'docs': [
            'sphinx',
            'sphinxcontrib-napoleon',
            'sphinx_rtd_theme',
            'numpydoc',
        ],
    },
    zip_safe=False
)
