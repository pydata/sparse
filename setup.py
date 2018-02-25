#!/usr/bin/env python

from setuptools import setup
from sparse._version import __version__

with open('requirements.txt') as f:
    reqs = list(f.read().strip().split('\n'))

with open('README.rst') as f:
    long_desc = f.read()

extras_require = {
    'tests': [
        'pytest',
    ],
    'docs': [
        'sphinx',
        'sphinxcontrib-napoleon',
        'sphinx_rtd_theme',
    ],
    'tox': [
        'tox',
    ],
    'tests-linting': [
        'flake8'
    ],
    'tests-cov': [
        'pytest-cov'
    ]
}

all_requires = []
tests_requires = []
for k, v in extras_require.items():
    all_requires.extend(v)

    if k.startswith('tests'):
        tests_requires.extend(v)

extras_require['all'] = all_requires
extras_require['tests-all'] = tests_requires

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
    extras_require=extras_require,
    zip_safe=False
)
