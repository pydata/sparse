#!/usr/bin/env python

from os.path import exists
from setuptools import setup


setup(name='sparse',
      version='0.3.1',
      description='Sparse n-dimensional arrays',
      url='http://github.com/pydata/sparse/',
      maintainer='Matthew Rocklin',
      maintainer_email='mrocklin@gmail.com',
      license='BSD',
      keywords='sparse,numpy,scipy,dask',
      packages=['sparse'],
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      install_requires=list(open('requirements.txt').read().strip().split('\n')),
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
      zip_safe=False)
