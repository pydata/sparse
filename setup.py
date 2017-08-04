#!/usr/bin/env python

from os.path import exists
from setuptools import setup


setup(name='sparse',
      version='0.1.1',
      description='Sparse',
      url='http://github.com/mrocklin/sparse/',
      maintainer='Matthew Rocklin',
      maintainer_email='mrocklin@gmail.com',
      license='BSD',
      keywords='sparse,numpy,scipy,dask',
      packages=['sparse'],
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      install_requires=list(open('requirements.txt').read().strip().split('\n')),
      zip_safe=False)
