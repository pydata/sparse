#!/usr/bin/env python

from setuptools import setup, find_packages
import versioneer

with open('requirements.txt') as f:
    reqs = list(f.read().strip().split('\n'))

with open('README.rst') as f:
    long_desc = f.read()

extras_require = {
    'tests': [
        'pytest>=3.5',
        'pytest-flake8',
        'pytest-cov'
    ],
    'docs': [
        'sphinx',
        'sphinx_rtd_theme',
    ],
    'tox': [
        'tox',
    ],
    'bench': [
        'asv'
    ],
}

all_requires = []

for v in extras_require.values():
    all_requires.extend(v)

extras_require['all'] = all_requires

setup(
    name='sparse',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Sparse n-dimensional arrays',
    url='https://github.com/pydata/sparse/',
    maintainer='Hameer Abbasi',
    maintainer_email='hameerabbasi@yahoo.com',
    license='BSD 3-Clause License (Revised)',
    keywords='sparse,numpy,scipy,dask',
    packages=find_packages(
        include=['sparse', 'sparse.*'],
    ),
    long_description=long_desc,
    install_requires=reqs,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
    ],
    project_urls={
        'Documentation': 'https://sparse.pydata.org/',
        'Source': 'https://github.com/pydata/sparse/',
        'Tracker': 'https://github.com/pydata/sparse/issues',
    },
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
)
