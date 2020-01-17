#!/usr/bin/env python

from setuptools import setup, find_packages
import versioneer
from pathlib import Path


def open_reqs_file(file, reqs_path=Path(".")):
    with (reqs_path / file).open() as f:
        reqs = list(f.read().strip().split("\n"))

    i = 0
    while i < len(reqs):
        if reqs[i].startswith("-r"):
            reqs[i : i + 1] = open_reqs_file(reqs[i][2:].strip(), reqs_path=reqs_path)
        else:
            i += 1

    return reqs


extras_require = {}
reqs = []


def parse_requires():
    reqs_path = Path("./requirements")
    reqs.extend(open_reqs_file("requirements.txt"))
    for f in reqs_path.iterdir():
        extras_require[f.stem] = open_reqs_file(f.parts[-1], reqs_path=reqs_path)


parse_requires()

with open("README.rst") as f:
    long_desc = f.read()

print(repr(reqs))
print(repr(reqs))

setup(
    name="sparse",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Sparse n-dimensional arrays",
    url="https://github.com/pydata/sparse/",
    maintainer="Hameer Abbasi",
    maintainer_email="hameerabbasi@yahoo.com",
    license="BSD 3-Clause License (Revised)",
    keywords="sparse,numpy,scipy,dask",
    packages=find_packages(include=["sparse", "sparse.*"]),
    long_description=long_desc,
    install_requires=reqs,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    project_urls={
        "Documentation": "https://sparse.pydata.org/",
        "Source": "https://github.com/pydata/sparse/",
        "Tracker": "https://github.com/pydata/sparse/issues",
    },
    entry_points={
        "numba_extensions": [
            "init = sparse._numba_extension:_init_extension",
        ],
    },
    python_requires=">=3.5, <4",
)
