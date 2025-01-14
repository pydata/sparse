#!/usr/bin/env bash
set -euxo pipefail

if [ $(python -c 'import numpy as np; print(np.lib.NumpyVersion(np.__version__) >= "2.0.0a1")') = 'True' ]; then
    pytest --pyargs sparse --doctest-modules --cov-report=xml:coverage_Numba.xml -n auto -vvv
else
    pytest --pyargs sparse --cov-report=xml:coverage_Numba.xml -n auto -vvv
fi
