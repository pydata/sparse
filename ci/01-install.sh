#!/usr/bin/env bash

if [[ $NUMPY_VERSION ]]; then
    pip install --user 'numpy==$NUMPY_VERSION';
fi

pip install -e .[tests]
pip install --user codecov
