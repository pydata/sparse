#!/usr/bin/env bash

rm -rf ~/.cache/pip/


if [[ $NUMPY_VERSION ]]; then
    pip install numpy$NUMPY_VERSION;
fi

pip install -e .[tests]
pip install codecov
