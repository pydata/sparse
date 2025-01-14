#!/usr/bin/env bash
set -euxo pipefail

ACTIVATE_VENV="${ACTIVATE_VENV:-0}"

if [ $ACTIVATE_VENV = "1" ]; then
    source .venv/bin/activate
fi

source ci/test_backends.sh
source ci/test_examples.sh
source ci/test_notebooks.sh
SPARSE_BACKEND="Numba" source ci/test_array_api.sh
SPARSE_BACKEND="Finch" PYTHONFAULTHANDLER="${HOME}/faulthandler.log" source ci/test_array_api.sh
