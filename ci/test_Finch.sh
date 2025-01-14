#!/usr/bin/env bash
set -euxo pipefail

python -c 'import finch'
PYTHONFAULTHANDLER="${HOME}/faulthandler.log" SPARSE_BACKEND=Finch pytest --pyargs sparse/tests --cov-report=xml:coverage_Finch.xml -n auto -vvv
