#!/usr/bin/env bash
set -euxo pipefail

SPARSE_BACKEND=MLIR pytest --pyargs sparse/mlir_backend --cov-report=xml:coverage_MLIR.xml -n auto -vvv
