#!/usr/bin/env bash
set -euxo pipefail

source ci/test_Numba.sh
source ci/test_Finch.sh
source ci/test_MLIR.sh
