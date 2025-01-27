#!/usr/bin/env bash
set -euxo pipefail

source ci/clone_array_api_tests.sh

if [ "${SPARSE_BACKEND}" = "Finch" ]; then
    python -c 'import finch'
fi
ARRAY_API_TESTS_MODULE="sparse" pytest "$ARRAY_API_TESTS_DIR/array_api_tests/" -v -c "$ARRAY_API_TESTS_DIR/pytest.ini" --ci --max-examples=2 --derandomize --disable-deadline --disable-warnings -o xfail_strict=True -n auto --xfails-file ../sparse/ci/${SPARSE_BACKEND}-array-api-xfails.txt --skips-file ../sparse/ci/${SPARSE_BACKEND}-array-api-skips.txt
