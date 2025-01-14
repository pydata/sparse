#!/usr/bin/env bash
set -euxo pipefail

ARRAY_API_TESTS_DIR="${ARRAY_API_TESTS_DIR:-"../array-api-tests"}"
if [ ! -d "$ARRAY_API_TESTS_DIR" ]; then
  git clone --recursive https://github.com/data-apis/array-api-tests.git "$ARRAY_API_TESTS_DIR"
fi

git --git-dir="$ARRAY_API_TESTS_DIR/.git" --work-tree "$ARRAY_API_TESTS_DIR" clean -xddf
git --git-dir="$ARRAY_API_TESTS_DIR/.git" --work-tree "$ARRAY_API_TESTS_DIR" fetch
git --git-dir="$ARRAY_API_TESTS_DIR/.git" --work-tree "$ARRAY_API_TESTS_DIR" reset --hard $(cat "ci/array-api-tests-rev.txt")
