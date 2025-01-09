#!/usr/bin/env bash
set -euxo pipefail

if [ ! -d "../array-api-tests" ]; then
  git clone https://github.com/data-apis/array-api-tests.git "../array-api-tests"
fi
git --git-dir="../array-api-tests/.git" fetch
git --git-dir="../array-api-tests/.git" checkout $(cat "ci/array-api-tests-rev.txt")
