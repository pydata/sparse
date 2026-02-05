#!/usr/bin/env bash
set -euxo pipefail

for example in $(find ./examples/ -iname '*.py'); do
  python $example
done
