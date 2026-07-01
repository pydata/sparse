#!/usr/bin/env bash
set -euxo pipefail

for example in $(find ./examples/ -iname '*.py'); do
  if grep -iq finch "$example"; then
    echo "Skipping finch example: $example"
    continue
  fi
  python "$example"
done
