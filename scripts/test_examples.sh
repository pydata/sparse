#!/usr/bin/env bash
set -euxo pipefail

for example in $(find ./examples/ -iname '*.py'); do
  if [[ ! "$example" =~ "finch" ]]; then
    python "$example"
  else
    echo "Skipping finch example: $example"
  fi
done
