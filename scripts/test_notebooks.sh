#!/usr/bin/env bash

for example in $(find ./examples/ -iname '*.ipynb'); do
  if grep -iq finch "$example"; then
    echo "Skipping finch example: $example"
    continue
  fi
  pytest -n 4 --nbmake --nbmake-timeout=600 "$example"
done
