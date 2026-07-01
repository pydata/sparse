#!/usr/bin/env bash

for example in $(find ./examples/ -iname '*.ipynb'); do
  if [ ! "$example" =~ "finch" ]; then
    pytest -n 4 --nbmake --nbmake-timeout=600 "$example"
  else
    echo "Skipping finch example: $example"
  fi
done
