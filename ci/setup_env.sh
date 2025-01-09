#!/usr/bin/env bash
set -euxo pipefail

if [ ! -d ".venv" ]; then
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .[all]
  source ci/clone_array_api_tests.sh
  pip install -r ../array-api-tests/requirements.txt
  pip uninstall -y matrepr
fi
