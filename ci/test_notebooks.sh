#!/usr/bin/env bash
set -euxo pipefail

CI_MODE=1 pytest -n 4 --nbmake --nbmake-timeout=600 ./examples/*.ipynb
