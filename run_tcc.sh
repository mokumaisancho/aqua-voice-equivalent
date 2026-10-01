#!/usr/bin/env bash
set -euo pipefail
if [ -d .venv ]; then . .venv/bin/activate; fi
python tcc_runner.py --mode preflight
python tcc_runner.py --mode execute
