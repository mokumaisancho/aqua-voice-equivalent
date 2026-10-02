#!/usr/bin/env bash
set -euo pipefail
if [ -d .venv ]; then . .venv/bin/activate; fi
python test_fixture_catalog.py
python fetch_public_fixtures.py
python extract_jecs_fixture.py
python tcc_runner.py --mode preflight
python tcc_runner.py --mode execute
