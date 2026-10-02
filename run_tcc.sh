#!/usr/bin/env bash
set -euo pipefail
python3 test_tcc_plan_v3.py
python3 tcc_orchestrator.py
