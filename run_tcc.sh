#!/usr/bin/env bash
set -euo pipefail
python3 test_tcc_plan_v4.py
python3 test_tcc_v4_metrics.py
python3 tcc_orchestrator_v4.py
