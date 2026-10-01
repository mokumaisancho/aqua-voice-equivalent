#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-offline}"
PY="${PYTHON_BIN:-python3}"
"$PY" - <<'PY'
import sys
if sys.version_info < (3,9): raise SystemExit('Python >=3.9 required')
print('python',sys.version.split()[0])
PY
"$PY" -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
if [ "$MODE" = "streaming" ]; then
  pip install -U 'qwen-asr[vllm]'
else
  pip install -U qwen-asr
fi
python tcc_runner.py --mode preflight
printf '\nSetup complete. Copy fixtures/manifest.template.json to fixtures/manifest.json, add frozen WAVs + SHA256, then run ./run_tcc.sh\n'
