import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import tcc_orchestrator as orch

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "TCC_EXECUTION_PLAN_v3.json").read_text(encoding="utf-8"))


def test_required_evidence_contract():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        names = PLAN["evidence"]["required"]
        for name in names:
            if name == "orchestration_result.json":
                continue
            (run_dir / name).write_text("{}\n", encoding="utf-8")
        reported, missing = orch.required_evidence(PLAN, run_dir)
        assert reported == names
        assert missing == []

        (run_dir / "stage_events.jsonl").unlink()
        _, missing = orch.required_evidence(PLAN, run_dir)
        assert missing == ["stage_events.jsonl"]


def test_jecs_missing_is_nonfatal_block_signal():
    with tempfile.TemporaryDirectory() as td:
        env = os.environ.copy()
        env["JECS_ZIP"] = str(Path(td) / "missing.zip")
        p = subprocess.run(
            [sys.executable, str(ROOT / "extract_jecs_fixture.py")],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
        row = json.loads(p.stdout.strip())
        assert row["status"] == "SKIP"
        assert row["reason"] == "JECS_ZIP missing"


if __name__ == "__main__":
    test_required_evidence_contract()
    test_jecs_missing_is_nonfatal_block_signal()
    print("PASS")
