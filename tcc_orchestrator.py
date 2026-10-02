from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "TCC_EXECUTION_PLAN_v3.json"
RUNS = ROOT / "evidence" / "runs"


def ts() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def emit(events: Path, stage: str, status: str, **extra) -> None:
    row = {"ts": ts(), "stage": stage, "status": status, **extra}
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(cmd, *, events: Path, stage: str, retries: int = 0, env=None):
    last = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env=env)
        last = {
            "returncode": p.returncode,
            "stdout": p.stdout[-12000:],
            "stderr": p.stderr[-12000:],
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "attempt": attempt + 1,
            "command": cmd,
        }
        emit(events, stage, "PASS" if p.returncode == 0 else "FAIL", **last)
        if p.returncode == 0:
            return last
    return last


def py_has_pkg(py: str, name: str) -> bool:
    p = subprocess.run(
        [py, "-c", f"import importlib.metadata as m; m.version({name!r})"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return p.returncode == 0


def linux_cuda() -> bool:
    return platform.system() == "Linux" and shutil.which("nvidia-smi") is not None


def ensure_venv(events: Path):
    venv = ROOT / ".venv"
    if not venv.exists():
        r = run([sys.executable, "-m", "venv", str(venv)], events=events, stage="dependency_venv")
        if r["returncode"] != 0:
            return None
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return py if py.exists() else None


def bootstrap_dependencies(events: Path):
    py_path = ensure_venv(events)
    if py_path is None:
        return {"offline": False, "streaming": False, "python": sys.executable}

    py = str(py_path)
    offline_ok = py_has_pkg(py, "qwen-asr")
    if not offline_ok:
        r = run([py, "-m", "pip", "install", "-U", "qwen-asr"], events=events, stage="bootstrap_offline")
        offline_ok = r["returncode"] == 0 and py_has_pkg(py, "qwen-asr")

    if linux_cuda():
        streaming_ok = py_has_pkg(py, "vllm")
        if not streaming_ok:
            r = run([py, "-m", "pip", "install", "-U", "qwen-asr[vllm]"], events=events, stage="bootstrap_streaming")
            streaming_ok = r["returncode"] == 0 and py_has_pkg(py, "vllm")
    else:
        streaming_ok = False
        emit(events, "bootstrap_streaming", "BLOCKED", reason="unsupported_non_linux_or_no_cuda")

    return {"offline": offline_ok, "streaming": streaming_ok, "python": py}


def required_evidence(run_dir: Path):
    names = [
        "runtime_snapshot.json",
        "fixture_manifest.json",
        "accuracy_metrics.json",
        "latency_metrics.json",
        "resource_metrics.json",
        "ablation_summary.json",
        "semantic_integrity.json",
        "validation_result.json",
        "current_work_state.json",
    ]
    return names, [n for n in names if not (run_dir / n).exists()]


def finish(result_path: Path, result: dict, events: Path, code: int) -> int:
    result["closed"] = result.get("status") == "PASS"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    emit(events, "final_disposition", result.get("status", "BLOCKED"), branch=result.get("branch"), closed=result["closed"])
    print(json.dumps({"status": result.get("status"), "branch": result.get("branch"), "run_dir": result.get("run_dir")}, ensure_ascii=False))
    return code


def main() -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    run_id = ts()
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events = run_dir / "stage_events.jsonl"
    result_path = run_dir / "orchestration_result.json"

    result = {
        "run_id": run_id,
        "plan_id": plan.get("plan_id"),
        "status": "RUNNING",
        "branch": None,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "stages": {},
    }

    mandatory = ["states", "branch_actions", "acceptance_gates", "evidence"]
    missing = [k for k in mandatory if k not in plan]
    if missing:
        result.update(status="BLOCKED", branch="BLOCKED_PLAN", detail={"missing": missing})
        return finish(result_path, result, events, 2)
    emit(events, "plan_validation", "PASS")

    r = run([sys.executable, "test_fixture_catalog.py"], events=events, stage="fixture_contract")
    result["stages"]["fixture_contract"] = r
    if r["returncode"] != 0:
        result.update(status="BLOCKED", branch="BLOCKED_FIXTURE_CONTRACT")
        return finish(result_path, result, events, 2)

    r = run([sys.executable, "fetch_public_fixtures.py"], events=events, stage="public_fixture_fetch", retries=2)
    result["stages"]["public_fixture_fetch"] = r
    if r["returncode"] != 0:
        result.update(status="BLOCKED", branch="BLOCKED_FIXTURE")
        return finish(result_path, result, events, 2)

    r = run([sys.executable, "extract_jecs_fixture.py"], events=events, stage="jecs_extract")
    result["stages"]["jecs_extract"] = r
    code_switch_blocked = r["returncode"] != 0

    deps = bootstrap_dependencies(events)
    result["dependencies"] = deps
    if not deps["offline"]:
        result.update(status="BLOCKED", branch="BLOCKED_PACKAGE")
        return finish(result_path, result, events, 2)

    py = deps["python"]
    evidence_arg = str(run_dir.relative_to(ROOT))

    r = run([py, "tcc_runner.py", "--plan", PLAN.name, "--mode", "preflight", "--evidence-dir", evidence_arg], events=events, stage="preflight")
    result["stages"]["preflight"] = r

    r = run([py, "tcc_runner.py", "--plan", PLAN.name, "--mode", "execute", "--evidence-dir", evidence_arg], events=events, stage="qwen_suite")
    result["stages"]["qwen_suite"] = r

    names, missing = required_evidence(run_dir)
    result["required_evidence"] = names
    if missing:
        result.update(status="BLOCKED", branch="BLOCKED_EVIDENCE", missing_evidence=missing)
        return finish(result_path, result, events, 2)

    validation = json.loads((run_dir / "validation_result.json").read_text(encoding="utf-8"))
    status = validation.get("status", "BLOCKED")
    branch = validation.get("branch", "UNKNOWN")
    if code_switch_blocked and status == "PASS":
        status = "BLOCKED"
        branch = "PARTIAL_CODE_SWITCH_BLOCKED"
    if not deps["streaming"] and status == "PASS":
        status = "BLOCKED"
        branch = "PARTIAL_STREAMING_BLOCKED"
    result.update(status=status, branch=branch, validation=validation)
    return finish(result_path, result, events, 0 if status == "PASS" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
