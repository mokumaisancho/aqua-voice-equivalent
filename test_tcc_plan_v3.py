import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / "TCC_EXECUTION_PLAN_v3.json").read_text(encoding="utf-8"))

assert plan["mode"] == "ONE_PASS_EXECUTABLE_STATE_MACHINE"
assert plan["global_policy"]["fail_closed"] is True
assert plan["global_policy"]["no_runtime_threshold_changes"] is True
assert plan["global_policy"]["no_fixture_specific_tuning"] is True
assert plan["global_policy"]["no_model_swap"] is True

states = {x["id"]: x for x in plan["states"]}
for sid in ["S00","S10","S20","S30","S40","S50","S60","S70","S80","S90","S100","S110"]:
    assert sid in states, sid

assert states["S110"].get("terminal") is True
assert plan["closure_rule"].startswith("CLOSED only when final disposition is PASS")

required = set(plan["evidence"]["required"])
for name in [
    "orchestration_result.json",
    "stage_events.jsonl",
    "runtime_snapshot.json",
    "fixture_manifest.json",
    "accuracy_metrics.json",
    "latency_metrics.json",
    "resource_metrics.json",
    "ablation_summary.json",
    "semantic_integrity.json",
    "validation_result.json",
    "current_work_state.json",
]:
    assert name in required, name

print("PASS")
