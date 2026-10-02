import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / "TCC_EXECUTION_PLAN_v4.json").read_text(encoding="utf-8"))

assert plan["mode"] == "ONE_PASS_DEPENDENCY_AWARE_FAIL_CLOSED"
assert plan["global_policy"]["fail_closed"] is True
assert plan["global_policy"]["not_run_is_pass"] is False
assert plan["global_policy"]["missing_hard_ac_is_blocked"] is True
assert plan["global_policy"]["no_runtime_threshold_changes"] is True
assert plan["global_policy"]["no_fixture_specific_tuning"] is True
assert plan["global_policy"]["no_model_swap"] is True

expected_deps = [f"D{i}" for i in [0,10,20,30,40,50,60,70,80,90,100]]
actual = set(plan["dependency_graph"])
for prefix in expected_deps:
    assert any(x.startswith(prefix + "_") for x in actual), prefix

hard = plan["hard_ac"]
for ac in [
    "AC-ACC-EN","AC-ACC-JA-AQUA","AC-TECH","AC-DICT-RECALL","AC-DICT-FALSE-BIAS",
    "AC-CTX-HARM","AC-LAT-P50","AC-LAT-P95","AC-SEM","AC-EVIDENCE"
]:
    assert ac in hard, ac
    assert hard[ac]["requires"], ac

issues = {x["audit"] for x in plan["issue_registry"]}
for audit in ["A1","A2","A3","A4","A5","A6","A7","A8","A9"]:
    assert audit in issues, audit

states = {x["id"]: x for x in plan["state_machine"]}
for sid in ["S00","S10","S20","S30","S40","S50","S60","S70","S80","S90","S100","S110","S120"]:
    assert sid in states, sid
assert states["S120"].get("terminal") is True
assert "every hard AC status is PASS" in plan["closure_rule"]
print("PASS")
