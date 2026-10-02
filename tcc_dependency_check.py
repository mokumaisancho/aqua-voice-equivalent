from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "TCC_EXECUTION_PLAN_v4.json"
OUT = ROOT / "evidence" / "dependency_snapshot.json"

ARTIFACTS = {
    "D00_audit_baseline": ["VALIDATION_AUDIT_2026-10-03.json"],
    "D10_model_identity": ["evidence/model_identity.json"],
    "D20_scorer_contract": ["evidence/scorer_contract.json"],
    "D30_fixture_contract": ["fixtures/manifest.json"],
    "D40_technical_term_contract": ["fixtures/technical_term_annotations.json"],
    "D50_aqua_pair_contract": ["evidence/aqua_pair_manifest.json"],
    "D60_context_dictionary_contract": ["fixtures/context_dictionary_manifest.json"],
    "D70_semantic_contract": ["fixtures/semantic_oracle_manifest.json"],
    "D80_latency_contract": ["evidence/latency_contract.json"],
    "D90_evaluator_completeness": ["evidence/hard_ac_coverage_schema.json"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as e:
        return None, str(e)


def artifact_state(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        return {"path": rel, "state": "MISSING"}
    row = {"path": rel, "state": "PRESENT", "sha256": sha256(p)}
    if p.suffix == ".json":
        data, err = read_json(p)
        if err:
            row.update(state="INVALID_JSON", error=err)
        elif isinstance(data, dict):
            declared = data.get("status") or data.get("state")
            if declared:
                row["declared_state"] = declared
            if declared in {"BLOCKED", "OPEN", "PENDING", "NOT_RUN", "INVALID"}:
                row["state"] = "NOT_READY"
    return row


def main() -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    graph = plan["dependency_graph"]
    nodes = {}
    unresolved = []

    for dep_id in graph:
        required = ARTIFACTS.get(dep_id, [])
        artifacts = [artifact_state(x) for x in required]
        deps = graph[dep_id].get("depends_on", [])
        upstream_blocked = [x for x in deps if nodes.get(x, {}).get("ready") is False]
        artifact_blocked = [x for x in artifacts if x["state"] != "PRESENT"]
        ready = not upstream_blocked and not artifact_blocked
        if dep_id == "D100_execution_ready":
            ready = all(nodes.get(x, {}).get("ready") for x in deps)
        nodes[dep_id] = {
            "ready": bool(ready),
            "depends_on": deps,
            "upstream_blocked": upstream_blocked,
            "artifacts": artifacts,
        }
        if not ready:
            unresolved.append(dep_id)

    snapshot = {
        "plan_id": plan["plan_id"],
        "status": "READY" if nodes["D100_execution_ready"]["ready"] else "BLOCKED",
        "execution_ready": nodes["D100_execution_ready"]["ready"],
        "unresolved_dependencies": unresolved,
        "nodes": nodes,
        "rule": "All dependency blockers are reported in one pass. Model execution is prohibited until D100 is ready.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": snapshot["status"],
        "execution_ready": snapshot["execution_ready"],
        "unresolved_dependencies": unresolved,
        "snapshot": str(OUT.relative_to(ROOT)),
    }, ensure_ascii=False))
    return 0 if snapshot["execution_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
