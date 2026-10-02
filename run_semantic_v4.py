from __future__ import annotations

import json
from pathlib import Path

from finalizer_v4 import VERSION, allowed_removed_token, finalize, removed_tokens
from qwen_model_pin import MODEL_ID, MODEL_REVISION

IN = Path("evidence/offline_predictions.jsonl")
OUT = Path("evidence/semantic_integrity_metrics.json")
DETAIL = Path("evidence/semantic_integrity_results.jsonl")


def main():
    if not IN.exists():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"status": "BLOCKED", "reason": "offline predictions missing"}, indent=2), encoding="utf-8")
        return 2

    rows = []
    critical = 0
    seen_revisions = set()
    for line in IN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        rev = x.get("model_revision")
        if rev:
            seen_revisions.add(rev)
        raw = str(x.get("hypothesis", "")).strip()
        final = finalize(raw)
        removed = removed_tokens(raw, final)
        invalid_removed = [t for t in removed if not allowed_removed_token(t)]
        violation = bool(invalid_removed)
        critical += int(violation)
        rows.append({
            "id": x.get("id"),
            "raw": raw,
            "final": final,
            "removed_tokens": removed,
            "invalid_removed_tokens": invalid_removed,
            "critical_violation": violation,
            "policy": VERSION,
            "model_revision": rev,
        })

    revision_ok = seen_revisions == {MODEL_REVISION}
    DETAIL.parent.mkdir(parents=True, exist_ok=True)
    DETAIL.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    obj = {
        "status": "READY" if rows and revision_ok else "BLOCKED",
        "critical_violations": critical if revision_ok else None,
        "evaluated": len(rows),
        "finalizer": VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "observed_revisions": sorted(seen_revisions),
        "revision_ok": revision_ok,
        "rule": "Only standalone allowlisted English filler tokens may be removed; any other token deletion is critical. Japanese filler deletion disabled.",
    }
    OUT.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(obj, ensure_ascii=False))
    return 0 if obj["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
