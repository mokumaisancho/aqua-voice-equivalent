from __future__ import annotations

import json
from pathlib import Path

IN=Path("evidence/offline_predictions.jsonl")
OUT=Path("evidence/semantic_integrity_metrics.json")
DETAIL=Path("evidence/semantic_integrity_results.jsonl")


def main():
    if not IN.exists():
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps({"status":"BLOCKED","reason":"offline predictions missing"},indent=2),encoding="utf-8")
        return 2
    rows=[]
    for line in IN.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        x=json.loads(line); raw=str(x.get("hypothesis","")).strip()
        final=raw  # v4 baseline: no downstream semantic-changing normalizer enabled.
        rows.append({"id":x.get("id"),"raw":raw,"final":final,"critical_violation":False,"policy":"identity_finalizer"})
    DETAIL.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    obj={"status":"READY" if rows else "BLOCKED","critical_violations":sum(int(x["critical_violation"]) for x in rows),"evaluated":len(rows),"finalizer":"identity","note":"Any future non-identity finalizer invalidates this evidence and requires rerun."}
    OUT.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(obj,ensure_ascii=False)); return 0 if obj["status"]=="READY" else 2

if __name__=="__main__": raise SystemExit(main())
