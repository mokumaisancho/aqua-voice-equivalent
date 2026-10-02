from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E = ROOT / "evidence"
F = ROOT / "fixtures"
PLAN = json.loads((ROOT / "TCC_EXECUTION_PLAN_v4.json").read_text(encoding="utf-8"))


def load_json(path: Path):
    if not path.exists(): return None
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None


def load_jsonl(path: Path):
    if not path.exists(): return None
    out=[]
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip(): out.append(json.loads(line))
        return out
    except Exception: return None


def norm(s: str) -> str:
    s=unicodedata.normalize("NFKC",str(s)).lower()
    return " ".join(s.split())


def pct(xs,p):
    xs=sorted(float(x) for x in xs)
    if not xs: return None
    k=(len(xs)-1)*p/100.0; lo=int(math.floor(k)); hi=min(lo+1,len(xs)-1)
    return xs[lo] if lo==hi else xs[lo]+(xs[hi]-xs[lo])*(k-lo)


def ac(status, metric=None, value=None, threshold=None, reason=None, evidence=None):
    row={"status":status}
    if metric is not None: row["metric"]=metric
    if value is not None: row["value"]=value
    if threshold is not None: row["threshold"]=threshold
    if reason is not None: row["reason"]=reason
    if evidence is not None: row["evidence"]=evidence
    return row


def main() -> int:
    hard={}
    gates=PLAN["hard_ac"]

    # AC-ACC-EN: official OpenASR only.
    oa=load_json(E/"openasr_result.json")
    if not oa or oa.get("status")!="READY" or oa.get("macro_wer") is None:
        hard["AC-ACC-EN"]=ac("BLOCKED",reason="official pinned OpenASR result missing",evidence="evidence/openasr_result.json")
    else:
        v=float(oa["macro_wer"]); hard["AC-ACC-EN"]=ac("PASS" if v<=0.0623 else "FAIL","english_wer",v,0.0623,evidence="evidence/openasr_result.json")

    # AC-ACC-JA-AQUA: paired same fixtures.
    aq=load_json(E/"aqua_pair_manifest.json")
    if not aq or aq.get("status")!="READY" or aq.get("paired_delta_pp") is None:
        hard["AC-ACC-JA-AQUA"]=ac("BLOCKED",reason="paired Aqua Japanese evidence missing",evidence="evidence/aqua_pair_manifest.json")
    else:
        v=float(aq["paired_delta_pp"]); hard["AC-ACC-JA-AQUA"]=ac("PASS" if v<=0.5 else "FAIL","paired_japanese_cer_delta_pp",v,0.5,evidence="evidence/aqua_pair_manifest.json")

    # Technical declared-term recall from offline raw predictions.
    preds=load_jsonl(E/"offline_predictions.jsonl")
    ann=load_json(F/"technical_term_annotations.json")
    if not preds or not ann or ann.get("status")!="READY" or not ann.get("cases"):
        hard["AC-TECH"]=ac("BLOCKED",reason="technical annotations or offline predictions missing")
    else:
        by_id={x.get("id"):norm(x.get("hypothesis","")) for x in preds}
        total=hit=0
        for case in ann["cases"]:
            hyp=by_id.get(case["fixture_id"])
            if hyp is None: continue
            for term in case["key_terms"]:
                total+=1; hit+=int(norm(term) in hyp)
        if total==0: hard["AC-TECH"]=ac("BLOCKED",reason="no annotated technical prediction pairs")
        else:
            v=hit/total; hard["AC-TECH"]=ac("PASS" if v>=0.95 else "FAIL","technical_key_term_accuracy",v,0.95)

    # Dictionary/context paired evidence must provide explicit aggregate metrics.
    ctx=load_json(E/"context_dictionary_metrics.json")
    if not ctx or ctx.get("status")!="READY":
        for aid in ["AC-DICT-RECALL","AC-DICT-FALSE-BIAS","AC-CTX-HARM"]:
            hard[aid]=ac("BLOCKED",reason="context/dictionary paired metrics missing",evidence="evidence/context_dictionary_metrics.json")
    else:
        dr=ctx.get("dictionary_target_recall"); fb=ctx.get("false_bias_rate"); ch=ctx.get("context_false_substitution_delta_pp")
        hard["AC-DICT-RECALL"]=ac("BLOCKED" if dr is None else ("PASS" if float(dr)>=0.98 else "FAIL"),"dictionary_target_recall",dr,0.98)
        hard["AC-DICT-FALSE-BIAS"]=ac("BLOCKED" if fb is None else ("PASS" if float(fb)<=0.01 else "FAIL"),"false_bias_rate",fb,0.01)
        hard["AC-CTX-HARM"]=ac("BLOCKED" if ch is None else ("PASS" if float(ch)<=1.0 else "FAIL"),"context_false_substitution_delta_pp",ch,1.0)

    # Latency: >=100 measured warm real-time-paced runs mandatory.
    lat=load_jsonl(E/"streaming_latency_runs.jsonl")
    valid=[] if not lat else [x for x in lat if x.get("measured") is True and x.get("real_time_paced") is True and x.get("release_to_final_ms") is not None]
    if len(valid)<100:
        for aid in ["AC-LAT-P50","AC-LAT-P95"]:
            hard[aid]=ac("BLOCKED",reason=f"need >=100 measured real-time-paced runs; found {len(valid)}",evidence="evidence/streaming_latency_runs.jsonl")
    else:
        vals=[float(x["release_to_final_ms"]) for x in valid]; p50=pct(vals,50); p95=pct(vals,95)
        hard["AC-LAT-P50"]=ac("PASS" if p50<=450 else "FAIL","release_to_final_p50_ms",p50,450)
        hard["AC-LAT-P95"]=ac("PASS" if p95<=700 else "FAIL","release_to_final_p95_ms",p95,700)

    # Semantic integrity cannot default to zero: result file required.
    sem=load_json(E/"semantic_integrity_metrics.json")
    if not sem or sem.get("status")!="READY" or sem.get("critical_violations") is None:
        hard["AC-SEM"]=ac("BLOCKED",reason="semantic integrity metrics missing",evidence="evidence/semantic_integrity_metrics.json")
    else:
        v=int(sem["critical_violations"]); hard["AC-SEM"]=ac("PASS" if v==0 else "FAIL","semantic_critical_violations",v,0)

    # Coverage AC is about explicit evaluation state, not success of the other ACs.
    expected=set(PLAN["hard_ac"])-{"AC-EVIDENCE"}
    covered=set(hard)
    missing=sorted(expected-covered)
    invalid=sorted(k for k,v in hard.items() if v.get("status") not in {"PASS","FAIL","BLOCKED"})
    hard["AC-EVIDENCE"]=ac("PASS" if not missing and not invalid else "BLOCKED","hard_ac_coverage",len(expected)-len(missing),len(expected),reason=None if not missing and not invalid else f"missing={missing}, invalid={invalid}")

    E.mkdir(parents=True,exist_ok=True)
    (E/"hard_ac_coverage_matrix.json").write_text(json.dumps({"plan_id":PLAN["plan_id"],"hard_ac":hard},ensure_ascii=False,indent=2),encoding="utf-8")

    statuses={k:v["status"] for k,v in hard.items()}
    if statuses.get("AC-SEM")=="FAIL": status,branch="REJECT","REJECT_SEMANTIC"
    elif any(v=="FAIL" for v in statuses.values()): status,branch="REFRAME","REFRAME_HARD_AC_FAIL"
    elif any(v=="BLOCKED" for v in statuses.values()): status,branch="BLOCKED","BLOCKED_HARD_AC_NOT_RUN"
    elif all(v=="PASS" for v in statuses.values()) and set(statuses)==set(PLAN["hard_ac"]): status,branch="PASS","PASS"
    else: status,branch="BLOCKED","BLOCKED_HARD_AC_NOT_RUN"

    result={"status":status,"branch":branch,"hard_ac_status":statuses,"product_pass":status=="PASS"}
    (E/"validation_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))
    return 0 if status=="PASS" else 2


if __name__=="__main__": raise SystemExit(main())
