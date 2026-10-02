from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/"evidence"/"runs"


def stamp(): return time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())

def emit(path,stage,status,**extra):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps({"ts":stamp(),"stage":stage,"status":status,**extra},ensure_ascii=False)+"\n")

def run(cmd,events,stage,retries=0,allow_fail=False):
    last=None
    for i in range(retries+1):
        t=time.perf_counter(); p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        last={"returncode":p.returncode,"stdout":p.stdout[-12000:],"stderr":p.stderr[-12000:],"elapsed_s":round(time.perf_counter()-t,3),"attempt":i+1,"command":cmd}
        emit(events,stage,"PASS" if p.returncode==0 else ("BLOCKED" if allow_fail else "FAIL"),**last)
        if p.returncode==0: break
    return last

def py_has(py,name):
    return subprocess.run([str(py),"-c",f"import importlib.metadata as m;m.version({name!r})"],cwd=ROOT).returncode==0

def copy_if_exists(src:Path,dst:Path):
    if src.exists(): dst.write_bytes(src.read_bytes())

def main():
    rid=stamp(); rd=RUNS/rid; rd.mkdir(parents=True,exist_ok=True); events=rd/"stage_events.jsonl"; result={"run_id":rid,"plan_id":"TCC-QWEN-ASR-004","status":"RUNNING","stages":{}}

    for script,stage in [("test_tcc_plan_v4.py","plan_test"),("prepare_tcc_v4_contracts.py","prepare_contracts"),("test_fixture_catalog.py","fixture_contract")]:
        r=run([sys.executable,script],events,stage); result["stages"][stage]=r
        if r["returncode"]!=0:
            result.update(status="BLOCKED",branch="BLOCKED_PLAN_OR_CONTRACT"); break
    else:
        r=run([sys.executable,"fetch_public_fixtures.py"],events,"fixture_fetch",retries=2); result["stages"]["fixture_fetch"]=r
        if r["returncode"]!=0:
            result.update(status="BLOCKED",branch="BLOCKED_FIXTURE")
        else:
            # Install offline runtime first. Streaming install is attempted but nonfatal.
            r=run(["bash","bootstrap_qwen.sh","offline"],events,"bootstrap_offline",allow_fail=True); result["stages"]["bootstrap_offline"]=r
            py=ROOT/".venv"/"bin"/"python"
            if os.name=="nt": py=ROOT/".venv"/"Scripts"/"python.exe"
            if not py.exists() or not py_has(py,"qwen-asr"):
                result.update(status="BLOCKED",branch="BLOCKED_PACKAGE")
            else:
                r=run([str(py),"freeze_model_identity.py"],events,"model_identity",allow_fail=True); result["stages"]["model_identity"]=r
                rdep=run([str(py),"tcc_dependency_check.py"],events,"dependency_inventory",allow_fail=True); result["stages"]["dependency_inventory"]=rdep
                snap=json.loads((ROOT/"evidence"/"dependency_snapshot.json").read_text(encoding="utf-8")) if (ROOT/"evidence"/"dependency_snapshot.json").exists() else {"execution_ready":False,"execution_blockers":["dependency_snapshot_missing"]}
                result["dependency_snapshot"]=snap
                if not snap.get("execution_ready"):
                    result.update(status="BLOCKED",branch="BLOCKED_EXECUTION_PREREQUISITE",execution_blockers=snap.get("execution_blockers",[]))
                else:
                    ro=run([str(py),"run_qwen_inference_only.py","--out","evidence/offline_predictions.jsonl"],events,"offline_inference",allow_fail=True); result["stages"]["offline_inference"]=ro
                    if ro["returncode"]!=0:
                        result.update(status="BLOCKED",branch="BLOCKED_MODEL_OR_RUNTIME")
                    else:
                        rc=run([str(py),"run_context_dictionary_v4.py"],events,"context_dictionary",allow_fail=True); result["stages"]["context_dictionary"]=rc
                        rs=run([str(py),"run_semantic_v4.py"],events,"semantic_integrity",allow_fail=True); result["stages"]["semantic_integrity"]=rs

                        # Streaming is optional for execution continuation but mandatory for Product PASS.
                        rb=run(["bash","bootstrap_qwen.sh","streaming"],events,"bootstrap_streaming",allow_fail=True); result["stages"]["bootstrap_streaming"]=rb
                        if py_has(py,"vllm"):
                            rl=run([str(py),"run_qwen_latency_v4.py","--runs","100","--warmup","5"],events,"streaming_latency",allow_fail=True); result["stages"]["streaming_latency"]=rl
                        else:
                            emit(events,"streaming_latency","BLOCKED",reason="vllm unavailable")

                        rv=run([str(py),"evaluate_tcc_v4.py"],events,"hard_ac_evaluation",allow_fail=True); result["stages"]["hard_ac_evaluation"]=rv
                        val=json.loads((ROOT/"evidence"/"validation_result.json").read_text(encoding="utf-8")) if (ROOT/"evidence"/"validation_result.json").exists() else {"status":"BLOCKED","branch":"BLOCKED_HARD_AC_NOT_RUN"}
                        result.update(status=val["status"],branch=val["branch"],validation=val)

    result["product_pass"]=result.get("status")=="PASS"
    (rd/"orchestration_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for name in ["dependency_snapshot.json","model_identity.json","scorer_contract.json","openasr_protocol.json","aqua_pair_manifest.json","latency_contract.json","hard_ac_coverage_schema.json","offline_predictions.jsonl","streaming_latency_runs.jsonl","context_dictionary_results.jsonl","context_dictionary_metrics.json","semantic_integrity_results.jsonl","semantic_integrity_metrics.json","hard_ac_coverage_matrix.json","validation_result.json"]:
        copy_if_exists(ROOT/"evidence"/name,rd/name)
    emit(events,"final_disposition",result.get("status","BLOCKED"),branch=result.get("branch"),product_pass=result["product_pass"])
    print(json.dumps({"status":result.get("status"),"branch":result.get("branch"),"run_dir":str(rd.relative_to(ROOT))},ensure_ascii=False))
    return 0 if result.get("status")=="PASS" else 2

if __name__=="__main__": raise SystemExit(main())
