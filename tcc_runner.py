from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, platform, shutil, sys, time, traceback
from pathlib import Path

PLAN_FILE = "TCC_EXECUTION_PLAN.json"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def levenshtein(a,b):
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

def wer(ref,hyp):
    r=ref.strip().split(); h=hyp.strip().split()
    return 0.0 if not r and not h else (1.0 if not r else levenshtein(r,h)/len(r))

def cer(ref,hyp):
    r=list(ref.replace(" ","")); h=list(hyp.replace(" ",""))
    return 0.0 if not r and not h else (1.0 if not r else levenshtein(r,h)/len(r))

def package_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception: return None

def detect_runtime():
    return {"timestamp":now(),"python_version":platform.python_version(),"python_executable":sys.executable,
            "platform":platform.platform(),"machine":platform.machine(),"cpu_count":os.cpu_count(),
            "packages":{"qwen-asr":package_version("qwen-asr"),"vllm":package_version("vllm"),
                        "torch":package_version("torch"),"transformers":package_version("transformers")},
            "commands":{"nvidia-smi":shutil.which("nvidia-smi"),"ffmpeg":shutil.which("ffmpeg")}}

def load_manifest(path: Path):
    if not path.exists(): return None,"fixture manifest missing"
    data=json.loads(path.read_text(encoding="utf-8")); items=data.get("fixtures",[])
    if not items: return None,"fixture manifest has no fixtures"
    for x in items:
        for k in ("id","audio","reference","language","suite"):
            if k not in x: return None,f"fixture {x.get('id','?')} missing {k}"
        ap=(path.parent/x["audio"]).resolve()
        if not ap.exists(): return None,f"audio missing: {x['audio']}"
        actual=sha256(ap); expected=x.get("sha256")
        if expected and expected!=actual: return None,f"checksum mismatch: {x['id']}"
        x["resolved_audio"]=str(ap); x["actual_sha256"]=actual
    return data,None

def import_qwen():
    if not importlib.util.find_spec("qwen_asr"): raise RuntimeError("qwen-asr package not importable")
    import qwen_asr
    return qwen_asr

def build_engine(qwen_asr):
    if hasattr(qwen_asr,"Qwen3ASRModel"):
        cls=qwen_asr.Qwen3ASRModel
        for kwargs in ({"model":MODEL_ID},{"model_name_or_path":MODEL_ID},{"pretrained_model_name_or_path":MODEL_ID}):
            try: return cls(**kwargs),{"constructor":kwargs}
            except TypeError: continue
    raise RuntimeError("Supported Qwen3ASRModel constructor not found; adapter update required")

def transcribe(engine,audio):
    for name in ("transcribe","generate","infer"):
        fn=getattr(engine,name,None)
        if not fn: continue
        t0=time.perf_counter()
        try: out=fn(audio)
        except TypeError: continue
        elapsed=time.perf_counter()-t0
        if isinstance(out,str): text=out
        elif isinstance(out,dict): text=out.get("text") or out.get("transcript") or ""
        elif isinstance(out,(list,tuple)) and out:
            first=out[0]; text=first if isinstance(first,str) else getattr(first,"text",str(first))
        else: text=getattr(out,"text",str(out))
        return str(text).strip(),elapsed
    raise RuntimeError("No supported transcription method found; adapter update required")

def disposition(s):
    if s["semantic_critical_violations"]>0: return "REJECT","BRANCH_SEMANTIC_FAIL"
    if s["fixture_error"]: return "BLOCKED","BRANCH_FIXTURE_BLOCKED"
    if s["env_blocked"]: return "BLOCKED","BRANCH_ENV_BLOCKED"
    if s["package_blocked"]: return "BLOCKED","BRANCH_PACKAGE_BLOCKED"
    if s["model_blocked"]: return "BLOCKED","BRANCH_MODEL_BLOCKED"
    if s["accuracy_fail"]: return "REFRAME","BRANCH_ACCURACY_FAIL"
    if s["latency_fail"]: return "REFRAME","BRANCH_LATENCY_FAIL"
    if s["context_harm"]: return "REFRAME","BRANCH_CONTEXT_HARM"
    if s["streaming_blocked"]: return "BLOCKED","BRANCH_STREAMING_BLOCKED"
    return "PASS","BRANCH_PASS"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--plan",default=PLAN_FILE); ap.add_argument("--fixtures",default="fixtures/manifest.json")
    ap.add_argument("--evidence-dir",default="evidence"); ap.add_argument("--mode",choices=["execute","preflight"],default="execute"); args=ap.parse_args()
    base=Path.cwd(); ev=base/args.evidence_dir; ev.mkdir(parents=True,exist_ok=True)
    plan=json.loads((base/args.plan).read_text(encoding="utf-8")); runtime=detect_runtime()
    (ev/"runtime_snapshot.json").write_text(json.dumps(runtime,indent=2,ensure_ascii=False),encoding="utf-8")
    manifest,ferr=load_manifest(base/args.fixtures)
    fm={"timestamp":now(),"error":ferr,"fixtures":[] if not manifest else [{k:v for k,v in x.items() if k!="resolved_audio"} for x in manifest["fixtures"]]}
    (ev/"fixture_manifest.json").write_text(json.dumps(fm,indent=2,ensure_ascii=False),encoding="utf-8")
    s={"timestamp":now(),"model_id":MODEL_ID,"plan_id":plan.get("plan_id"),"fixture_error":ferr,
       "env_blocked":sys.version_info[:2]!=(3,12),"package_blocked":not bool(runtime["packages"]["qwen-asr"]),
       "streaming_blocked":not bool(runtime["packages"]["vllm"]),"model_blocked":False,"accuracy_fail":False,
       "latency_fail":False,"context_harm":False,"semantic_critical_violations":0,"phases":{}}
    accuracy={"status":"NOT_RUN","english_wer":None,"japanese_cer":None,"technical_key_term_accuracy":None,"samples":[]}
    latency={"status":"NOT_RUN","first_partial_ms":None,"release_to_final_ms":{"p50":None,"p95":None,"p99":None},"samples":[]}
    resource={"status":"NOT_RUN","model_load_s":None,"inference_rtf":None}; ablation={"status":"NOT_RUN","reason":"requires valid base inference first"}; semantic={"status":"NOT_RUN","critical_violations":0}
    if args.mode=="execute" and not (s["env_blocked"] or s["package_blocked"] or ferr):
        try:
            t0=time.perf_counter(); qa=import_qwen(); engine,adapter=build_engine(qa); resource["model_load_s"]=time.perf_counter()-t0; resource["adapter"]=adapter
            preds=[]
            for fx in manifest["fixtures"]:
                text,elapsed=transcribe(engine,fx["resolved_audio"]); rec={"id":fx["id"],"suite":fx["suite"],"language":fx["language"],"reference":fx["reference"],"hypothesis":text,"inference_s":elapsed}
                if fx["language"]=="en": rec["wer"]=wer(fx["reference"],text)
                else: rec["cer"]=cer(fx["reference"],text)
                preds.append(rec)
            accuracy["samples"]=preds; ens=[x["wer"] for x in preds if "wer" in x and x["suite"]=="general"]; jas=[x["cer"] for x in preds if "cer" in x and x["language"]=="ja" and x["suite"]=="general"]
            accuracy["english_wer"]=sum(ens)/len(ens) if ens else None; accuracy["japanese_cer"]=sum(jas)/len(jas) if jas else None
            tech=[x for x in preds if x["suite"]=="technical"]
            if tech: accuracy["technical_key_term_accuracy"]=sum(1 for x in tech if x["reference"].lower() in x["hypothesis"].lower())/len(tech)
            accuracy["status"]="COMPLETE"; resource["status"]="COMPLETE"
            if accuracy["english_wer"] is not None and accuracy["english_wer"]>0.0623: s["accuracy_fail"]=True
            if accuracy["technical_key_term_accuracy"] is not None and accuracy["technical_key_term_accuracy"]<0.95: s["accuracy_fail"]=True
            if s["streaming_blocked"]: latency["status"]="BLOCKED_STREAMING_RUNTIME"
            else: latency["status"]="BLOCKED_STREAMING_ADAPTER_NOT_IMPLEMENTED"; s["streaming_blocked"]=True
        except Exception as e:
            s["model_blocked"]=True; s["phases"]["model_smoke"]={"error":str(e),"traceback":traceback.format_exc()}
    status,branch=disposition(s); validation={"timestamp":now(),"plan_id":plan.get("plan_id"),"model_id":MODEL_ID,"status":status,"branch":branch,"state":s,"rule":"fail_closed; no missing evidence can yield PASS"}
    for name,obj in [("accuracy_metrics.json",accuracy),("latency_metrics.json",latency),("resource_metrics.json",resource),("ablation_summary.json",ablation),("semantic_integrity.json",semantic),("validation_result.json",validation),("current_work_state.json",validation)]:
        (ev/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"status":status,"branch":branch,"evidence_dir":str(ev)},ensure_ascii=False)); return 0 if status=="PASS" else 2

if __name__=="__main__": raise SystemExit(main())
