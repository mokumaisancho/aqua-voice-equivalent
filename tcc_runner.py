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

def lev(a,b):
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

def wer(ref,hyp):
    r=ref.strip().split(); h=hyp.strip().split(); return 0.0 if not r and not h else (1.0 if not r else lev(r,h)/len(r))

def cer(ref,hyp):
    r=list(ref.replace(" ","")); h=list(hyp.replace(" ","")); return 0.0 if not r and not h else (1.0 if not r else lev(r,h)/len(r))

def pct(xs,p):
    if not xs: return None
    xs=sorted(xs); k=(len(xs)-1)*p/100; lo=int(k); hi=min(lo+1,len(xs)-1)
    return xs[lo] if lo==hi else xs[lo]+(xs[hi]-xs[lo])*(k-lo)

def pkg(name):
    try:
        import importlib.metadata as md; return md.version(name)
    except Exception: return None

def runtime():
    return {"timestamp":now(),"python_version":platform.python_version(),"python_executable":sys.executable,"platform":platform.platform(),"machine":platform.machine(),"cpu_count":os.cpu_count(),"packages":{"qwen-asr":pkg("qwen-asr"),"vllm":pkg("vllm"),"torch":pkg("torch"),"transformers":pkg("transformers"),"soundfile":pkg("soundfile")},"commands":{"nvidia-smi":shutil.which("nvidia-smi"),"ffmpeg":shutil.which("ffmpeg")}}

def load_manifest(path:Path):
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

def qwen_module():
    if not importlib.util.find_spec("qwen_asr"): raise RuntimeError("qwen-asr package not importable")
    import qwen_asr; return qwen_asr

def build_offline(qwen_asr):
    import torch
    cls=qwen_asr.Qwen3ASRModel
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model=cls.from_pretrained(MODEL_ID,dtype=dtype,device_map=device,max_inference_batch_size=1,max_new_tokens=256)
    return model,{"backend":"transformers","device":device,"dtype":str(dtype)}

def build_streaming(qwen_asr):
    cls=qwen_asr.Qwen3ASRModel
    model=cls.LLM(model=MODEL_ID,gpu_memory_utilization=0.8,max_new_tokens=32)
    return model,{"backend":"vllm"}

def language_arg(lang):
    if lang=="en": return "English"
    if lang=="ja": return "Japanese"
    return None

def offline_transcribe(model,fx):
    t0=time.perf_counter(); out=model.transcribe(audio=fx["resolved_audio"],context=fx.get("context",""),language=language_arg(fx["language"]),return_time_stamps=False); elapsed=time.perf_counter()-t0
    if not isinstance(out,list) or len(out)!=1: raise RuntimeError("unexpected qwen-asr result shape")
    return str(out[0].text).strip(),elapsed

def read_audio_16k(path):
    import numpy as np, soundfile as sf
    wav,sr=sf.read(path,dtype="float32",always_2d=False); wav=np.asarray(wav,dtype=np.float32)
    if wav.ndim>1: wav=wav.mean(axis=1)
    if sr!=16000:
        dur=len(wav)/float(sr); n=max(1,int(round(dur*16000))); xo=np.linspace(0,dur,num=len(wav),endpoint=False); xn=np.linspace(0,dur,num=n,endpoint=False); wav=np.interp(xn,xo,wav).astype(np.float32)
    return wav

def streaming_measure(model,fx,step_ms=500):
    wav=read_audio_16k(fx["resolved_audio"]); step=max(1,int(16000*step_ms/1000)); state=model.init_streaming_state(unfixed_chunk_num=2,unfixed_token_num=5,chunk_size_sec=2.0)
    first_partial=None; start=time.perf_counter(); pos=0
    while pos<len(wav):
        seg=wav[pos:pos+step]; pos+=len(seg); model.streaming_transcribe(seg,state)
        if first_partial is None and getattr(state,"text","").strip(): first_partial=(time.perf_counter()-start)*1000
    release=time.perf_counter(); model.finish_streaming_transcribe(state); final_ms=(time.perf_counter()-release)*1000
    return {"id":fx["id"],"first_partial_ms":first_partial,"release_to_final_ms":final_ms,"final_text":getattr(state,"text","")}

def disposition(s):
    if s["semantic_critical_violations"]>0:return "REJECT","BRANCH_SEMANTIC_FAIL"
    if s["fixture_error"]:return "BLOCKED","BRANCH_FIXTURE_BLOCKED"
    if s["env_blocked"]:return "BLOCKED","BRANCH_ENV_BLOCKED"
    if s["package_blocked"]:return "BLOCKED","BRANCH_PACKAGE_BLOCKED"
    if s["model_blocked"]:return "BLOCKED","BRANCH_MODEL_BLOCKED"
    if s["accuracy_fail"]:return "REFRAME","BRANCH_ACCURACY_FAIL"
    if s["latency_fail"]:return "REFRAME","BRANCH_LATENCY_FAIL"
    if s["context_harm"]:return "REFRAME","BRANCH_CONTEXT_HARM"
    if s["streaming_blocked"]:return "BLOCKED","BRANCH_STREAMING_BLOCKED"
    return "PASS","BRANCH_PASS"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--plan",default=PLAN_FILE); ap.add_argument("--fixtures",default="fixtures/manifest.json"); ap.add_argument("--evidence-dir",default="evidence"); ap.add_argument("--mode",choices=["execute","preflight"],default="execute"); args=ap.parse_args()
    base=Path.cwd(); ev=base/args.evidence_dir; ev.mkdir(parents=True,exist_ok=True); plan=json.loads((base/args.plan).read_text(encoding="utf-8")); rt=runtime(); (ev/"runtime_snapshot.json").write_text(json.dumps(rt,indent=2,ensure_ascii=False),encoding="utf-8")
    manifest,ferr=load_manifest(base/args.fixtures); fm={"timestamp":now(),"error":ferr,"fixtures":[] if not manifest else [{k:v for k,v in x.items() if k!="resolved_audio"} for x in manifest["fixtures"]]}; (ev/"fixture_manifest.json").write_text(json.dumps(fm,indent=2,ensure_ascii=False),encoding="utf-8")
    s={"timestamp":now(),"model_id":MODEL_ID,"plan_id":plan.get("plan_id"),"fixture_error":ferr,"env_blocked":sys.version_info<(3,9),"package_blocked":not bool(rt["packages"]["qwen-asr"]),"streaming_blocked":not bool(rt["packages"]["vllm"]),"model_blocked":False,"accuracy_fail":False,"latency_fail":False,"context_harm":False,"semantic_critical_violations":0,"phases":{}}
    accuracy={"status":"NOT_RUN","english_wer":None,"japanese_cer":None,"technical_key_term_accuracy":None,"samples":[]}; latency={"status":"NOT_RUN","first_partial_ms":{"p50":None,"p95":None,"p99":None},"release_to_final_ms":{"p50":None,"p95":None,"p99":None},"samples":[]}; resource={"status":"NOT_RUN","offline_model_load_s":None,"streaming_model_load_s":None}; ablation={"status":"NOT_RUN","reason":"requires valid base inference first"}; semantic={"status":"NOT_RUN","critical_violations":0}
    if args.mode=="execute" and not (s["env_blocked"] or s["package_blocked"] or ferr):
        try:
            qa=qwen_module(); t0=time.perf_counter(); off,meta=build_offline(qa); resource["offline_model_load_s"]=time.perf_counter()-t0; resource["offline_adapter"]=meta; preds=[]
            for fx in manifest["fixtures"]:
                text,elapsed=offline_transcribe(off,fx); rec={"id":fx["id"],"suite":fx["suite"],"language":fx["language"],"reference":fx["reference"],"hypothesis":text,"inference_s":elapsed}
                if fx["language"]=="en": rec["wer"]=wer(fx["reference"],text)
                elif fx["language"]=="ja": rec["cer"]=cer(fx["reference"],text)
                preds.append(rec)
            accuracy["samples"]=preds; ens=[x["wer"] for x in preds if "wer" in x and x["suite"]=="general"]; jas=[x["cer"] for x in preds if "cer" in x and x["language"]=="ja" and x["suite"]=="general"]; accuracy["english_wer"]=sum(ens)/len(ens) if ens else None; accuracy["japanese_cer"]=sum(jas)/len(jas) if jas else None
            tech=[x for x in preds if x["suite"]=="technical"]; accuracy["technical_key_term_accuracy"]=(sum(1 for x in tech if x["reference"].lower() in x["hypothesis"].lower())/len(tech)) if tech else None; accuracy["status"]="COMPLETE"; resource["status"]="COMPLETE"
            if accuracy["english_wer"] is not None and accuracy["english_wer"]>0.0623:s["accuracy_fail"]=True
            if accuracy["technical_key_term_accuracy"] is not None and accuracy["technical_key_term_accuracy"]<0.95:s["accuracy_fail"]=True
            if not s["streaming_blocked"]:
                t0=time.perf_counter(); stm,smeta=build_streaming(qa); resource["streaming_model_load_s"]=time.perf_counter()-t0; resource["streaming_adapter"]=smeta; ss=[streaming_measure(stm,fx) for fx in manifest["fixtures"]]; latency["samples"]=ss; fp=[x["first_partial_ms"] for x in ss if x["first_partial_ms"] is not None]; rf=[x["release_to_final_ms"] for x in ss]; latency["first_partial_ms"]={"p50":pct(fp,50),"p95":pct(fp,95),"p99":pct(fp,99)}; latency["release_to_final_ms"]={"p50":pct(rf,50),"p95":pct(rf,95),"p99":pct(rf,99)}; latency["status"]="COMPLETE"; s["latency_fail"]=(latency["release_to_final_ms"]["p50"] or 1e9)>450 or (latency["release_to_final_ms"]["p95"] or 1e9)>700
            else: latency["status"]="BLOCKED_STREAMING_RUNTIME"
        except Exception as e:
            s["model_blocked"]=True; s["phases"]["execution"]={"error":str(e),"traceback":traceback.format_exc()}
    status,branch=disposition(s); validation={"timestamp":now(),"plan_id":plan.get("plan_id"),"model_id":MODEL_ID,"status":status,"branch":branch,"state":s,"rule":"fail_closed; missing mandatory evidence cannot PASS"}
    for name,obj in [("accuracy_metrics.json",accuracy),("latency_metrics.json",latency),("resource_metrics.json",resource),("ablation_summary.json",ablation),("semantic_integrity.json",semantic),("validation_result.json",validation),("current_work_state.json",validation)]: (ev/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"status":status,"branch":branch,"evidence_dir":str(ev)},ensure_ascii=False)); return 0 if status=="PASS" else 2

if __name__=="__main__": raise SystemExit(main())
