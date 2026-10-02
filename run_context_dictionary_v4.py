from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

MODEL_ID="Qwen/Qwen3-ASR-1.7B"


def norm(s): return " ".join(unicodedata.normalize("NFKC",str(s)).lower().split())

def chars(s): return list(re.sub(r"\s+","",norm(s)))

def lev(a,b):
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

def err(ref,hyp,lang):
    if lang=="ja":
        r=chars(ref); h=chars(hyp); return 0.0 if not r else lev(r,h)/len(r)
    r=norm(ref).split(); h=norm(hyp).split(); return 0.0 if not r else lev(r,h)/len(r)

def lang_arg(x): return "English" if x=="en" else ("Japanese" if x=="ja" else None)

def tx(model,fx,context):
    out=model.transcribe(audio=str(Path("fixtures")/fx["audio"]),context=context,language=lang_arg(fx["language"]),return_time_stamps=False)
    return str(out[0].text).strip()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",default="fixtures/manifest.json"); ap.add_argument("--cases",default="fixtures/context_dictionary_manifest.json"); ap.add_argument("--out",default="evidence/context_dictionary_results.jsonl"); args=ap.parse_args()
    from qwen_asr import Qwen3ASRModel
    import torch
    man=json.loads(Path(args.manifest).read_text(encoding="utf-8")); cases=json.loads(Path(args.cases).read_text(encoding="utf-8")); by_id={x["id"]:x for x in man["fixtures"]}
    device="cuda:0" if torch.cuda.is_available() else "cpu"; dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model=Qwen3ASRModel.from_pretrained(MODEL_ID,dtype=dtype,device_map=device,max_inference_batch_size=1,max_new_tokens=256)
    rows=[]; target_total=target_hit=0; forbidden_total=forbidden_hit=0; ctx_deltas=[]
    for c in cases["cases"]:
        fx=by_id.get(c["source_fixture"])
        if not fx: continue
        base=tx(model,fx,"")
        if c["id"].startswith("dict-positive"):
            targets=c["targets"]; prompt="Preferred spellings and domain terms: "+", ".join(targets); treated=tx(model,fx,prompt)
            for t in targets: target_total+=1; target_hit+=int(norm(t) in norm(treated))
            rows.append({"id":c["id"],"fixture_id":fx["id"],"mode":"dictionary","base":base,"treated":treated,"targets":targets})
        elif c["id"].startswith("false-bias"):
            forbidden=c["forbidden_injection"]; prompt="Preferred spellings and domain terms: "+", ".join(forbidden); treated=tx(model,fx,prompt)
            for t in forbidden: forbidden_total+=1; forbidden_hit+=int(norm(t) in norm(treated))
            rows.append({"id":c["id"],"fixture_id":fx["id"],"mode":"false_bias","base":base,"treated":treated,"forbidden":forbidden})
        else:
            treated=tx(model,fx,"Active application context: preserve the speaker's wording and language; do not invent facts.")
            b=err(fx["reference"],base,fx["language"]); t=err(fx["reference"],treated,fx["language"]); ctx_deltas.append((t-b)*100.0)
            rows.append({"id":c["id"],"fixture_id":fx["id"],"mode":"context","base":base,"treated":treated,"base_error":b,"context_error":t})
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    metrics={"status":"READY" if rows else "BLOCKED","dictionary_target_recall":(target_hit/target_total if target_total else None),"false_bias_rate":(forbidden_hit/forbidden_total if forbidden_total else None),"context_false_substitution_delta_pp":(sum(ctx_deltas)/len(ctx_deltas) if ctx_deltas else None),"pairs":len(rows)}
    Path("evidence/context_dictionary_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metrics,ensure_ascii=False)); return 0 if metrics["status"]=="READY" else 2

if __name__=="__main__": raise SystemExit(main())
