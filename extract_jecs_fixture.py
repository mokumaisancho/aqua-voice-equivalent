from __future__ import annotations
import hashlib, json, os, re, shutil, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FIX=ROOT/'fixtures'; MAN=FIX/'manifest.json'; ARCH=Path(os.environ.get('JECS_ZIP', FIX/'jecs.zip'))
JP=re.compile(r'[ぁ-んァ-ン一-龯]'); EN=re.compile(r'\b[A-Za-z]{2,}\b')
AUDIO={'.wav','.flac'}; TEXT={'.txt','.tsv','.csv','.lab'}
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''): h.update(c)
 return h.hexdigest()
def mixed(s): return bool(JP.search(s) and EN.search(s))
def norm(s):
 s=s.strip().strip('\ufeff'); parts=re.split(r'\t+|\s{2,}',s,maxsplit=1); return parts[-1].strip() if len(parts)>1 else s
def score(t,a):
 ts=Path(t).stem.lower(); ap=Path(a); ast=ap.stem.lower(); sc=0
 if ts==ast: sc+=100
 if ts and ts in ast: sc+=30
 if ap.parent==Path(t).parent: sc+=20
 nums=set(re.findall(r'\d+',ts)); sc+=5*len(nums & set(re.findall(r'\d+',ast))); return sc
def main():
 if not ARCH.exists():
  print(json.dumps({'status':'SKIP','reason':'JECS_ZIP missing','expected':str(ARCH)})); return 3
 with zipfile.ZipFile(ARCH) as z:
  names=z.namelist(); aud=[n for n in names if Path(n).suffix.lower() in AUDIO]; txt=[n for n in names if Path(n).suffix.lower() in TEXT]; cand=[]
  for tn in txt:
   try: body=z.read(tn).decode('utf-8-sig')
   except Exception: continue
   for i,line in enumerate(body.splitlines()):
    r=norm(line)
    if mixed(r): cand.append((tn,i,r))
  if not cand: raise SystemExit('No mixed JA/EN transcript found')
  cand.sort(key=lambda x:(x[0],x[1],x[2])); tn,i,ref=cand[0]
  ranked=sorted(((score(tn,a),a) for a in aud), reverse=True); ranked=[x for x in ranked if x[0]>0]
  if not ranked: raise SystemExit('No audio mapping candidate')
  if len(ranked)>1 and ranked[0][0]==ranked[1][0]: raise SystemExit('Ambiguous audio mapping')
  sc,an=ranked[0]; out=FIX/'jecs-code-switch-001.wav'
  with z.open(an) as src, out.open('wb') as dst: shutil.copyfileobj(src,dst)
 data=json.loads(MAN.read_text()) if MAN.exists() else {'schema_version':1,'frozen':True,'fixtures':[]}
 data['fixtures']=[x for x in data.get('fixtures',[]) if x.get('id')!='jecs-code-switch-001']
 data['fixtures'].append({'id':'jecs-code-switch-001','audio':out.name,'reference':ref,'language':'en-ja-code-switch','suite':'code_switch','sha256':sha(out),'corpus':'JECS','license':'text CC BY 3.0; audio research/personal-use terms','source_transcript_file':tn,'source_transcript_line':i+1,'source_audio_file':an,'mapping_score':sc})
 data['fixtures'].sort(key=lambda x:x['id']); MAN.write_text(json.dumps(data,ensure_ascii=False,indent=2))
 print(json.dumps({'status':'OK','reference':ref,'audio':an,'sha256':sha(out)},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())