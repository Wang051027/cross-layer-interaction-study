"""Acquire official Fashion-MNIST binaries, verify published MD5 and local SHA-256.
No experiment may proceed unless all four files pass. Does not silently substitute data.
Usage: python code/acquire_fashion.py [--offline] [--raw-dir DIR]
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, struct, urllib.request
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FILES={
 'train-images-idx3-ubyte.gz':'8d4fb7e6c68d591d4c3dfef9ec88bf0d',
 'train-labels-idx1-ubyte.gz':'25c81989df183df01b3e8a0aad5dffbe',
 't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79',
 't10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310'}
BASES=['https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/',
       'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/']
def now():return datetime.now(timezone.utc).isoformat()
# 原 v0.7 的 Fashion-MNIST 原始文件下载与校验入口。
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--offline',action='store_true');ap.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw');args=ap.parse_args()
 args.raw_dir.mkdir(parents=True,exist_ok=True);records=[];allok=True
 for name,md5 in FILES.items():
  p=args.raw_dir/name;rec={'file':name,'expected_md5':md5,'attempts':[]}
  if not p.exists() and not args.offline:
   for base in BASES:
    try:
     req=urllib.request.Request(base+name,headers={'User-Agent':'CAT-MT-data-audit/0.7'})
     with urllib.request.urlopen(req,timeout=8) as response:body=response.read()
     if hashlib.md5(body).hexdigest()!=md5:raise ValueError('official MD5 mismatch; bytes not accepted')
     p.write_bytes(body);rec['attempts'].append({'utc':now(),'url':base+name,'success':True});break
    except Exception as e:rec['attempts'].append({'utc':now(),'url':base+name,'success':False,'error':repr(e)})
  if p.exists():
   b=p.read_bytes();rec.update(bytes=len(b),actual_md5=hashlib.md5(b).hexdigest(),sha256=hashlib.sha256(b).hexdigest())
   rec['ok']=rec['actual_md5']==md5
   if rec['ok']:
    content=gzip.decompress(b);magic,n=struct.unpack('>II',content[:8]);images='images' in name
    expect=60000 if name.startswith('train') else 10000
    if images:
     _,_,h,w=struct.unpack('>IIII',content[:16]);rec['shape']=[n,h,w];rec['ok']=magic==2051 and n==expect and (h,w)==(28,28) and len(content)==16+n*h*w
    else:
     y=np.frombuffer(content[8:],np.uint8);rec['shape']=[n];rec['ok']=magic==2049 and n==expect and len(y)==n and int(y.min())==0 and int(y.max())==9
    rec['idx_magic']=magic
  else:rec.update(ok=False,error='binary file absent')
  allok&=rec['ok'];records.append(rec)
 out={'utc':now(),'official_readme':'https://github.com/zalandoresearch/fashion-mnist','records':records,'gate_passed':bool(allok),'external_experiments_allowed':bool(allok)}
 (ROOT/'logs').mkdir(exist_ok=True)
 (ROOT/'logs/fashion_acquisition.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 print(json.dumps(out,ensure_ascii=False,indent=2))
 raise SystemExit(0 if allok else 2)
if __name__=='__main__':main()
