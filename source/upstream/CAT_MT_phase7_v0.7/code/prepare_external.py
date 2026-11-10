"""Hard-gated Fashion-MNIST preparation; no missing-file fallback to another task.
Validates publisher MD5s, exact IDX sizes, image hashes and labels, then keeps duplicate
training images in the same split. Official test split is unchanged, with a separately
recorded train-overlap-free sensitivity subset. Approximate 50k/5k/5k sizes are recorded.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,struct
from pathlib import Path
import numpy as np
from acquire_fashion import FILES
ROOT=Path(__file__).resolve().parents[1]
# 解压并解析 IDX 图像/标签文件，核对头部类型和数组尺寸。
def load_idx(raw,name):
 p=raw/name
 if not p.is_file():raise FileNotFoundError(f'Required official binary not present: {p}')
 b=p.read_bytes()
 if hashlib.md5(b).hexdigest()!=FILES[name]:raise ValueError(f'Publisher MD5 mismatch: {name}')
 c=gzip.decompress(b);magic,n=struct.unpack('>II',c[:8])
 if magic==2051:
  _,_,h,w=struct.unpack('>IIII',c[:16]);a=np.frombuffer(c[16:],np.uint8).reshape(n,h,w)
  if (h,w)!=(28,28):raise ValueError('Unexpected image shape')
 elif magic==2049:a=np.frombuffer(c[8:],np.uint8).reshape(n)
 else:raise ValueError('Unexpected IDX magic')
 return a.copy()
# 对数组样本或清单文件计算/核对哈希；本文件的输入形式决定具体用法。
def hashes(x):return np.array([hashlib.sha256(v.tobytes()).hexdigest() for v in x])
# 按类别及图像哈希分组划分数据，使重复样本不会跨集合。
def grouped_split(y,hh,seed):
 rng=np.random.default_rng(seed);groups={}
 for i,h in enumerate(hh):groups.setdefault(str(h),[]).append(i)
 for h,ids in groups.items():
  if len(np.unique(y[ids]))!=1:raise ValueError(f'Identical pixels have conflicting labels: {h}; human data decision required')
 splits={'train':[],'validation':[],'calibration':[]}
 for cl in range(10):
  gg=[ii for ii in groups.values() if y[ii[0]]==cl];rng.shuffle(gg)
  count={'validation':0,'calibration':0,'train':0}
  for ids in gg:
   sp='validation' if count['validation']<500 else 'calibration' if count['calibration']<500 else 'train'
   splits[sp].extend(ids);count[sp]+=len(ids)
 return {k:rng.permutation(np.array(v,dtype=np.int64)) for k,v in splits.items()}
# 原 v0.7 的 IDX 读取、图像哈希和按重复组分层划分。
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw');a=ap.parse_args()
 try:
  x=load_idx(a.raw_dir,'train-images-idx3-ubyte.gz');y=load_idx(a.raw_dir,'train-labels-idx1-ubyte.gz')
  xt=load_idx(a.raw_dir,'t10k-images-idx3-ubyte.gz');yt=load_idx(a.raw_dir,'t10k-labels-idx1-ubyte.gz')
  assert len(x)==60000 and len(xt)==10000 and len(y)==60000 and len(yt)==10000
  assert y.min()==0 and y.max()==9 and yt.min()==0 and yt.max()==9
  h=hashes(x);ht=hashes(xt);ix=grouped_split(y,h,2026090717)
  ix['test_official']=np.arange(10000);ix['test_no_train_pixel_overlap']=np.flatnonzero(~np.isin(ht,h[ix['train']]))
  out=ROOT/'data/fashion_prepared';out.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(out/'split_indices.npz',**ix)
  np.save(out/'official_train_images.npy',x);np.save(out/'official_train_labels.npy',y)
  np.save(out/'official_test_images.npy',xt);np.save(out/'official_test_labels.npy',yt)
  np.savez_compressed(out/'image_hashes.npz',train=h,test=ht)
  card={'source':'Zalando official Fashion-MNIST','license_reference':'MIT in official repository','publisher_checksums_verified':True,'pixel_dtype':'uint8','normalization_for_training':'float32 /255; no fitted normalization','all_train_unique':int(len(set(h))),'all_test_unique':int(len(set(ht))),'official_train_test_exact_overlap_hashes':int(len(set(h)&set(ht))),'split_rule':'class-balanced whole-pixel-hash groups, approximate target 50000/5000/5000; official test retained unchanged','split_sizes':{k:len(v) for k,v in ix.items()},'perceptual_deduplication':'NOT performed','manual_label_review':'NOT performed','removed_samples':0,'training_test_metrics_allowed':True,'labels_used_for_splitting_only':True}
  (out/'data_card.json').write_text(json.dumps(card,indent=2));print(json.dumps(card,indent=2))
 except Exception as e:
  state={'external_ready':False,'stage':'data-admission gate','error':repr(e),'external_teacher_training_runs':0,'external_intervention_runs':0}
  (ROOT/'logs/external_gate.json').write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2));raise SystemExit(2)
if __name__=='__main__':main()
