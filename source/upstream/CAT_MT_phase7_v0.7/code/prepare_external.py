# 中文阅读说明｜用途：原 v0.7 的 IDX 读取、图像哈希和按重复组分层划分。
# 数据流：同像素图像按组分配，防止完全重复图像跨越训练/验证/校准集合。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
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
# 用途：解压并解析 IDX 图像/标签文件，核对头部类型和数组尺寸。
# 参数：raw、name；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：a.copy()。
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
# 用途：对数组样本或清单文件计算/核对哈希；本文件的输入形式决定具体用法。
# 参数：x；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：np.array([hashlib.sha256(v.tobytes()).hexdigest() for v in x])。
def hashes(x):return np.array([hashlib.sha256(v.tobytes()).hexdigest() for v in x])
# 用途：按类别及图像哈希分组划分数据，使重复样本不会跨集合。
# 参数：y、hh、seed；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：字典（字段见 return）。
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
# 用途：原 v0.7 的 IDX 读取、图像哈希和按重复组分层划分。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
 # 解析运行阶段/路径等命令行参数；具体允许的阶段由 choices 或分派逻辑限定。
 ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw');a=ap.parse_args()
 try:
  x=load_idx(a.raw_dir,'train-images-idx3-ubyte.gz');y=load_idx(a.raw_dir,'train-labels-idx1-ubyte.gz')
  xt=load_idx(a.raw_dir,'t10k-images-idx3-ubyte.gz');yt=load_idx(a.raw_dir,'t10k-labels-idx1-ubyte.gz')
  assert len(x)==60000 and len(xt)==10000 and len(y)==60000 and len(yt)==10000
  assert y.min()==0 and y.max()==9 and yt.min()==0 and yt.max()==9
  h=hashes(x);ht=hashes(xt);ix=grouped_split(y,h,2026090717)
  ix['test_official']=np.arange(10000);ix['test_no_train_pixel_overlap']=np.flatnonzero(~np.isin(ht,h[ix['train']]))
  out=ROOT/'data/fashion_prepared';out.mkdir(parents=True,exist_ok=True)
  # 保存具名数值数组，便于后来按同一索引回读和独立复算。
  np.savez_compressed(out/'split_indices.npz',**ix)
  np.save(out/'official_train_images.npy',x);np.save(out/'official_train_labels.npy',y)
  np.save(out/'official_test_images.npy',xt);np.save(out/'official_test_labels.npy',yt)
  np.savez_compressed(out/'image_hashes.npz',train=h,test=ht)
  card={'source':'Zalando official Fashion-MNIST','license_reference':'MIT in official repository','publisher_checksums_verified':True,'pixel_dtype':'uint8','normalization_for_training':'float32 /255; no fitted normalization','all_train_unique':int(len(set(h))),'all_test_unique':int(len(set(ht))),'official_train_test_exact_overlap_hashes':int(len(set(h)&set(ht))),'split_rule':'class-balanced whole-pixel-hash groups, approximate target 50000/5000/5000; official test retained unchanged','split_sizes':{k:len(v) for k,v in ix.items()},'perceptual_deduplication':'NOT performed','manual_label_review':'NOT performed','removed_samples':0,'training_test_metrics_allowed':True,'labels_used_for_splitting_only':True}
  # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
  (out/'data_card.json').write_text(json.dumps(card,indent=2));print(json.dumps(card,indent=2))
 except Exception as e:
  state={'external_ready':False,'stage':'data-admission gate','error':repr(e),'external_teacher_training_runs':0,'external_intervention_runs':0}
  (ROOT/'logs/external_gate.json').write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2));raise SystemExit(2)
# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=='__main__':main()
