# 中文阅读说明｜用途：原始 digits/合成条纹研究的训练、状态编码、代理模型与统计工具。
# 数据流：这是较早研究分支，并非本轮 Fashion-MNIST 的训练入口；保留其原有算法。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
from __future__ import annotations
import hashlib,itertools,json,math,os,random
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Dict,List,Mapping,Sequence,Tuple
import numpy as np,pandas as pd,torch,torch.nn as nn,torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from torch.utils.data import DataLoader,TensorDataset
Array=np.ndarray

# 用途：计算给定字节串的 SHA-256，标识数组或数据内容。
# 参数：b；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：hashlib.sha256(b).hexdigest()。
def hbytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
# 用途：按块读取文件并计算 SHA-256，避免一次读入大文件。
# 参数：p；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：h.hexdigest()。
def hfile(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
# 用途：设置 Python、NumPy 和 PyTorch 随机种子，并配置 CPU 线程与确定性算法。
# 参数：s；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def seed_all(s:int):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)));torch.use_deterministic_algorithms(True,warn_only=True)
# 用途：保存一个数据划分的图像 x、标签 y 和样本标识 ids。
# 结构：成员字段和方法共同保存本阶段需要的状态。
@dataclass(frozen=True)
class Split: x:Array;y:Array;ids:Array
# 用途：早期手写数字任务的 CNN，trace 暴露各层预激活和输出。
# 结构：继承 nn.Module；成员字段和方法共同保存本阶段需要的状态。
class DigitsCNN(nn.Module):
 # 用途：建立对象所需的网络层、固定映射或配置。早期手写数字任务的 CNN，trace 暴露各层预激活和输出。
 # 参数：无额外参数；读取或初始化对象状态。
 # 返回：初始化对象字段，没有显式返回值。
 def __init__(self):
  super().__init__();self.conv1=nn.Conv2d(1,8,3,padding=1);self.conv2=nn.Conv2d(8,16,3,padding=1);self.fc1=nn.Linear(16*4*4,64);self.fc2=nn.Linear(64,10);self.transition_padding_mode='zeros'
 # 用途：执行前向并返回各层预激活、激活与最终输出，供后续状态分析。
 # 参数：x；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：字典（字段见 return）。
 def trace(self,x):
  z1=self.conv1(x);a1=F.relu(z1);z2=self.conv2(a1);a2=F.relu(z2);z3=self.fc1(F.max_pool2d(a2,2).flatten(1));a3=F.relu(z3);o=self.fc2(a3);return {'z1':z1,'a1':a1,'z2':z2,'a2':a2,'z3':z3,'a3':a3,'logits':o}
 # 用途：执行该网络的前向计算并返回分类 logits；训练时反向传播由外部训练循环控制。
 # 参数：x；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：self.trace(x)['logits']。
 def forward(self,x):return self.trace(x)['logits']
# 用途：早期合成条纹任务的 CNN，卷积采用循环边界。
# 结构：继承 nn.Module；成员字段和方法共同保存本阶段需要的状态。
class SyntheticCNN(nn.Module):
 # 用途：建立对象所需的网络层、固定映射或配置。早期合成条纹任务的 CNN，卷积采用循环边界。
 # 参数：无额外参数；读取或初始化对象状态。
 # 返回：初始化对象字段，没有显式返回值。
 def __init__(self):
  super().__init__();self.conv1=nn.Conv2d(1,6,3,padding=1,padding_mode='circular');self.conv2=nn.Conv2d(6,12,3,padding=1,padding_mode='circular');self.fc1=nn.Linear(12*8*8,48);self.fc2=nn.Linear(48,2);self.transition_padding_mode='circular'
 # 用途：执行前向并返回各层预激活、激活与最终输出，供后续状态分析。
 # 参数：x；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：字典（字段见 return）。
 def trace(self,x):
  z1=self.conv1(x);a1=F.relu(z1);z2=self.conv2(a1);a2=F.relu(z2);z3=self.fc1(F.max_pool2d(a2,2).flatten(1));a3=F.relu(z3);o=self.fc2(a3);return {'z1':z1,'a1':a1,'z2':z2,'a2':a2,'z3':z3,'a3':a3,'logits':o}
 # 用途：执行该网络的前向计算并返回分类 logits；训练时反向传播由外部训练循环控制。
 # 参数：x；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：self.trace(x)['logits']。
 def forward(self,x):return self.trace(x)['logits']
# 用途：把 Split 中的图像和标签包装为具有固定打乱种子的 PyTorch DataLoader。
# 参数：s、b、sh、seed；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：DataLoader(TensorDataset(torch.from_numpy(s.x), torch.from_numpy(s.y)), batch_size=b, s…。
def loader(s:Split,b:int,sh:bool,seed:int):return DataLoader(TensorDataset(torch.from_numpy(s.x),torch.from_numpy(s.y)),batch_size=b,shuffle=sh,generator=torch.Generator().manual_seed(seed),num_workers=0)
# 用途：在评价模式批量输出 logits 及其 argmax 类别。
# 参数：m、s、seed；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(z, z.argmax(1))。
def predict(m,s,seed):
 m.eval();z=[]
 with torch.no_grad():
  for xb,_ in loader(s,256,False,seed):z.append(m(xb).numpy())
 z=np.concatenate(z);return z,z.argmax(1)
# 用途：用 AdamW 训练早期教师，按验证准确率选择检查点并在无改善时提前停止。
# 参数：m、tr、va、seed、maxe、pat、lr、wd；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(m, {'best_epoch': be, 'best_validation_accuracy': ba, 'history': hist})。
def train(m,tr,va,seed,maxe,pat,lr,wd):
 seed_all(seed);op=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd);sch=torch.optim.lr_scheduler.ReduceLROnPlateau(op,mode='max',factor=.5,patience=4);best=None;ba=-1.;be=0;st=0;hist=[]
 for e in range(1,maxe+1):
  m.train();ls=[]
  # 分类交叉熵是教师学习/评价的任务损失，区别于后续干预响应的 NRMSE。
  # 这里才会计算训练梯度并更新模型；后续干预阶段固定模型权重。
  for xb,yb in loader(tr,64,True,seed):op.zero_grad(set_to_none=True);l=F.cross_entropy(m(xb),yb);l.backward();op.step();ls.append(float(l.detach()))
  _,p=predict(m,va,seed);a=float(accuracy_score(va.y,p));sch.step(a);hist.append({'epoch':e,'train_loss':float(np.mean(ls)),'validation_accuracy':a})
  if a>ba+1e-8:ba=a;be=e;best={k:v.detach().clone() for k,v in m.state_dict().items()};st=0
  else:st+=1
  if st>=pat:break
 if best is None:raise RuntimeError('no checkpoint')
 m.load_state_dict(best);m.eval();return m,{'best_epoch':be,'best_validation_accuracy':ba,'history':hist}
# 用途：批量保存各层预激活、激活和 logits，并附上标签、标识和教师预测。
# 参数：m、s、seed；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：o。
def trace(m,s,seed):
 o={k:[] for k in ['z1','a1','z2','a2','z3','a3','logits']};m.eval()
 with torch.no_grad():
  for xb,_ in loader(s,128,False,seed):
   t=m.trace(xb)
   for k in o:o[k].append(t[k].numpy())
 o={k:np.concatenate(v) for k,v in o.items()};o['y']=s.y.copy();o['ids']=s.ids.copy();o['teacher_pred']=o['logits'].argmax(1);return o
# 用途：拼接各层 ReLU 开关模式并压缩为位串，统计不同全局模式数量。
# 参数：t；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(int(np.unique(np.packbits(b, axis=1), axis=0).shape[0]), int(b.shape[1]))。
def global_patterns(t):
 b=np.concatenate([(t['z1']>0).reshape(len(t['y']),-1),(t['z2']>0).reshape(len(t['y']),-1),(t['z3']>0).reshape(len(t['y']),-1)],1).astype(np.uint8);return int(np.unique(np.packbits(b,axis=1),axis=0).shape[0]),int(b.shape[1])
# 用途：按第二次卷积的窗口展开输入；循环边界与零填充分别处理。
# 参数：x、m；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：F.unfold(F.pad(x, (p, p, p, p), mode='circular'), k, padding=0, stride=c.stride).transp… / F.unfold(x, k, padding=p, stride=c.stride).transpose(1, 2)。
def unfold(x,m):
 c=m.conv2;k=c.kernel_size[0];p=c.padding[0]
 if m.transition_padding_mode=='circular':return F.unfold(F.pad(x,(p,p,p,p),mode='circular'),k,padding=0,stride=c.stride).transpose(1,2)
 return F.unfold(x,k,padding=p,stride=c.stride).transpose(1,2)
# 用途：保存源/目标门控、激活及正负贡献等局部描述量。
# 结构：成员字段和方法共同保存本阶段需要的状态。
@dataclass
class Components:
 sm:Array;sa:Array;tm:Array;ta:Array;pc:Array;nc:Array
 # 用途：按方法需要拼接源、目标或联合描述量，输出 float32 特征。
 # 参数：k；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：np.concatenate(d, 2).astype(np.float32, copy=False)。
 def desc(self,k):
  d={'source':[self.sm,self.sa],'target':[self.tm,self.ta],'joint_ma':[self.sm,self.tm,self.sa,self.ta],'joint_mac':[self.sm,self.tm,self.sa,self.ta,self.pc,self.nc]}[k];return np.concatenate(d,2).astype(np.float32,copy=False)
# 用途：从中间轨迹构造源/目标门控与激活，以及卷积正、负贡献描述量。
# 参数：m、t；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：Components(**{k: np.concatenate(v) for k, v in o.items()})。
def components(m,t):
 z1=torch.from_numpy(t['z1']);a1=torch.from_numpy(t['a1']);z2=torch.from_numpy(t['z2']);a2=torch.from_numpy(t['a2']);w=m.conv2.weight.detach().reshape(m.conv2.out_channels,-1);o={k:[] for k in ['sm','sa','tm','ta','pc','nc']}
 for s in range(0,len(a1),96):
  zz=z1[s:s+96];aa=a1[s:s+96];ww=z2[s:s+96];bb=a2[s:s+96];sa=unfold(aa,m);sm=unfold((zz>0).float(),m);B,P,_=sa.shape;tm=(ww>0).permute(0,2,3,1).reshape(B,P,m.conv2.out_channels).float();ta=bb.permute(0,2,3,1).reshape(B,P,m.conv2.out_channels);e=sa[:,:,None,:]*w[None,None,:,:];pc=e.clamp_min(0).sum(3);nc=(-e.clamp_max(0)).sum(3)
  for k,v in [('sm',sm),('sa',sa),('tm',tm),('ta',ta),('pc',pc),('nc',nc)]:o[k].append(v.numpy().astype(np.float32,copy=False))
 return Components(**{k:np.concatenate(v) for k,v in o.items()})
# 用途：组合标准化、PCA 和 MiniBatchKMeans，将描述量映射为离散状态及距离。
# 结构：成员字段和方法共同保存本阶段需要的状态。
@dataclass
class Encoder:
 sc:StandardScaler;pca:PCA;km:MiniBatchKMeans;thr:float;K:int
 # 用途：对描述量应用已拟合的标准化、PCA 和聚类，返回最近簇编号及距离。
 # 参数：d；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：(lab.reshape(d.shape[:2]), dis.reshape(d.shape[:2]))。
 def assign(self,d):
  f=d.reshape(-1,d.shape[-1]);lab=np.empty(len(f),np.int32);dis=np.empty(len(f),np.float32)
  for s in range(0,len(f),25000):
   z=self.pca.transform(self.sc.transform(f[s:s+25000]));dd=self.km.transform(z);lab[s:s+25000]=dd.argmin(1);dis[s:s+25000]=dd.min(1)
  return lab.reshape(d.shape[:2]),dis.reshape(d.shape[:2])
# 用途：从训练描述量固定抽样，拟合标准化、PCA、MiniBatchKMeans 及覆盖距离阈值。
# 参数：d、K、seed、maxp、pcs、q；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：Encoder(sc, pca, km, thr, K)。
def fitenc(d,K,seed,maxp,pcs,q):
 f=d.reshape(-1,d.shape[-1]);r=np.random.default_rng(seed);z=f[r.choice(len(f),min(maxp,len(f)),False)];sc=StandardScaler();z=sc.fit_transform(z);pca=PCA(n_components=max(2,min(pcs,z.shape[1],z.shape[0]-1)),random_state=seed,svd_solver='randomized');z=pca.fit_transform(z);km=MiniBatchKMeans(n_clusters=K,random_state=seed,n_init=1,batch_size=1024,max_iter=60,reassignment_ratio=.01).fit(z);thr=float(np.quantile(km.transform(z).min(1),q));return Encoder(sc,pca,km,thr,K)
# 用途：按方法分别拟合源/目标编码器或一个联合编码器。
# 参数：method、c、cfg、seed；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：字典（字段见 return）。
def fitmethod(method,c,cfg,seed):
 kw=dict(maxp=cfg['max_fit_points'],pcs=cfg['pca_components'],q=cfg['coverage_quantile'])
 # 独立编码方法分别离散化源与目标，再用笛卡尔积组合状态。
 if method=='ilt':return {'s':fitenc(c.desc('source'),cfg['ilt_source_k'],seed+11,**kw),'t':fitenc(c.desc('target'),cfg['ilt_target_k'],seed+29,**kw)}
 return {'j':fitenc(c.desc(method),cfg['joint_k'],seed+47,**kw)}
# 用途：把编码器结果转换为方法的离散状态编号和覆盖掩码。
# 参数：method、c、e、cfg；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：((s * cfg['ilt_target_k'] + t).astype(np.int32), (ds <= e['s'].thr) & (dt <= e['t'].thr)) / (a.astype(np.int32), d <= e['j'].thr)。
def assign(method,c,e,cfg):
 if method=='ilt':
  s,ds=e['s'].assign(c.desc('source'));t,dt=e['t'].assign(c.desc('target'));return (s*cfg['ilt_target_k']+t).astype(np.int32),(ds<=e['s'].thr)&(dt<=e['t'].thr)
 a,d=e['j'].assign(c.desc(method));return a.astype(np.int32),d<=e['j'].thr
# 用途：把局部离散状态汇总为全图或空间分区的频率直方图。
# 参数：a、K、mode；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：x。
def pool(a,K,mode):
 n,P=a.shape;S=int(round(math.sqrt(P)));m=a.reshape(n,S,S)
 if mode=='global':rs=[(0,S,0,S)]
 else:q=S//2;rs=[(0,S,0,S),(0,q,0,q),(0,q,q,S),(q,S,0,q),(q,S,q,S)]
 x=np.zeros((n,len(rs)*K),np.float32)
 for j,(r0,r1,c0,c1) in enumerate(rs):
  v=m[:,r0:r1,c0:c1].reshape(n,-1)
  for i in range(n):x[i,j*K:(j+1)*K]=np.bincount(v[i],minlength=K)/v.shape[1]
 return x
# 用途：从位置覆盖掩码计算平均覆盖率及达到 90% 覆盖的样本比例。
# 参数：c；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：字典（字段见 return）。
def cov(c):
 p=c.mean(1);return {'local_coverage':float(c.mean()),'sample_coverage_at_90pct':float((p>=.9).mean()),'mean_sample_coverage':float(p.mean())}
# 用途：比较代理 logits 与教师/真实标签，返回教师保真度、任务准确率和 logit MAE。
# 参数：z、t；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：字典（字段见 return）。
def metrics(z,t):
 p=z.argmax(1);return {'fidelity':float(accuracy_score(t['teacher_pred'],p)),'task_accuracy':float(accuracy_score(t['y'],p)),'logit_mae':float(mean_absolute_error(t['logits'],z))}
# 用途：在验证集上比较岭回归系数，优先教师保真度，其次 logit 误差和正则强度。
# 参数：x、y、xv、tv、als；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(best[1], best[2])。
def selridge(x,y,xv,tv,als):
 best=None
 for a in als:
  m=Ridge(alpha=a).fit(x,y);q=metrics(m.predict(xv),tv);key=(q['fidelity'],-q['logit_mae'],-a)
  if best is None or key>best[0]:best=(key,m,{'alpha':a,**q})
 return best[1],best[2]
# 用途：先用回归树分区域，再在叶节点拟合岭回归；样本不足的叶使用全局模型。
# 结构：成员字段和方法共同保存本阶段需要的状态。
class ALT:
 # 用途：建立对象所需的网络层、固定映射或配置。先用回归树分区域，再在叶节点拟合岭回归；样本不足的叶使用全局模型。
 # 参数：d、n、a、s；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：初始化对象字段，没有显式返回值。
 def __init__(self,d,n,a,s):self.tr=DecisionTreeRegressor(max_depth=d,min_samples_leaf=n,random_state=s);self.a=a;self.g=Ridge(alpha=a);self.ms={}
 # 用途：拟合全局树与岭回归，再对至少三个样本的叶节点拟合各自线性模型。
 # 参数：x、y；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：self。
 def fit(self,x,y):
  self.tr.fit(x,y);self.g.fit(x,y);l=self.tr.apply(x)
  for z in np.unique(l):
   ix=np.flatnonzero(l==z)
   if len(ix)>=3:self.ms[int(z)]=Ridge(alpha=self.a).fit(x[ix],y[ix])
  return self
 # 用途：将样本路由到树叶，使用对应叶模型；没有叶模型时使用全局岭回归。
 # 参数：x；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：o。
 def predict(self,x):
  l=self.tr.apply(x);o=np.empty((len(x),self.g.coef_.shape[0]))
  for z in np.unique(l):
   ix=np.flatnonzero(l==z);o[ix]=self.ms.get(int(z),self.g).predict(x[ix])
  return o
 # 用途：返回已拟合回归树的叶节点数，作为模型复杂度指标。
 # 参数：无额外参数；读取或初始化对象状态。
 # 返回：int(self.tr.get_n_leaves())。
 @property
 def leaves(self):return int(self.tr.get_n_leaves())
# 用途：在验证集上选择仿射叶树的深度与正则化，保持复杂度的并列优先规则。
# 参数：x、y、xv、tv、ds、als、n、seed；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(best[1], best[2])。
def seltree(x,y,xv,tv,ds,als,n,seed):
 best=None
 for d in ds:
  for a in als:
   m=ALT(d,n,a,seed).fit(x,y);q=metrics(m.predict(xv),tv);key=(q['fidelity'],-q['logit_mae'],-m.leaves)
   if best is None or key>best[0]:best=(key,m,{'depth':d,'alpha':a,'n_leaves':m.leaves,**q})
 return best[1],best[2]
# 用途：用教师预测标签拟合分类树，在验证集选择保真度高且叶数少的模型。
# 参数：x、y、xv、yv、ds、n、seed；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(best[1], best[2])。
def selclf(x,y,xv,yv,ds,n,seed):
 best=None
 for d in ds:
  m=DecisionTreeClassifier(max_depth=d,min_samples_leaf=n,random_state=seed).fit(x,y);f=float(accuracy_score(yv,m.predict(xv)));key=(f,-m.get_n_leaves())
  if best is None or key>best[0]:best=(key,m,{'depth':d,'n_leaves':int(m.get_n_leaves()),'fidelity':f})
 return best[1],best[2]
# 用途：对每张图像的连续内存字节计算哈希，用于完全重复检查。
# 参数：x；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：[hbytes(np.ascontiguousarray(r).tobytes()) for r in x]。
def hashes(x):return [hbytes(np.ascontiguousarray(r).tobytes()) for r in x]
# 用途：分批计算测试图像到训练图像的最近欧氏距离，避免建立完整距离矩阵。
# 参数：tr、te；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：np.asarray(o)。
def crossdist(tr,te):
 a=tr.reshape(len(tr),-1).astype(np.float32);b=te.reshape(len(te),-1).astype(np.float32);aa=(a*a).sum(1)[None,:];o=[]
 for s in range(0,len(b),128):
  q=b[s:s+128];d=(q*q).sum(1)[:,None]+aa-2*q@a.T;o.extend(np.sqrt(np.maximum(d.min(1),0)))
 return np.asarray(o)
# 用途：读取 sklearn digits，固定分层划分，并返回数据、质量审计和划分索引。
# 参数：split_seed；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(sp, audit, {'train': tr, 'validation': va, 'test': te})。
def digits(split_seed):
 d=load_digits();xr=d.images.astype(np.float32);y=d.target.astype(np.int64);ids=np.arange(len(y));tr,tmp=train_test_split(ids,test_size=.4,random_state=split_seed,stratify=y);va,te=train_test_split(tmp,test_size=.5,random_state=split_seed,stratify=y[tmp]);x=(xr/16.)[:,None];sp={'train':Split(x[tr],y[tr],tr),'validation':Split(x[va],y[va],va),'test':Split(x[te],y[te],te)};ds=crossdist(sp['train'].x,sp['test'].x);hs=hashes(xr);audit={'dataset':'sklearn_digits','dataset_role':'internal replication; prior pilot used same corpus','n_total':len(y),'shape':list(x.shape),'class_counts':{str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))},'pixel_min':float(x.min()),'pixel_max':float(x.max()),'exact_unique_images':len(set(hs)),'exact_duplicate_images':len(hs)-len(set(hs)),'raw_data_sha256':hbytes(np.ascontiguousarray(xr).tobytes()),'raw_labels_sha256':hbytes(np.ascontiguousarray(y).tobytes()),'split_seed':split_seed,'split_sizes':{k:len(v.y) for k,v in sp.items()},'test_to_train_nearest_l2':{'min':float(ds.min()),'median':float(np.median(ds)),'p05':float(np.quantile(ds,.05))}};return sp,audit,{'train':tr,'validation':va,'test':te}
# 用途：在数组给定位置累加一根竖条，用于合成条纹图像。
# 参数：a、x、y、v；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def dv(a,x,y,v):a[y-2:y+3,x-1:x+2]+=v
# 用途：在数组给定位置累加一根横条，用于合成条纹图像。
# 参数：a、x、y、v；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def dh(a,x,y,v):a[y-1:y+2,x-2:x+3]+=v
# 用途：生成共享背景的成对条纹图像，标签由横竖条左右关系决定，并检查配对背景一致。
# 参数：n、seed、name、S、noise、ood；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(Split(x, y, ids), meta, audit)。
def gensplit(n,seed,name,S,noise,ood=False):
 r=np.random.default_rng(seed);xs=[];ys=[];ids=[];rec=[];checks=[]
 for b in range(n):
  lx=int(r.choice([2,3,6] if ood else [3,4,5]));rx=int(r.choice([9,12,13] if ood else [10,11,12]));rx=max(rx,lx+4);vy=int(r.integers(4,S-4));hy=int(r.integers(4,S-4));vv=float(r.uniform(.85,1.15));hv=float(r.uniform(.85,1.15));ns=noise*1.75 if ood else noise;bg=r.normal(0,ns,(S,S)).astype(np.float32);pair=[]
  for lab in [0,1]:
   vx,hx=(lx,rx) if lab==0 else (rx,lx);im=bg.copy();dv(im,vx,vy,vv);dh(im,hx,hy,hv);im=np.clip(im,0,1).astype(np.float32);sid=f'{name}-b{b:05d}-y{lab}';bid=f'{name}-b{b:05d}';xs.append(im[None]);ys.append(lab);ids.append(sid);pair.append(im);rec.append({'sample_id':sid,'split':name,'base_scene_id':bid,'counterfactual_pair_id':bid,'label':lab,'vertical_x':vx,'vertical_y':vy,'horizontal_x':hx,'horizontal_y':hy,'left_anchor_x':lx,'right_anchor_x':rx,'vertical_intensity':vv,'horizontal_intensity':hv,'noise_sd':ns,'generator_seed':seed,'generator_version':'paired-bars-v0.2','ood':ood,'image_sha256':hbytes(im.tobytes())})
  u=np.zeros((S,S),bool)
  for x in [lx,rx]:u[vy-2:vy+3,x-1:x+2]=1;u[hy-1:hy+2,x-2:x+3]=1
  checks.append(np.array_equal(pair[0][~u],pair[1][~u]))
 x=np.stack(xs);y=np.array(ys,np.int64);ids=np.array(ids);o=r.permutation(len(y));x,y,ids=x[o],y[o],ids[o];meta=pd.DataFrame(rec).set_index('sample_id').loc[ids].reset_index();hs=hashes(x);audit={'split':name,'n_base_scenes':n,'n_images':len(y),'class_counts':{str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))},'pixel_min':float(x.min()),'pixel_max':float(x.max()),'all_pair_background_checks_pass':bool(all(checks)),'exact_unique_images':len(set(hs)),'array_sha256':hbytes(x.tobytes()),'labels_sha256':hbytes(y.tobytes())};return Split(x,y,ids),meta,audit
# 用途：按配置生成训练、验证、测试及分布外条纹数据，并检查跨集合标识和重复。
# 参数：cfg；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(sp, M, audit)。
def synthetic(cfg):
 s=cfg['synthetic'];sp={};ms=[];au={}
 for n in ['train','validation','test','ood_test']:
  sp[n],m,au[n]=gensplit(s[f'{n}_base_scenes'],s[f'{n}_seed'],n,s['image_size'],s['noise_sd'],n=='ood_test');ms.append(m)
 M=pd.concat(ms,ignore_index=True);hs=M.image_sha256.tolist();d=crossdist(sp['train'].x,sp['test'].x);audit={'dataset':'paired_counterfactual_bars_v0.2','dataset_role':'fresh locked mechanism-ground-truth confirmation','generator_version':'paired-bars-v0.2','no_sample_id_overlap':len(set(M.sample_id))==len(M),'no_base_scene_cross_split':bool((M.groupby('base_scene_id').split.nunique()==1).all()),'exact_unique_images_all_splits':len(set(hs)),'exact_duplicate_images_all_splits':len(hs)-len(set(hs)),'test_to_train_nearest_l2':{'min':float(d.min()),'median':float(np.median(d)),'p05':float(np.quantile(d,.05))},'splits':au};return sp,M,audit
# 用途：将划分数组、样本元数据和数据质量记录保存，供后续训练和审计。
# 参数：name、sp、audit、out、meta、idx；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def save_data(name,sp,audit,out,meta=None,idx=None):
 ar={}
 for k,v in sp.items():ar['x_'+k]=v.x;ar['y_'+k]=v.y;ar['ids_'+k]=v.ids
 # 保存具名数值数组，便于后来按同一索引回读和独立复算。
 np.savez_compressed(out/f'{name}.npz',**ar)
 if meta is not None:meta.to_csv(out/f'{name}_metadata.csv',index=False)
 if idx is not None:np.savez_compressed(out/f'{name}_split_indices.npz',**idx)
 # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
 (out/f'{name}_quality_report.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
# 用途：返回该方法的离散状态数；独立源/目标编码用二者状态数的乘积。
# 参数：m、cfg；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：cfg['ilt_source_k'] * cfg['ilt_target_k'] if m == 'ilt' else cfg['joint_k']。
def nstates(m,cfg):return cfg['ilt_source_k']*cfg['ilt_target_k'] if m=='ilt' else cfg['joint_k']
# 用途：用训练数据拟合编码器和代理，用验证数据选择岭回归/树参数并保存覆盖信息。
# 参数：method、ct、cv、tt、tv、cfg、seed、poolmode、trees；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(fit, sel)。
def fit_tv(method,ct,cv,tt,tv,cfg,seed,poolmode,trees):
 e=fitmethod(method,ct,cfg,seed);at,kt=assign(method,ct,e,cfg);av,kv=assign(method,cv,e,cfg);K=nstates(method,cfg);x=pool(at,K,poolmode);xv=pool(av,K,poolmode);r,rs=selridge(x,tt['logits'],xv,tv,cfg['ridge_alphas']);fit={'enc':e,'ridge':r,'pool':poolmode,'K':K};sel={'method':method,'pooling_mode':poolmode,'n_states':K,'feature_dimension':x.shape[1],'train_coverage':cov(kt),'validation_coverage':cov(kv),'ridge_selection':rs}
 if trees:
  mn=max(cfg['tree_min_samples_leaf'],round(len(x)*.02));a,ss=seltree(x,tt['logits'],xv,tv,cfg['tree_depths'],cfg['ridge_alphas'],mn,seed+101);cl,cs=selclf(x,tt['teacher_pred'],xv,tv['teacher_pred'],cfg['classification_tree_depths'],mn,seed+131);fit.update(alt=a,clf=cl);sel.update(affine_leaf_tree_selection=ss,classification_tree_selection=cs)
 return fit,sel
# 用途：将已固定的编码器与代理应用到测试描述量，报告覆盖、保真度和任务准确率。
# 参数：method、c、t、fit、cfg；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：o。
def evaltest(method,c,t,fit,cfg):
 a,k=assign(method,c,fit['enc'],cfg);x=pool(a,fit['K'],fit['pool']);o={'coverage':cov(k),'ridge':metrics(fit['ridge'].predict(x),t)}
 if 'alt' in fit:o['affine_leaf_tree']={**metrics(fit['alt'].predict(x),t),'n_leaves':fit['alt'].leaves};p=fit['clf'].predict(x);o['classification_tree']={'fidelity':float(accuracy_score(t['teacher_pred'],p)),'task_accuracy':float(accuracy_score(t['y'],p)),'n_leaves':int(fit['clf'].get_n_leaves())}
 return o
# 用途：用自动微分构造输入点的局部仿射表达，并在不同扰动尺度下检查区域是否保持。
# 参数：m、s、seed、n、scales、R；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：字典（字段见 return）。
def localaff(m,s,seed,n,scales,R):
 r=np.random.default_rng(seed);ix=r.choice(len(s.y),min(n,len(s.y)),False);errs=[];stab={str(q):[] for q in scales}
 # 用途：把 ReLU 符号与池化赢家拼接为区域签名，判断扰动是否跨越分段边界。
 # 参数：x；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：torch.cat([(t['z1'] > 0).flatten().long(), (t['z2'] > 0).flatten().long(), (t['z3'] > 0…。
 def sig(x):
  t=m.trace(x);_,pi=F.max_pool2d(t['a2'],2,return_indices=True);return torch.cat([(t['z1']>0).flatten().long(),(t['z2']>0).flatten().long(),(t['z3']>0).flatten().long(),pi.flatten().long()])
 for i in ix:
  x=torch.from_numpy(s.x[i:i+1]).clone().requires_grad_(True);y=m(x).detach().numpy()[0];J=[]
  for c in range(len(y)):J.append(torch.autograd.grad(m(x)[0,c],x,retain_graph=True)[0].detach().numpy().ravel())
  J=np.stack(J);xx=x.detach().numpy().ravel();b=y-J@xx;errs.append(float(np.max(np.abs(J@xx+b-y))));sg=sig(x.detach())
  for q in scales:
   same=0
   for _ in range(R):same+=int(torch.equal(sg,sig(torch.clamp(x.detach()+torch.from_numpy(r.normal(0,q,x.shape).astype(np.float32)),0,1))))
   stab[str(q)].append(same/R)
 return {'n_samples':len(ix),'max_reconstruction_error':max(errs),'mean_reconstruction_error':float(np.mean(errs)),'same_region_fraction':{k:float(np.mean(v)) for k,v in stab.items()}}
# 用途：穷举配对差值的正负号组合，计算双侧精确符号翻转 p 值。
# 参数：d；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：sum((v >= obs - 1e-15 for v in vals)) / len(vals)。
def signp(d):
 d=np.asarray(d);obs=abs(d.mean());vals=[abs((d*np.asarray(s)).mean()) for s in itertools.product([-1.,1.],repeat=len(d))];return sum(v>=obs-1e-15 for v in vals)/len(vals)
# 用途：以固定种子对配对差值均值进行 10,000 次 bootstrap，返回 95% 分位区间。
# 参数：d、seed；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)))。
def boots(d,seed):
 d=np.asarray(d);r=np.random.default_rng(seed);m=np.array([r.choice(d,len(d),True).mean() for _ in range(10000)]);return float(np.quantile(m,.025)),float(np.quantile(m,.975))
# 用途：汇总两组配对指标的均值、差值、bootstrap 区间和符号翻转检验。
# 参数：a、b、seed；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：字典（字段见 return）。
def paired(a,b,seed):
 a=np.asarray(a);b=np.asarray(b);d=a-b;lo,hi=boots(d,seed);return {'n_pairs':len(d),'mean_a':float(a.mean()),'sd_a':float(a.std(ddof=1)),'mean_b':float(b.mean()),'sd_b':float(b.std(ddof=1)),'mean_difference_a_minus_b':float(d.mean()),'sd_difference':float(d.std(ddof=1)),'bootstrap_95ci_low':lo,'bootstrap_95ci_high':hi,'exact_two_sided_sign_flip_p':float(signp(d)),'positive_differences':int((d>0).sum()),'zero_differences':int((d==0).sum())}
