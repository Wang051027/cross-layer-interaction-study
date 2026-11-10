"""Restricted background selection. Original weights are retained; train-only masks
select exact nonlinear head responses. Omitted responses use a train mean-gate map.
No claim of a novel pruning/interpretability primitive or semantic mechanism.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

# 原上下文方法的拟合状态容器，包含平均门控、排序、PCA 和回归系数。
@dataclass
class Fit:
 gate: np.ndarray
 order: np.ndarray
 mean_window: np.ndarray
 mean_z: np.ndarray
 pca_components: np.ndarray
 random_order: np.ndarray
 weight_order: np.ndarray
 ridge: np.ndarray
 train_gain: np.ndarray

# 从源坐标衰减计算受影响池化窗口的输出差，保持窗口竞争项完整。
def get_dp(mod,ctx,bits,alpha=1.,window=None):
 h=ctx['h'].numpy();z=ctx['window'].numpy() if window is None else window
 zz=z[None]+(-alpha*bits*h[None]@mod.A.numpy()).reshape(len(bits),-1,4)
 a=np.maximum(zz,0) if mod.layer==1 else zz
 a0=np.maximum(z,0) if mod.layer==1 else z
 return a.max(2)-a0.max(1)[None]

# 使用 NumPy 计算原局部非线性头部的精确输出差，作为同公式对照。
def exact_numpy(mod,ctx,bits,alpha=1.):
 dp=get_dp(mod,ctx,bits,alpha);z=ctx['z3'].numpy()
 return (np.maximum(z+dp@mod.W.numpy().T,0)-np.maximum(z,0))@mod.U.numpy().T

# 用非线性余项与输出权重的 Gram 矩阵逐步选择状态，考虑通道间相消。
def greedy_order(R,U):
 # Residual component Gram matrix accounts for cancellation across output channels.
 # 残差 Gram 与输出权重 Gram 逐项相乘，把不同状态间的输出相消纳入选择。
 gram=(R.T@R)*(U.T@U);remain=list(range(len(gram)));order=[];gains=[]
 while remain:
  # 计算从当前剩余集合中精确保留一个状态可减少多少残差平方。
  ii=np.asarray(remain);g=2*gram[np.ix_(ii,ii)].sum(1)-np.diag(gram)[ii]
  pos=int(np.argmax(g));j=remain.pop(pos);order.append(j);gains.append(float(g[pos]))
 return np.asarray(order),np.asarray(gains)

# 只用训练上下文拟合平均门控、贪心顺序、PCA 和池化岭回归等基线。
def fit_module(mod,contexts,bits,seed):
 U=mod.U.numpy();W=mod.W.numpy();Z=np.stack([c['z3'].numpy() for c in contexts]);gate=(Z>0).mean(0)
 dp=np.concatenate([get_dp(mod,c,bits) for c in contexts]);zz=np.repeat(Z,len(bits),axis=0);t=dp@W.T
 exact_hidden=np.maximum(zz+t,0)-np.maximum(zz,0);R=exact_hidden-t*gate
 # 用奇异值分解提取低维方向；中心化对象和保留维数由本段数据与预算确定。
 order,gain=greedy_order(R,U);_,_,vh=np.linalg.svd(Z-Z.mean(0),full_matrices=False)
 # 给池化响应的线性回归加入小正则，改善病态设计矩阵的求解。
 lam=max(1e-8,1e-6*float(np.trace(dp.T@dp))/max(1,dp.shape[1]))
 ridge=np.linalg.solve(dp.T@dp+lam*np.eye(dp.shape[1]),dp.T@(exact_hidden@U.T))
 return Fit(gate,order,np.stack([c['window'].numpy() for c in contexts]).mean(0),Z.mean(0),vh,
  np.random.default_rng(seed+1701).permutation(len(gate)),
  np.argsort(-(np.linalg.norm(W,axis=1)*np.linalg.norm(U,axis=0))),ridge,gain)

# 选中的头部状态精确计算非线性，其余状态合并为训练平均门控的线性响应。
def selected(mod,fit,ctx,bits,k,alpha=1.,order=None):
 idx=fit.order[:k] if order is None else np.asarray(order)[:k];U=mod.U.numpy();W=mod.W.numpy()
 mask=np.ones(len(fit.gate),bool);mask[idx]=False;linear=(U[:,mask]*fit.gate[mask])@W[mask]
 dp=get_dp(mod,ctx,bits,alpha);pred=dp@linear.T
 if len(idx):
  z=ctx['z3'].numpy()[idx];t=dp@W[idx].T;pred+=(np.maximum(z+t,0)-np.maximum(z,0))@U[:,idx].T
 return pred

# 按方法名分派到状态选择、岭回归、平均背景、PCA 或完整精确响应。
def predict(mod,fit,ctx,bits,method,k=0,alpha=1.):
 if method in ('greedy','random','weight'):
  order=fit.order if method=='greedy' else fit.random_order if method=='random' else fit.weight_order
  return selected(mod,fit,ctx,bits,k,alpha,order)
 dp=get_dp(mod,ctx,bits,alpha,fit.mean_window if method=='local_only' else None)
 if method=='pool_ridge':return dp@fit.ridge
 if method in ('local_only','boundary_only'):return dp@((mod.U.numpy()*fit.gate)@mod.W.numpy()).T
 if method in ('mean_head','pca'):
  z=fit.mean_z
  if method=='pca' and k:
   V=fit.pca_components[:k];z=z+(ctx['z3'].numpy()-z)@V.T@V
  return (np.maximum(z+dp@mod.W.numpy().T,0)-np.maximum(z,0))@mod.U.numpy().T
 if method=='exact':return exact_numpy(mod,ctx,bits,alpha)
 raise ValueError(method)

class RuntimeModule:
 """prepare does not compute unselected original head preactivations.
 Boundary conv/pool remain exact. Source prefix is common and excluded in timing.
 """
 def __init__(self,mod,fit,k,order=None):
  self.mod=mod;self.idx=np.asarray(fit.order if order is None else order)[:k]
  self.A=mod.A.numpy();self.W=mod.W.numpy()[self.idx].copy();self.U=mod.U.numpy()[:,self.idx].copy()
  mask=np.ones(len(fit.gate),bool);mask[self.idx]=False
  self.linear=(mod.U.numpy()[:,mask]*fit.gate[mask])@mod.W.numpy()[mask]
  self.headW=mod.model.fc1.weight.detach().numpy()[self.idx].copy();self.headb=mod.model.fc1.bias.detach().numpy()[self.idx].copy()
  self.bytes=sum(v.nbytes for v in (self.A,self.W,self.U,self.linear,self.headW,self.headb,self.idx))
 # 计算真实卷积/池化边界，只获取选中头部单元的预激活。
 @torch.no_grad()
 def prepare(self,source):
  if self.mod.layer==1:pre=self.mod.model.conv2(source);a=F.relu(pre)
  else:pre=source;a=source
  p=F.max_pool2d(a,2).flatten().numpy();window=pre.flatten()[self.mod.window_ids].numpy()
  z=self.headW@p+self.headb
  return source.flatten()[self.mod.nodes].numpy(),window,z
 # 将精确选中状态的响应与省略状态的平均门控线性响应相加。
 def evaluate(self,ctx,bits,alpha=1.):
  h,window,z=ctx;zz=window[None]+(-alpha*bits*h[None]@self.A).reshape(len(bits),-1,4)
  a=np.maximum(zz,0) if self.mod.layer==1 else zz;a0=np.maximum(window,0) if self.mod.layer==1 else window
  dp=a.max(2)-a0.max(1)[None];out=dp@self.linear.T
  if len(self.idx):out+=(np.maximum(z+dp@self.W.T,0)-np.maximum(z,0))@self.U.T
  return out
