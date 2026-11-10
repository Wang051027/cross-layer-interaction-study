"""Phase VI. Exact context-conditioned local compilation and conservative interaction groups.
The compiler reuses weights, not learned semantic mechanisms. All numerical proofs are
real-arithmetic identities tested in float64; interval bounds are not directed-rounded.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn.functional as F


# 按整数的二进制位枚举全部 2**q 种干预掩码；低位对应低编号源坐标。
def bit_design(q:int):
    return ((np.arange(1<<q)[:,None]>>np.arange(q))&1).astype(np.float64)


# 列出空、单坐标和双坐标组合的整数索引，并同时返回坐标对。
def low_order(q:int):
    masks=[1<<i for i in range(q)]
    pairs=[(i,j) for i in range(q) for j in range(i+1,q)]
    return np.array([0]+masks+[(1<<i)|(1<<j) for i,j in pairs]),pairs


# 把掩码的一阶位值与所有两两乘积拼成二阶多项式设计矩阵。
def pair_features(bits):
    q=bits.shape[1]
    return np.column_stack([bits]+[(bits[:,i]*bits[:,j])[:,None] for i in range(q) for j in range(i+1,q)])


# 从按低阶索引排列的响应计算一阶和二阶有限差分系数。
def pair_coeff(values,q):
    # values ordered as low_order(q)[0]; supports full logit vectors
    single=values[1:1+q]-values[0]
    pairs=low_order(q)[1]
    pp=np.stack([values[1+q+k]-values[0]-single[i]-single[j] for k,(i,j) in enumerate(pairs)])
    return np.concatenate([single,pp],axis=0)


# 从指定中间层继续执行原网络，返回边界、头部状态及最终 logits。
def trace_from_source(m,source,layer):
    if layer==1:
        z2=m.conv2(source); a2=F.relu(z2)
    elif layer==2:
        z2=None; a2=source
    else:raise ValueError('layer must be 1 or 2')
    pool=F.max_pool2d(a2,2)
    z3=m.fc1(pool.flatten(1))
    logits=m.fc2(F.relu(z3))
    return dict(z2=z2,a2=a2,pool=pool,z3=z3,logits=logits)


# 执行冻结模型的前缀，取得第一或第二层的源激活。
def source_activation(m,x,layer):
    h=F.relu(m.conv1(x))
    return h if layer==1 else F.relu(m.conv2(h))


class LocalModule:
    """Compile the exact affected cone at fixed source coordinates for one frozen model.
    Contexts differ across inputs. No intervened target values are used for compilation.
    Includes all competitors in affected MaxPool windows and the complete nonlinear head.
    """
    def __init__(self,m,shape,nodes,layer=1):
        t=time.perf_counter()
        self.model=m; self.shape=tuple(shape); self.nodes=np.array(nodes,dtype=np.int64)
        self.q=len(nodes); self.layer=layer
        basis=torch.zeros((self.q,)+self.shape,dtype=torch.float64)
        for j,n in enumerate(nodes):basis.reshape(self.q,-1)[j,int(n)]=1.
        with torch.no_grad():
            if layer==1:
                c=m.conv2
                xx=F.pad(basis,c._reversed_padding_repeated_twice,mode=c.padding_mode) if c.padding_mode!='zeros' else basis
                A=F.conv2d(xx,c.weight,None,c.stride,0 if c.padding_mode!='zeros' else c.padding,c.dilation,c.groups)
            else:A=basis
            self.target_shape=tuple(A.shape[1:]);C,H,W=self.target_shape
            changed=torch.nonzero((A!=0).any(0).flatten()).flatten().numpy()
            pidx=(changed//(H*W))*(H//2)*(W//2)+((changed%(H*W))//W//2)*(W//2)+(changed%W//2)
            self.pool_ids=np.unique(pidx)
            win=[]
            for p in self.pool_ids:
                c=int(p)//((H//2)*(W//2)); y=(int(p)%((H//2)*(W//2)))//(W//2)*2; x=int(p)%(W//2)*2
                win.append([c*H*W+(y+dy)*W+x+dx for dy,dx in [(0,0),(0,1),(1,0),(1,1)]])
            # 保留每个受影响最大池化窗口的全部四个竞争位置。
            self.window_ids=np.asarray(win,dtype=np.int64)
            self.A=A.flatten(1)[:,self.window_ids.reshape(-1)].contiguous().detach()
            self.W=m.fc1.weight[:,self.pool_ids].detach().clone()
            self.U=m.fc2.weight.detach().clone()
            self.wp=(self.W!=0).numpy();self.u_live=(self.U!=0).any(0).numpy()
        self.compile_seconds=time.perf_counter()-t
        self.numeric_bytes=int(sum(v.numel()*v.element_size() for v in (self.A,self.W,self.U))+self.nodes.nbytes+self.window_ids.nbytes+self.pool_ids.nbytes)

    # 执行原尾部一次，缓存源值、完整竞争窗口、头部预激活和原输出。
    @torch.no_grad()
    def prepare(self,source):
        t=trace_from_source(self.model,source,self.layer)
        pre=t['z2'] if self.layer==1 else t['a2']
        window=pre.flatten()[self.window_ids]
        # Keep only needed original-state values. No intervened pass here.
        return {'source':source,'h':source.flatten()[self.nodes], 'window':window,
                'pool':t['pool'].flatten()[self.pool_ids], 'z3':t['z3'][0],
                'logits':t['logits'][0]}

    # 只更新受影响窗口及其下游头部，利用未改变部分相消计算精确输出差。
    @torch.no_grad()
    def evaluate(self,ctx,bits,alpha=1.0):
        if not isinstance(bits,torch.Tensor):bits=torch.as_tensor(bits,dtype=torch.float64)
        delta=-alpha*bits*ctx['h'][None,:]
        pre=ctx['window'][None,:,:]+(delta@self.A).reshape(len(bits),len(self.pool_ids),4)
        act=F.relu(pre) if self.layer==1 else pre
        dp=act.amax(2)-ctx['pool'][None,:]
        # 将池化变化传入分类头，再计算 ReLU。
        z3new=ctx['z3'][None,:]+dp@self.W.T
        return (F.relu(z3new)-F.relu(ctx['z3'])[None,:])@self.U.T

    # 逐组合修改原源特征图并执行完整尾部，作为局部编译的参考。
    @torch.no_grad()
    def full_evaluate(self,ctx,bits,alpha=1.0):
        if not isinstance(bits,torch.Tensor):bits=torch.as_tensor(bits,dtype=torch.float64)
        h=ctx['source'].expand(len(bits),-1,-1,-1).clone()
        h.flatten(1)[:,self.nodes]*=(1-alpha*bits)
        return trace_from_source(self.model,h,self.layer)['logits']-ctx['logits']

    # 固定原 ReLU 符号和池化赢家，计算每个源坐标的原区域响应斜率。
    @torch.no_grad()
    def reference_slopes(self,ctx):
        dz=(-ctx['h'][:,None]*self.A).reshape(self.q,len(self.pool_ids),4)
        if self.layer==1:dz=dz*(ctx['window']>0)[None,:,:]
        a=F.relu(ctx['window']) if self.layer==1 else ctx['window']
        ix=a.argmax(1)
        dp=dz.gather(2,ix[None,:,None].expand(self.q,-1,1)).squeeze(-1)
        return ((dp@self.W.T)*(ctx['z3']>0)[None,:])@self.U.T

    # 构造源局部、边界竞争和下游状态/敏感性三组描述量。
    @torch.no_grad()
    def descriptors(self,ctx):
        # Three blocks: local source values; boundary context; downstream routing/sensitivity.
        C,H,W=self.shape;ys=self.nodes%(H*W)//W;xs=self.nodes%W
        rr,cc=int(np.median(ys)),int(np.median(xs))
        local=ctx['source'][0,:,max(0,rr-1):min(H,rr+2),max(0,cc-1):min(W,cc+2)].flatten().numpy()
        a=F.relu(ctx['window']) if self.layer==1 else ctx['window']
        gaps=a.amax(1)[:,None]-a
        boundary=np.concatenate([ctx['window'].flatten().numpy(),gaps.flatten().numpy()])
        slopes=self.reference_slopes(ctx).numpy()
        downstream=np.concatenate([ctx['z3'].numpy(),ctx['logits'].numpy(),slopes.ravel()])
        return local,boundary,downstream,slopes

    def blocks(self,ctx,alpha):
        """Conservative interaction supergraph over the entire attenuation box.
        An uncertified nonlinear branch joins all its source dependencies. Fixed affine
        sums do not join blocks. This may over-merge and does not prove irreducibility.
        """
        start=time.perf_counter();q=self.q;bit=np.left_shift(np.uint64(1),np.arange(q,dtype=np.uint64))
        A=self.A.numpy();h=ctx['h'].numpy();D=-alpha*h[:,None]*A
        z=ctx['window'].numpy().ravel();dep=((D.T!=0).astype(np.uint64)*bit).sum(1,dtype=np.uint64)
        lo=z+np.minimum(D,0).sum(0);hi=z+np.maximum(D,0).sum(0)
        slack=1e-11*(1+np.maximum(abs(lo),abs(hi)))
        # 并查集记录保守交互分组；共享未确定非线性的源坐标会被合并。
        parent=list(range(q))
        # 查找并查集根节点并压缩路径，用于维护源坐标分组。
        def find(x):
            while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
            return x
        # 将依赖位掩码中涉及的源坐标合并到同一组。
        def join(mask):
            inds=[i for i in range(q) if int(mask)&(1<<i)]
            if len(inds)>1:
                a=find(inds[0])
                for i in inds[1:]:parent[find(i)]=a
        # 回收当前并查集分组，返回每组的源坐标索引列表。
        def sizes():
            gs={}
            for i in range(q):gs.setdefault(find(i),[]).append(i)
            return list(gs.values())
        unstable_conv=0
        if self.layer==1:
            off=(hi<=0);on=(lo>=slack)
            uncertain=(~off)&(~on)&(dep!=0)
            for d in np.unique(dep[uncertain]):join(d)
            unstable_conv=int(uncertain.sum());dep[off]=0
            lo=np.maximum(lo,0);hi=np.maximum(hi,0)
        conv_sizes=[len(g) for g in sizes()]
        lo=lo.reshape(-1,4);hi=hi.reshape(-1,4);dep=dep.reshape(-1,4)
        pl=[];pu=[];pd=[];unstable_pool=0
        for l,u,d in zip(lo,hi,dep):
            # Sound candidate superset; fixed winners include exact ties at zero.
            # 保留所有可能成为池化赢家的位置；只有唯一候选时才固定赢家。
            candidate=u>=l.max()-1e-11*(1+abs(l.max()))
            candidates=np.flatnonzero(candidate)
            if len(candidates)==1:
                j=candidates[0];dm=d[j]
            else:
                dm=np.bitwise_or.reduce(d[candidate]);unstable_pool+=int(dm!=0);join(dm)
            pl.append(float(l.max()));pu.append(float(u.max()));pd.append(dm)
        pool_sizes=[len(g) for g in sizes()]
        pd=np.array(pd,dtype=np.uint64);p0=ctx['pool'].numpy();l=np.array(pl)-p0;u=np.array(pu)-p0
        W=self.W.numpy();z3=ctx['z3'].numpy();pos=np.maximum(W,0);neg=np.minimum(W,0)
        l3=z3+pos@l+neg@u;u3=z3+pos@u+neg@l
        memberships=(pd[:,None]&bit[None,:])!=0
        memberships=(self.wp.astype(np.int32)@memberships.astype(np.int32))>0
        d3=(memberships.astype(np.uint64)*bit).sum(1,dtype=np.uint64)
        eps=1e-11*(1+np.maximum(abs(l3),abs(u3)))
        # 只有可能跨越头部 ReLU、依赖源坐标且能影响输出的状态才需要合并依赖。
        uncertain3=(u3>0)&(l3<eps)&(d3!=0)&self.u_live
        for d in np.unique(d3[uncertain3]):join(d)
        groups=sizes();active=int(np.sum(h!=0));estimated=1+sum((1<<len(g))-1 for g in groups)
        return {'groups':groups,'largest_block':max(map(len,groups)),'n_blocks':len(groups),
                'largest_after_conv':max(conv_sizes),'largest_after_pool':max(pool_sizes),
                'uncertain_conv':unstable_conv,'uncertain_pool':unstable_pool,'uncertain_hidden':int(uncertain3.sum()),
                'active_sources':active,'block_queries':estimated,'full_queries':1<<q,
                'certified_query_fraction':estimated/(1<<q),'analysis_seconds':time.perf_counter()-start}


# 按保守分组取出组内响应并相加，重建所有组合的输出变化。
def block_reconstruction(values,groups):
    out=np.broadcast_to(values[0],values.shape).copy()
    for g in groups:
        mask=sum(1<<i for i in g)
        out+=values[np.arange(len(values))&mask]-values[0]
    return out
