# 中文阅读说明｜用途：把最后残差分支内的激活干预写成局部状态方程，并实现背景近似。
# 数据流：h 是源激活，z 是下游 ReLU 前状态，W/U 是线性映射；残差旁路保持原值。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Local finite interventions inside the last branch of a residual CNN.

The source branch activation changes; the identity path remains the original
value. Fused evaluation-time BatchNorm is affine and permits indexed acquisition.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from residual_teacher import *
from external_experiment import bit_design, low_order, choose_nodes, csv_write, csv_read, score
from context_module import greedy_order

METHODS = ["greedy", "weight", "gate_weight", "activity", "pca_fused"] + [f"active_random_{i}" for i in range(4)]
FRACTIONS = [0.125, 0.25, 0.5]
STRENGTHS = [1., .5]
EXACT_ENGINES = ["exact_indexed", "exact_crop", "exact_fullconv"]


# 用途：从冻结残差块提取受影响状态和线性映射，提供局部精确计算与原尾部对照。
# 结构：成员字段和方法共同保存本阶段需要的状态。
class ResidualLocal:
    # 用途：建立对象所需的网络层、固定映射或配置。从冻结残差块提取受影响状态和线性映射，提供局部精确计算与原尾部对照。
    # 参数：model、shape、nodes；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：初始化对象字段，没有显式返回值。
    def __init__(self, model, shape, nodes):
        self.model=model;self.shape=tuple(shape);self.nodes=np.asarray(nodes,dtype=np.int64)
        self.q=len(nodes);channels,height,width=self.shape
        assert (channels,height,width)==(32,14,14)
        block=model.block3
        assert isinstance(block.shortcut,nn.Identity) and not model.training
        # 评价模式中的 BatchNorm 是固定仿射变换，可与卷积合并而保持函数一致。
        folded=torch.nn.utils.fuse_conv_bn_eval(block.conv2,block.bn2)
        self.kernel=folded.weight.detach().contiguous();self.bias=folded.bias.detach().contiguous()
        basis=torch.zeros((self.q,)+self.shape,dtype=torch.float64)
        basis.flatten(1)[torch.arange(self.q),self.nodes]=1
        with torch.no_grad(): mapping=F.conv2d(basis,self.kernel,None,padding=1).flatten(1)
        # 只保留源坐标确实能够影响的状态；其他输出变化在相减时抵消。
        self.affected=torch.nonzero((mapping!=0).any(0)).flatten()
        self.J=len(self.affected)
        self.W=mapping[:,self.affected].T.contiguous()
        self.channels=self.affected//(height*width)
        self.positions=self.affected%(height*width)
        # 全局平均池化给每个空间位置相同的 1/(H×W) 权重。
        self.U=model.readout.weight.detach()[:,self.channels].contiguous()/(height*width)
        rows=self.positions//width;cols=self.positions%width
        # The fixed central source selection keeps all required patches in bounds.
        assert int(rows.min())>=1 and int(rows.max())<height-1
        assert int(cols.min())>=1 and int(cols.max())<width-1
        cin=torch.arange(channels)[:,None,None]
        dy=torch.arange(-1,2)[None,:,None];dx=torch.arange(-1,2)[None,None,:]
        self.patch_indices=(cin[None]*height*width+(rows[:,None,None,None]+dy[None])*width+
                            cols[:,None,None,None]+dx[None]).reshape(self.J,-1)
        self.patch_weights=self.kernel[self.channels].reshape(self.J,-1).contiguous()
        self.row_bias=self.bias[self.channels].contiguous()
        self.feature_indices,inverse=torch.unique(self.patch_indices.flatten(),sorted=True,return_inverse=True)
        self.feature_count=len(self.feature_indices)
        self.affine=torch.zeros((self.J,self.feature_count+self.J),dtype=torch.float64)
        # 把所需卷积补丁组织成稀疏位置的线性映射，另拼接原旁路贡献。
        self.affine.scatter_(1,inverse.reshape(self.J,-1),self.patch_weights)
        self.affine[:,self.feature_count:]=torch.eye(self.J,dtype=torch.float64)
        self.crop=(int(rows.min())-1,int(rows.max())+2,int(cols.min())-1,int(cols.max())+2)
        crop_height=self.crop[1]-self.crop[0]-2;crop_width=self.crop[3]-self.crop[2]-2
        self.crop_indices=self.channels*(crop_height*crop_width)+(rows-(self.crop[0]+1))*crop_width+cols-(self.crop[2]+1)

    # 用途：按所选获取后端取得 h 与 z；三种实现应得到同一个原始上下文。
    # 参数：source、residual、engine；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：(h, z)。
    @torch.no_grad()
    def states(self, source, residual, engine="exact_crop"):
        h=source.flatten()[self.nodes]
        if engine=="exact_indexed":
            z=(source.flatten()[self.patch_indices]*self.patch_weights).sum(1)+self.row_bias+residual.flatten()[self.affected]
        elif engine=="exact_fullconv":
            z=(F.conv2d(source,self.kernel,self.bias,padding=1)+residual).flatten()[self.affected]
        elif engine=="exact_crop":
            r0,r1,c0,c1=self.crop
            conv=F.conv2d(source[:,:,r0:r1,c0:c1].contiguous(),self.kernel,self.bias)
            z=conv.flatten()[self.crop_indices]+residual.flatten()[self.affected]
        else:raise ValueError(engine)
        return h,z

    # 用途：拼接所需卷积输入和旁路状态，供融合仿射获取使用。
    # 参数：source、residual；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：torch.cat([source.flatten()[self.feature_indices], residual.flatten()[self.affected]])。
    @torch.no_grad()
    def vector(self,source,residual):
        return torch.cat([source.flatten()[self.feature_indices],residual.flatten()[self.affected]])

    # 用途：按完整受影响状态计算真实局部响应。
    # 输入：context=(h,z)，bits[组合,q]，alpha 为源激活衰减强度。
    # 输出：返回 [组合,10] 输出差：ReLU 状态差右乘 U 的转置。
    @torch.no_grad()
    def exact(self,context,bits,alpha=1.):
        h,z=context;t=(-alpha*bits*h[None])@self.W.T
        return (F.relu(z[None]+t)-F.relu(z)[None])@self.U.T

    # 用途：直接修改源特征图并调用原教师 tail，作为局部方程的独立对照。
    # 参数：source、residual、bits、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：self.model.tail(changed, residual) - self.model.tail(source, residual)。
    @torch.no_grad()
    def full(self,source,residual,bits,alpha=1.):
        changed=source.expand(len(bits),-1,-1,-1).clone()
        changed.flatten(1)[:,self.nodes]*=(1-alpha*bits)
        return self.model.tail(changed,residual)-self.model.tail(source,residual)


# 用途：保存训练上下文得到的门控频率、PCA 分量及各选择基线的顺序。
# 结构：成员字段和方法共同保存本阶段需要的状态。
@dataclass
class ResidualFit:
    gate:np.ndarray
    mean_z:np.ndarray
    components:np.ndarray
    greedy:np.ndarray
    gains:np.ndarray
    weight:np.ndarray
    gate_weight:np.ndarray
    activity:np.ndarray
    active_random_0:np.ndarray
    active_random_1:np.ndarray
    active_random_2:np.ndarray
    active_random_3:np.ndarray


# 用途：利用训练上下文计算平均门控、残差贪心排序、PCA 和活动/权重基线。
# 参数：local、contexts、seed；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(ResidualFit(**values), dict(training_active_units=int((gate > 0).sum()), always_open_t…。
def fit_context(local, contexts, seed):
    H=torch.stack([c[0] for c in contexts]).numpy();Z=torch.stack([c[1] for c in contexts]).numpy()
    bits=bit_design(8)[low_order(8)[0]]
    # 门控频率仅由训练上下文估计；不是每次干预重新观察的真实开关。
    gate=(Z>0).mean(0);delta=(-H[:,None]*bits[None]).reshape(-1,8)
    T=delta@local.W.numpy().T;ZZ=np.repeat(Z,len(bits),axis=0)
    residual=np.maximum(ZZ+T,0)-np.maximum(ZZ,0)-T*gate
    order,gains=greedy_order(residual,local.U.numpy())
    # PCA 压缩的是中心化的预激活背景；响应阶段仍保留非线性。
    # 用奇异值分解提取低维方向；中心化对象和保留维数由本段数据与预算确定。
    _,singular,components=np.linalg.svd(Z-Z.mean(0),full_matrices=False)
    norm=np.linalg.norm(local.W.numpy(),axis=1)*np.linalg.norm(local.U.numpy(),axis=0)
    values=dict(gate=gate,mean_z=Z.mean(0),components=components,greedy=order,gains=gains,
        weight=np.argsort(-norm,kind="stable"),gate_weight=np.argsort(-(gate*(1-gate)*norm),kind="stable"),
        activity=np.argsort(-gate,kind="stable"))
    for i in range(4):
        rng=np.random.default_rng(2026091300+seed*10+i)
        values[f"active_random_{i}"]=np.concatenate([rng.permutation(np.flatnonzero(gate>0)),rng.permutation(np.flatnonzero(gate==0))])
    return ResidualFit(**values),dict(training_active_units=int((gate>0).sum()),always_open_training_units=int((gate==1).sum()),
        changing_training_gates=int(((gate>0)&(gate<1)).sum()),singular_values=singular.tolist())


# 用途：回读保存的背景统计、排序或 PCA 参数，恢复拟合结果对象。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(ResidualFit(**{name: saved[name].copy() for name in ResidualFit.__annotations__}), sav…。
def load_fit(path):
    with np.load(path,allow_pickle=False) as saved:
        return ResidualFit(**{name:saved[name].copy() for name in ResidualFit.__annotations__}),saved["nodes"].copy()


# 用途：列出本阶段各方法、预算比例和实际保留状态数。
# 参数：local；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：[(method, float(fraction), math.ceil(local.J * fraction)) for method in METHODS for fra…。
def specs(local):
    return [(method,float(fraction),math.ceil(local.J*fraction)) for method in METHODS for fraction in FRACTIONS]+[
        ("mean_gate",0.,0),("original_region",1.,local.J),("exact",1.,local.J)]


# 用途：生成包含方法、预算比例和干预强度的结果键。
# 参数：method、fraction、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：f'{method}_f{fraction}_a{alpha}'。
def key(method,fraction,alpha):
    return f"{method}_f{fraction}_a{alpha}"


# 用途：按给定预算选择/压缩残差状态，并计算对应近似或精确响应。
# 结构：成员字段和方法共同保存本阶段需要的状态。
class ContextRuntime:
    # 用途：建立对象所需的网络层、固定映射或配置。按给定预算选择/压缩残差状态，并计算对应近似或精确响应。
    # 参数：local、fit、method、k、engine；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：初始化对象字段，没有显式返回值。
    def __init__(self,local,fit,method,k,engine="exact_crop"):
        self.local=local;self.method=method;self.k=k;self.engine=engine
        self.idx=torch.arange(local.J,dtype=torch.long)
        self.W=local.W;self.U=local.U
        self.linear=None
        if method in METHODS and method!="pca_fused" or method=="mean_gate":
            order=fit.greedy if method=="mean_gate" else getattr(fit,method)
            self.idx=torch.tensor(order[:k].copy(),dtype=torch.long)
            omitted=torch.ones(local.J,dtype=torch.bool);omitted[self.idx]=False
            gate=torch.tensor(fit.gate,dtype=torch.float64)
            self.linear=((local.U[:,omitted]*gate[omitted])@local.W[omitted]).contiguous()
            self.W=local.W[self.idx].contiguous();self.U=local.U[:,self.idx].contiguous()
            self.patch_indices=local.patch_indices[self.idx].contiguous()
            self.patch_weights=local.patch_weights[self.idx].contiguous()
            self.state_ids=local.affected[self.idx].contiguous();self.bias=local.row_bias[self.idx].contiguous()
        elif method=="pca_fused":
            self.mean=torch.tensor(fit.mean_z,dtype=torch.float64)
            self.V=torch.tensor(fit.components[:k].copy(),dtype=torch.float64)
            assert len(self.V)==k
            self.encoder=(self.V@local.affine).contiguous()
            self.encoder_bias=(self.V@(local.row_bias-self.mean)).contiguous()
        elif method not in ("exact","original_region"):
            raise ValueError(method)

    # 用途：从已获取的完整 h/z 中选出方法需要的状态或 PCA 编码。
    # 参数：context；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：(h, (z - self.mean) @ self.V.T) / (h, z[self.idx])。
    @torch.no_grad()
    def cached(self,context):
        h,z=context
        if self.method=="pca_fused":return h,(z-self.mean)@self.V.T
        return h,z[self.idx]

    # 用途：从源特征图与旁路获取该方法需要的状态，计入准备成本。
    # 参数：source、residual；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：self.local.states(source, residual, self.engine) / (h, z)。
    @torch.no_grad()
    def prepare(self,source,residual):
        h=source.flatten()[self.local.nodes]
        if self.method=="pca_fused":
            z=self.encoder@self.local.vector(source,residual)+self.encoder_bias
        elif self.linear is not None:
            z=(source.flatten()[self.patch_indices]*self.patch_weights).sum(1)+self.bias+residual.flatten()[self.state_ids]
        else:return self.local.states(source,residual,self.engine)
        return h,z

    # 用途：把源坐标衰减映射到保留状态，计算非线性响应并加上省略状态的线性项。
    # 参数：context、bits、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：t * (z > 0)[None] @ self.U.T / nonlinear if self.linear is None else nonlinear + delta @ self.linear.T。
    @torch.no_grad()
    def evaluate(self,context,bits,alpha=1.):
        h,z=context;delta=-alpha*bits*h[None]
        if self.method=="pca_fused":z=self.mean+z@self.V
        t=delta@self.W.T
        if self.method=="original_region":return (t*(z>0)[None])@self.U.T
        nonlinear=(F.relu(z[None]+t)-F.relu(z)[None])@self.U.T
        return nonlinear if self.linear is None else nonlinear+delta@self.linear.T

    # 用途：将输入上下文分批并行计算，输出 [输入,组合,类别]，与逐输入路径一致。
    # 参数：contexts、bits、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：torch.cat(outputs)。
    @torch.no_grad()
    def evaluate_many(self,contexts,bits,alpha=1.):
        outputs=[]
        for begin in range(0,len(contexts),16):
            group=contexts[begin:begin+16]
            h=torch.stack([c[0] for c in group]);z=torch.stack([c[1] for c in group])
            if self.method=="pca_fused":z=self.mean+z@self.V
            delta=(-alpha*h[:,None]*bits[None]).reshape(-1,8)
            if self.W.shape[0]==0:
                value=delta@self.linear.T
            else:
                t=(delta@self.W.T).reshape(len(group),len(bits),-1)
                response=t*(z>0)[:,None] if self.method=="original_region" else F.relu(z[:,None]+t)-F.relu(z)[:,None]
                value=response.reshape(-1,self.W.shape[0])@self.U.T
                if self.linear is not None:value+=delta@self.linear.T
            outputs.append(value.reshape(len(group),len(bits),10))
        return torch.cat(outputs)

    # 用途：记录实际保留矩阵与缓存规模，用于分开比较状态获取和查询成本。
    # 参数：无额外参数；读取或初始化对象状态。
    # 返回：字典（字段见 return）。
    def costs(self):
        arrays=[self.W,self.U,self.idx]
        if self.linear is not None:
            arrays.extend([self.linear,self.patch_indices,self.patch_weights,self.state_ids,self.bias])
        elif self.method=="pca_fused":
            arrays.extend([self.mean,self.V,self.encoder,self.encoder_bias,self.local.feature_indices,self.local.affected])
        else:
            arrays.extend([self.local.kernel,self.local.bias,self.local.affected,self.local.patch_indices,self.local.patch_weights,self.local.row_bias,self.local.crop_indices])
        unique={t.data_ptr():t for t in arrays}
        retained=[v for owner in (self,self.local) for v in vars(owner).values() if isinstance(v,torch.Tensor)]
        storage={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in retained}
        return dict(source_scalars=8,context_scalars=self.k,affected_states=self.local.J,
            runtime_array_bytes=sum(t.numel()*t.element_size() for t in unique.values()),
            retained_non_teacher_tensor_bytes=sum(storage.values()),
            teacher_state_bytes=sum(t.numel()*t.element_size() for t in self.local.model.state_dict().values()),
            scope="runtime_array_bytes counts active execution arrays; retained_non_teacher_tensor_bytes also includes shared construction maps. Neither is peak process memory; NumPy indices/Python overhead and query temporaries excluded",
            feature_prefix="complete last-branch source and identity prefix computed for every from-image method")


# 用途：计算当前设置的预测误差并附上教师、方法、预算等标签；分母规则见函数体。
# 参数：prediction、reference、seed、method、fraction、k、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：字典（字段见 return）。
def metrics(prediction,reference,seed,method,fraction,k,alpha):
    held=bit_design(8).sum(1)>2
    result=score(prediction,reference,held)
    den=np.square(reference[:,held]).sum((1,2));num=np.square((prediction-reference)[:,held]).sum((1,2))
    usable=den>1e-24;per=np.sqrt(num[usable]/den[usable])
    return dict(seed=seed,method=method,fraction=fraction,k=k,alpha=alpha,**result,
        inputs=len(prediction),scored_coalitions=219,zero_effect_inputs=int((~usable).sum()),
        input_p95=float(np.quantile(per,.95)) if len(per) else 0.,
        input_fraction_over_10pct=float((per>.1).mean()) if len(per) else 0.)
