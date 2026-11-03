"""预定例子的分段梯度积分：先找第一ReLU根，再找每段第二ReLU根。

每段内尾部仿射，用段中点梯度积分。只用于第一教师案例0的
G0/G1状态和全四组状态，两强度；不据结果再选例子。
"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from layer_transfer_mechanism import tail


def unique_roots(values):
    values=np.sort(np.asarray(values,dtype=np.float64));kept=[0.]
    for v in values:
        if 1e-12<v<1-1e-12 and v-kept[-1]>1e-12:kept.append(float(v))
    kept.append(1.);return np.array(kept)


@torch.no_grad()
def segments(model,a,r):
    """返回全部开关区间边界；根合并容差1e−12，忽略同量级斜率。"""
    b=model.block3;za=b.bn1(b.conv1(a));slope=b.bn1(b.conv1(a+r))-za
    z=za.cpu().numpy().ravel();v=slope.cpu().numpy().ravel();valid=abs(v)>1e-12
    first=unique_roots(-z[valid]/v[valid]);roots=list(first)
    # 第一层开关在每个first区间内固定，第二预激活在该段内为仿射。
    for offset in range(0,len(first)-1,32):
        starts=first[offset:offset+32];ends=first[offset+1:offset+33]
        starts=starts[:len(ends)]
        ts=np.r_[starts,ends];x=a+torch.as_tensor(ts,device=a.device)[:,None,None,None]*r
        z2=b.bn2(b.conv2(F.relu(b.bn1(b.conv1(x)))))+x
        left,right=z2[:len(starts)].flatten(1).cpu().numpy(),z2[len(starts):].flatten(1).cpu().numpy()
        dv=right-left;mask=abs(dv)>1e-12;relative=np.zeros_like(left);relative[mask]=-left[mask]/dv[mask]
        rows,cols=np.where(mask&(relative>1e-12)&(relative<1-1e-12))
        roots.extend(starts[rows]+relative[rows,cols]*(ends-starts)[rows])
    return unique_roots(roots)


@torch.enable_grad()
def integrate(model,a,h,weight):
    """原block3自动微分的完整分段积分、通道贡献和开关事件坐标。"""
    r=h-a;bounds=segments(model,a,r);mid=(bounds[:-1]+bounds[1:])/2;dt=np.diff(bounds)
    w=torch.as_tensor(weight,dtype=a.dtype,device=a.device);channels=[];output=[];events=[];previous=None
    b=model.block3
    for offset in range(0,len(mid),64):
        times=mid[offset:offset+64];leaf=(a+torch.as_tensor(times,device=a.device)[:,None,None,None]*r).detach().requires_grad_(True)
        logits=model.readout(b(leaf).mean((2,3)));g=torch.autograd.grad((logits*w).sum(),leaf)[0]
        channels.extend((g*r).sum((2,3)).detach().cpu().numpy());output.extend((logits@w).detach().cpu().numpy())
        with torch.no_grad():
            z1=b.bn1(b.conv1(leaf));z2=b.bn2(b.conv2(F.relu(z1)))+leaf
            pattern=np.stack(((z1>0).flatten(1).cpu().numpy(),(z2>0).flatten(1).cpu().numpy()),1)
            for row,flags in enumerate(pattern):
                index=offset+row
                if previous is not None:
                    stage,coord=np.where(previous!=flags)
                    for s,c in zip(stage,coord):events.append([bounds[index],int(s),int(c),int(previous[s,c]),int(flags[s,c])])
                previous=flags.copy()
    channels=np.array(channels);output=np.array(output)
    with torch.no_grad():endpoint=(tail(model,torch.cat((a,h)))@w).cpu().numpy()
    integrated=(channels*dt[:,None]).sum(0);error=abs(float(integrated.sum()-(endpoint[1]-endpoint[0])))
    assert error<1e-9,('path_integral',len(mid),error)
    # 每个区间内的线性中点输出也应与端点递推一致。
    derivative=channels.sum(1);predicted=endpoint[0]+np.r_[0,np.cumsum(derivative*dt)[:-1]]+derivative*dt/2
    assert float(np.max(abs(predicted-output)))<1e-9
    return dict(bounds=bounds,midpoint=mid,width=dt,channel_derivative=channels,midpoint_output=output,
        integrated_channel=integrated,endpoints=endpoint,events=np.array(events,dtype=np.float64).reshape(-1,5),error=np.array(error))
