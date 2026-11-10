"""分别替换原通道组 C 和组外通道，计算四个中间状态及其输出差。"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F

ENDPOINTS = tuple(range(25, 37))+(1, 4, 7, 10)
STATES = ('00_before', '10_C_only', '01_outside_only', '11_after')
MODES = ('natural', 'j_at_before', 'j_at_after')
TERMS = ('C_before', 'outside_before', 'interaction', 'total', 'C_after', 'outside_after')
# 对四状态值施加相同线性对比；输出本身仍由非线性网络真实计算。
CONTRAST = np.array([[-1,1,0,0], [-1,0,1,0], [1,-1,-1,1],
                     [-1,0,0,1], [0,0,-1,1], [0,-1,0,1]], dtype=float)
DROP_SLOTS = (0, 3, 6, 9)  # aligned、定向四族；非成功解也保留。


@torch.no_grad()
def endpoints(model, half, full, mask, specs):
    """仅重算目标源状态的实际H；与旧39端点整图读出对照，禁止重选强度。"""
    t = 5 if mask == 3 else 6
    h0 = full['h'][t]; values = []
    for family, ids, amount in specs:
        if family < 0 or amount == 0:
            h = h0.clone()
        elif family == 0:
            x = full['x'][t:t+1].clone()
            x[:, ids] += amount*(half['x'][t:t+1, ids]-x[:, ids])
            h = model.block2(x)[0]
        elif family == 2:
            u = full['u'][t:t+1].clone()
            u.flatten(1)[:, ids] += amount*(half['u'][t:t+1].flatten(1)[:, ids]-u.flatten(1)[:, ids])
            h = model.block2.finish(u, full['skip'][t:t+1])[0]
        else:
            h = h0.clone()
            if family == 1: h[ids] += amount*(half['h'][t, ids]-h[ids])
            else: h.flatten()[ids] += amount*(half['h'][t].flatten()[ids]-h.flatten()[ids])
        values.append(h)
    result = torch.stack(values)
    assert float(result.min()) >= 0.
    return h0, result


def four_fields(before, after, channels):
    """交换整张C通道图，组外值由另一状态保留；四场均非负。"""
    original = before.expand_as(after)
    c_only = original.clone(); c_only[:, channels] = after[:, channels]
    o_only = after.clone(); o_only[:, channels] = original[:, channels]
    return torch.stack((original, c_only, o_only, after), 1)


@torch.no_grad()
def measure(model, before, after, channels, point, weight):
    """输出三类logit、六种对比的通道/空间抵消及两层ReLU精确余项。

非线性余项相对H00的一阶门控展开是恒等分解，并非唯一因果份额。
对j固定非负值是独立的计算干预；其余路径和第二ReLU保持真实响应。
"""
    fields = four_fields(before, after, channels); n = len(after)
    x = fields.flatten(0, 1); b = model.block3
    z = b.bn1(b.conv1(x)); u = F.relu(z)
    v = b.bn2(b.conv2(u))+x; e = F.relu(v)
    w = torch.as_tensor(weight, device=x.device, dtype=x.dtype)
    q = (w@model.readout.weight)[None, :, None, None]/1024
    matrix = torch.as_tensor(CONTRAST, device=x.device, dtype=x.dtype)
    z4, u4, v4, e4 = [a.reshape(n, 4, 32, 32, 32) for a in (z,u,v,e)]
    logits = [model.readout(e.mean((2,3))).reshape(n,4,3)]
    all_e = [e4]; second_flips = []
    # 固定值来自该端点00或11的原生第一ReLU输出；不强制负激活。
    point_u = u.flatten(1)[:, point].reshape(n,4)
    for reference in (0,3):
        patched = u.clone()
        patched.flatten(1)[:, point] = point_u[:, reference:reference+1].expand(-1,4).reshape(-1)
        pv = b.bn2(b.conv2(patched))+x; pe = F.relu(pv)
        logits.append(model.readout(pe.mean((2,3))).reshape(n,4,3))
        all_e.append(pe.reshape(n,4,32,32,32))
        second_flips.append(((pv>0)!=(v>0)).sum((1,2,3)).reshape(n,4))
    logit = torch.stack(logits, 2)  # endpoint,state,mode,class
    channel_terms, spatial_pn = [], []
    for ev in all_e:
        maps = torch.einsum('ts,nschw->ntchw', matrix, ev)*q[:,None]
        channel_terms.append(maps.sum((-1,-2)))
        spatial_pn.append(torch.stack((maps.clamp_min(0).sum((2,3,4)), maps.clamp_max(0).sum((2,3,4))), -1))
    channel_terms = torch.stack(channel_terms, 1)  # endpoint,mode,term,32
    spatial_pn = torch.stack(spatial_pn, 1)
    task_terms = torch.einsum('ts,nsmk,k->nmt', matrix, logit, w)
    assert float(abs(channel_terms.sum(-1)-task_terms).max()) < 1e-9
    assert float(abs(spatial_pn.sum(-1)-task_terms).max()) < 1e-9
    # H00两层门控固定：δe = 线性传递 + 第一ReLU余项传递 + 第二ReLU余项。
    g1 = (z4[:, :1]>0); g2 = (v4[:, :1]>0)
    dz = z4-z4[:,:1]; du = u4-u4[:,:1]; dv = v4-v4[:,:1]
    dh = fields-fields[:,:1]; r1 = du-g1*dz; r2 = e4-e4[:,:1]-g2*dv
    scale = b.bn2.weight/torch.sqrt(b.bn2.running_var+b.bn2.eps)
    linear2 = lambda value: (F.conv2d(value.reshape(-1,32,32,32), b.conv2.weight, padding=b.conv2.padding)*scale[None,:,None,None]).reshape(n,4,32,32,32)
    parts = (g2*(linear2(g1*dz)+dh), g2*linear2(r1), r2)
    branch_channels = torch.stack([(part*q[:,None]).sum((-1,-2)) for part in parts], 2)
    branch_terms = torch.einsum('ts,nspc->ntp', matrix, branch_channels)
    assert float(abs(branch_terms.sum(-1)-task_terms[:,0]).max()) < 1e-9
    assert float(abs(branch_terms[:,2,0]).max()) < 1e-9  # 四状态的仿射交互严格为0。
    # 单个成员恢复为干预前的实际值，检验“成员的变化”而非旧A1必要性。
    drop_logits = np.zeros((4,2,32,3)); drop_z = np.zeros((4,2,32))
    drops = []
    for slot in DROP_SLOTS:
        for state in (1,3):
            for channel in channels:
                value = fields[slot,state].clone(); value[channel] = before[channel]; drops.append(value)
    if drops:
        dx = torch.stack(drops); dzj = b.bn1(b.conv1(dx)).flatten(1)[:,point]
        dl = model.readout(b(dx).mean((2,3)))
        count = len(channels)
        drop_logits[:,:,:count] = dl.reshape(4,2,count,3).cpu().numpy()
        drop_z[:,:,:count] = dzj.reshape(4,2,count).cpu().numpy()
    delta = after-before
    inc = torch.zeros(32, dtype=torch.bool, device=x.device); inc[channels] = True
    cmax = abs(delta[:,inc]).flatten(1).max(1).values if inc.any() else torch.zeros(n,device=x.device,dtype=x.dtype)
    omax = abs(delta[:,~inc]).flatten(1).max(1).values if (~inc).any() else torch.zeros(n,device=x.device,dtype=x.dtype)
    out = dict(logits=logit, point_z=z.flatten(1)[:,point].reshape(n,4), point_u=point_u,
        channel_terms=channel_terms, spatial_pn=spatial_pn, branch_terms=branch_terms,
        natural_flips=torch.stack((((z4>0)!=(z4[:,:1]>0)).sum((2,3,4)), ((v4>0)!=(v4[:,:1]>0)).sum((2,3,4))), -1),
        patch_second_flips=torch.stack(second_flips,-1), shifts=torch.stack((cmax,omax),-1))
    return {**{k:v.cpu().numpy() for k,v in out.items()}, 'drop_logits':drop_logits, 'drop_z':drop_z}
