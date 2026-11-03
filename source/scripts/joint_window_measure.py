"""整图重算冻结的39个端点，区分分解变化、实际传递、C消融与最终任务输出。"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from audit_dsprites_gradient import affine_adjoint
from joint_window_mechanism import rebuild, geometry, crop

FIELDS = ('moving_C', 'fixed_C', 'actual_H', 'actual_no_C')
DROP_ENDPOINTS = (0, 25, 28, 31, 34)


@torch.no_grad()
def measure(model, seven, original, mask, frozen, weight, local_cache):
    """同一原A1的C替换值和j开关参考用于全部端点，不能随AI漂移。

上游回退非负；AI及固定j仍为内部诊断反事实，可产生负值。原始F、
H变化和点控制都保留，任何一个开关跨阈值不直接当成任务收益。
"""
    a, r = rebuild(seven, mask); a1, _ = rebuild(original['h'], mask)
    n = len(seven); target = seven[:, 5 if mask == 3 else 6]
    C = frozen['channels']; C = C[C >= 0]; point = int(frozen['lists'][0, 0])
    moving = a.clone(); moving[:, C] = target[:, C]
    fixed = a1.expand_as(a).clone(); fixed[:, C] = target[:, C]
    noC = target.clone(); noC[:, C] = a1[C]
    x = torch.stack((moving, fixed, target, noC), 1).reshape(-1, 32, 32, 32)
    b = model.block3; z = b.bn1(b.conv1(x)); u = F.relu(z); v = b.bn2(b.conv2(u))+x; e = F.relu(v)
    natural = model.readout(e.mean((2, 3)))
    z0 = b.bn1(b.conv1(a1[None])); g0 = z0 > 0; u0 = b.mid(a1[None])
    v0 = b.bn2(b.conv2(u0))+a1[None]; g2 = v0 > 0
    q = (torch.as_tensor(weight, device=x.device, dtype=x.dtype)@model.readout.weight)[None, :, None, None]/1024
    p0 = affine_adjoint(b.conv2, b.bn2, g2*q, u0.shape)
    held_u = u.clone(); held_u.flatten(1)[:, point] = g0.flatten()[point]*z.flatten(1)[:, point]
    held_v = b.bn2(b.conv2(held_u))+x; held_e = F.relu(held_v)
    held = model.readout(held_e.mean((2, 3)))
    direct = p0.flatten()[point]*(u-g0*z).flatten(1)[:, point]
    response = (((e-g2*v)-(held_e-g2*held_v))*q).sum((1, 2, 3))
    net = (natural-held)@torch.as_tensor(weight, device=x.device, dtype=x.dtype)
    assert float(abs(net-direct-response).max()) < 1e-9
    channel_net = ((e-held_e)*q).sum((2, 3)); assert float(abs(channel_net.sum(1)-net).max()) < 1e-9
    e_actual = e.reshape(n, 4, 32, 32, 32)[:, 2]
    channel_change = ((e_actual-e_actual[:1])*q).sum((2, 3))
    logits = natural.reshape(n, 4, 3)
    assert float(abs(channel_change.sum(1)-(logits[:, 2]-logits[:1, 2])@torch.as_tensor(weight, device=x.device, dtype=x.dtype)).max()) < 1e-9
    h0 = original['h'][5 if mask == 3 else 6]; deltaH = target-h0
    deltaA = a-a1; deltaR = r-(h0-a1)
    identity = abs(deltaH-deltaA-deltaR).flatten(1).max(1).values
    assert float(identity.max()) < 1e-9
    shifts = torch.stack((abs(deltaH).flatten(1).max(1).values, deltaH.square().mean((1, 2, 3)).sqrt(),
                          abs(deltaA).flatten(1).max(1).values, abs(deltaR).flatten(1).max(1).values), 1)
    za = z.reshape(n, 4, 32, 32, 32)[:, 2]; va = v.reshape(n, 4, 32, 32, 32)[:, 2]
    flips = torch.stack((((za > 0) != (za[:1] > 0)).sum((1, 2, 3)), ((va > 0) != (va[:1] > 0)).sum((1, 2, 3))), 1)
    out = dict(natural=logits.cpu().numpy(), held=held.reshape(n, 4, 3).cpu().numpy(),
        direct=direct.reshape(n, 4).cpu().numpy(), response=response.reshape(n, 4).cpu().numpy(),
        point_channel_net=channel_net.reshape(n, 4, 32).cpu().numpy(), channel_change=channel_change.cpu().numpy(),
        shifts=shifts.cpu().numpy(), identity=identity.cpu().numpy(), actual_flips=flips.cpu().numpy(),
        point_z=z.flatten(1)[:, point].reshape(n, 4).cpu().numpy(),
        actual_point_flip=((za.flatten(1)[:, point] > 0) != (za[:1].flatten(1)[:, point] > 0)).cpu().numpy(),
        baseline=model.readout(b(a).mean((2, 3))).cpu().numpy())
    # 几何量从整图H裁剪重算，与搜索时局部支持独立拼接。
    local_h = crop(seven, local_cache['y'], local_cache['x'], 1)
    out['margins'] = geometry(local_cache, local_h).cpu().numpy()
    drops = []; dids = []
    for endpoint in DROP_ENDPOINTS:
        for channel in C:
            value = target[endpoint].clone(); value[channel] = a1[channel]; drops.append(value); dids.append((endpoint, channel))
    drop_fields = torch.stack(drops); drop_z = b.bn1(b.conv1(drop_fields)).flatten(1)[:, point]
    drop_logits = model.readout(b(drop_fields).mean((2, 3)))
    padded_logits = np.zeros((5, 32, 3)); padded_z = np.zeros((5, 32))
    for k, (endpoint, channel) in enumerate(dids):
        i = DROP_ENDPOINTS.index(endpoint); j = int(np.flatnonzero(C == channel)[0])
        padded_logits[i, j] = drop_logits[k].cpu().numpy(); padded_z[i, j] = float(drop_z[k])
    out.update(drop_logits=padded_logits, drop_z=padded_z)
    return out
