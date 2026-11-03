"""冻结原通道组C和下游点j，比较A单独修复与上游联动重算的阈值和输出。"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from residual_route_mechanism import trace
from upstream_rescue_mechanism import EPS, METHODS, target_weight

REGIMES = ('A_only', 'coherent')
CONTEXTS = ('frozen_C', 'all32')


@torch.no_grad()
def measure(model, a, r, r1, weight, frozen):
    """输出轴为方法×两种R控制×C/all32；每种条件保留三类原始logit。

点干预净量=第一余项直接项+第二层响应，可为零或负；它与整个
上游干预对输出的总作用不同。基准输出单独保存，不能只报告d。
"""
    point = int(frozen['lists'][0, 0]); C = frozen['channels']; C = C[C >= 0]
    direction = int(frozen['half_direction']); k = len(METHODS)
    residuals = torch.stack((r1.expand_as(r), r), 1)
    restricted = torch.zeros_like(residuals); restricted[:, :, C] = residuals[:, :, C]
    bases = a[:, None, None].expand(k, 2, 2, 32, 32, 32).reshape(-1, 32, 32, 32)
    fields = (a[:, None, None]+torch.stack((restricted, residuals), 2)).reshape(-1, 32, 32, 32)
    t = trace(model, bases, fields, weight); b = model.block3
    natural = t['modes'][:, 0]; u = t['uh'].clone()
    u.flatten(1)[:, point] = (t['ga']*t['zh']).flatten(1)[:, point]
    v = b.bn2(b.conv2(u))+fields; e = F.relu(v)
    held = model.readout(e.mean((2, 3)))
    direct = t['first'].flatten(1)[:, point]
    response = (t['second']-(e-t['gb']*v)*t['q']).sum((1, 2, 3))
    channel_net = ((t['eh']-e)*t['q']).sum((2, 3))
    net = (natural-held)@torch.as_tensor(weight, device=a.device, dtype=a.dtype)
    assert float(abs(net-direct-response).max()) < 1e-9
    assert float(abs(net-channel_net.sum(1)).max()) < 1e-9
    d_channel = ((t['eh']-F.relu(t['va']))*t['q']).sum((2, 3))
    parts = torch.stack([t[key].sum((2, 3)) for key in ('linear', 'first', 'second')], 1)
    closed = ((t['vh'] > 0) & ~(v > 0)).sum((1, 2, 3)); opened = (~(t['vh'] > 0) & (v > 0)).sum((1, 2, 3))
    flips = torch.stack([((t['zh'] > 0) != t['ga']).sum((1, 2, 3)), ((t['vh'] > 0) != t['gb']).sum((1, 2, 3))], 1)
    values = dict(natural=natural, held=held, direct=direct, response=response, channel_net=channel_net,
                  d_channel=d_channel, parts=parts, second_flips=torch.stack((closed, opened), 1), flips=flips,
                  point_remainder=(t['uh']-t['ga']*t['zh']).flatten(1)[:, point])
    out = {key: val.cpu().numpy().reshape(k, 2, 2, *val.shape[1:]) for key, val in values.items()}
    za = b.bn1(b.conv1(a)); base = za.flatten(1)[:, point]
    wj = target_weight(model, point)
    incoming = (residuals*wj[:, None]).sum((3, 4))
    zc = base[:, None]+incoming[:, :, C].sum(2); zall = base[:, None]+incoming.sum(2)
    z = t['zh'].flatten(1)[:, point].reshape(k, 2, 2)
    assert float(abs(z-torch.stack((zc, zall), 2)).max()) < 1e-9
    same = -direction*base > EPS
    cross = same[:, None] & (direction*zc > EPS)
    solo = (direction*(base[:, None, None]+incoming)).max(2).values < -EPS
    flags = torch.stack((same[:, None].expand(k, 2), cross, solo, cross & solo,
                         same[:, None] & (direction*zall > EPS)), 2)
    out.update(baseline=model.readout(model.block3(a).mean((2, 3))).cpu().numpy(),
               base_z=base.cpu().numpy(), incoming=incoming.cpu().numpy(), group_z=zc.cpu().numpy(),
               all_z=zall.cpu().numpy(), threshold=flags.cpu().numpy())
    return out
