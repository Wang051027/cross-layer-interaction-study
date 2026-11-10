"""分开控制基准A和组合残差R，解释冻结通道组的阈值失效及输出变化。

四格按00、01、10、11排列：第一个下标控制A，第二个控制R；
0来自半强度，1来自完整强度。通道、开关和随机名单全部沿用前轮，
没有重新定位。C为冻结通道组，all32为全部中间组合残差。
"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from residual_route_mechanism import trace
from joint_relu_mechanism import incoming, minimum_order, HOLD_NAMES, EPS

CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
CONTEXTS = ('frozen_C', 'all32')


def factorial(values):
    """分解四格响应，分别返回基准项、残差项和交互项。"""
    y00, y01, y10, y11 = np.moveaxis(np.asarray(values), -1, 0)
    return np.stack((y10-y00, y01-y00, y11-y10-y01+y00), axis=-1)


def gate_groups(selection):
    """前轮八套冻结第一ReLU名单，预算与原随机对照保持一致。"""
    lists = selection['lists']
    return [ids[ids >= 0] for ids in [v[:1] for v in lists]+list(lists)] + [np.flatnonzero(selection['pool']), np.arange(32768)]


@torch.no_grad()
def fixed_first(model, x, z, reference_gate, groups):
    """只固定第一开关，第二层始终重新运行；支持同一参考或逐样本参考。"""
    fields = []
    for ids in groups:
        u = F.relu(z).clone()
        u.flatten(1)[:, ids] = (reference_gate*z).flatten(1)[:, ids]
        fields.append(u)
    count = len(x); u = torch.cat(fields)
    v = model.block3.bn2(model.block3.conv2(u)) + x.repeat(len(groups), 1, 1, 1)
    e = F.relu(v)
    logits = model.readout(e.mean((2, 3))).reshape(len(groups), count, 3).transpose(0, 1)
    return logits, v.reshape(len(groups), count, 32, 32, 32), e.reshape(len(groups), count, 32, 32, 32)


@torch.no_grad()
def compute(model, a, r, weight, selection):
    """A[2]和R[2]形成四格；同时记录阈值、两层传递和实际读出。"""
    assert a.shape == r.shape == (2, 32, 32, 32)
    point = int(selection['lists'][0, 0]); assert point >= 0
    channels = selection['channels']; channels = channels[channels >= 0]
    direction = int(selection['half_direction']); groups = gate_groups(selection)
    ab = torch.stack([a[i] for i, j in CELLS])
    rr = torch.stack([r[j] for i, j in CELLS])
    restricted = torch.zeros_like(rr); restricted[:, channels] = rr[:, channels]
    # 两种空间场都使用同一个Ai作基准，组外残差不隐式加入。
    bases = ab.repeat(2, 1, 1, 1)
    x = torch.cat((ab+restricted, ab+rr))
    t = trace(model, bases, x, weight); w = np.asarray(weight)
    natural = t['modes'][:, 0]
    held, held_v, held_e = fixed_first(model, x, t['zh'], t['ga'], groups)
    direct = torch.stack([t['first'].flatten(1)[:, ids].sum(1) for ids in groups], 1)
    remainder = (held_e-t['gb'][None]*held_v)*t['q'][None]
    response = (t['second'][None]-remainder).sum((2, 3, 4)).transpose(0, 1)
    channel_net = ((t['eh'][None]-held_e)*t['q'][None]).sum((3, 4)).permute(1, 0, 2)
    closed = ((t['vh'] > 0)[None] & ~(held_v > 0)).sum((2, 3, 4)).transpose(0, 1)
    opened = (~(t['vh'] > 0)[None] & (held_v > 0)).sum((2, 3, 4)).transpose(0, 1)
    second_flips = torch.stack((closed, opened), 2)
    net = (natural[:, None]-held).cpu().numpy()@w
    assert np.max(abs(net-direct.cpu().numpy()-response.cpu().numpy())) < 1e-9
    assert np.max(abs(channel_net.sum(2).cpu().numpy()-net)) < 1e-9

    # 同一A0开关的控制：先扣除该控制本身对Ai基准输出的改变。
    common_gate = t['ga'][:1]
    common_held = fixed_first(model, x, t['zh'], common_gate, groups)[0]
    zbase = model.block3.bn1(model.block3.conv1(a))
    common_baseline_held = fixed_first(model, a, zbase, common_gate, groups)[0]
    baseline_logits = model.readout(model.block3(a).mean((2, 3)))

    # 在第一层的自然激活上给同一坐标加 1，作为等幅度探针。
    probe_u = t['uh'].clone(); probe_u.flatten(1)[:, point] += 1.
    probe = model.readout(model.block3.finish(probe_u, x).mean((2, 3)))
    point_remainder = (t['uh']-t['ga']*t['zh']).flatten(1)[:, point]
    point_effect = torch.as_tensor(net[:, 0], device=x.device)
    gain_valid = point_remainder > EPS
    point_gain = torch.where(gain_valid, point_effect/point_remainder.clamp_min(EPS), 0.)

    # 第一预激活对Ai和Rj可严格分离；对ReLU之后和最终输出不作线性假设。
    b = zbase.flatten(1)[:, point].cpu().numpy()
    changes = np.stack([incoming(model, r[j:j+1], [point])[0] for j in range(2)])
    zc = t['zh'][:4].flatten(1)[:, point].cpu().numpy()
    za = t['zh'][4:].flatten(1)[:, point].cpu().numpy()
    predicted_c = np.array([b[i]+changes[j, channels].sum() for i, j in CELLS])
    predicted_all = np.array([b[i]+changes[j].sum() for i, j in CELLS])
    assert np.max(abs(predicted_c-zc)) < 1e-9 and np.max(abs(predicted_all-za)) < 1e-9
    assert abs(float(factorial(zc)[2])) < 1e-9
    rows = []
    for cell, (i, j) in enumerate(CELLS):
        baseline_same = -direction*b[i] > EPS
        group_cross = baseline_same and direction*zc[cell] > EPS
        safe = np.max(direction*(b[i]+changes[j])) < -EPS
        minimum = minimum_order([b[i]], changes[j:j+1], [direction])
        rows.append([baseline_same, group_cross, safe, group_cross and safe,
                     int(minimum['k'][0]), minimum['before_margin'][0], minimum['after_margin'][0],
                     baseline_same and direction*za[cell] > EPS])
    threshold = np.asarray(rows)
    assert threshold[0, 3] == 1  # 原半强度选择应精确复现。
    d_channel = ((t['eh']-F.relu(t['va']))*t['q']).sum((2, 3))
    parts = torch.stack([t[key].sum((2, 3)) for key in ('linear', 'first', 'second')], 1)
    assert np.max(abs(parts.sum((1, 2)).cpu().numpy()-(natural-t['baseline']).cpu().numpy()@w)) < 1e-9
    flips = torch.stack([((t['zh'] > 0) != t['ga']).sum((1, 2, 3)),
                         ((t['vh'] > 0) != t['gb']).sum((1, 2, 3))], 1)
    arrays = dict(natural=natural, held=held, direct=direct, response=response, channel_net=channel_net,
                  second_flips=second_flips, common_held=common_held, probe=probe,
                  point_remainder=point_remainder, point_gain=point_gain, gain_valid=gain_valid,
                  d_channel=d_channel, parts=parts, flips=flips)
    # 第一轴为C/all32，第二轴四格，后续轴按上面字段定义。
    out = {key: value.cpu().numpy().reshape(2, 4, *value.shape[1:]) for key, value in arrays.items()}
    out.update(baseline=baseline_logits.cpu().numpy(), common_baseline_held=common_baseline_held.cpu().numpy(),
               base_z=b, incoming=changes, group_z=zc, all_z=za, threshold=threshold)
    return out
