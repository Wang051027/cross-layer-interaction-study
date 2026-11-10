"""对第一处 ReLU 的 32 个通道贡献按目标方向排序，计算触发阈值所需的最小通道数，再重算输出。"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from residual_route_mechanism import trace

EPS = 1e-8  # 预激活落入这个容差带时跳过最小通道数判定。
MASKS = (3, 15)
HOLD_NAMES = ('selected1', 'random0_1', 'random1_1', 'selected32',
              'random0_32', 'random1_32', 'all_half_joint', 'all_first')


def minimum_order(base, changes, direction):
    """每行一个坐标；给出全32通道中跨阈值的最小基数及严格界。

    direction=+1为开，−1为关。只有基准、最优k−1和最优k均离零
    至少EPS才标记为可判定。按物理通道编号打破平分。
    """
    base = np.atleast_1d(base).astype(float)
    changes = np.asarray(changes, dtype=float).reshape(-1, 32)
    direction = np.atleast_1d(direction).astype(int)
    oriented = changes * direction[:, None]
    order = np.argsort(-oriented, axis=1, kind='stable')
    ranked = np.take_along_axis(oriented, order, axis=1)
    cumulative = np.cumsum(ranked, axis=1)
    gap = -direction * base
    crossed = cumulative > gap[:, None] + EPS
    reachable = crossed.any(axis=1) & (gap > EPS)
    k = np.where(reachable, crossed.argmax(axis=1) + 1, 0)
    before = np.zeros(len(base)); after = np.zeros(len(base))
    rows = np.flatnonzero(reachable)
    before[rows] = np.where(k[rows] == 1, 0., cumulative[rows, np.maximum(k[rows] - 2, 0)])
    after[rows] = cumulative[rows, k[rows] - 1]
    before_margin = gap - before
    after_margin = after - gap
    valid = reachable & (before_margin > EPS) & (after_margin > EPS)
    return dict(k=np.where(valid, k, 0), order=order, valid=valid,
                before_margin=before_margin, after_margin=after_margin)


@torch.no_grad()
def incoming(model, residual, coordinates):
    """精确提取每个坐标的32通道预激活变化；不保存全体巨型张量。"""
    ids = np.asarray(coordinates, dtype=np.int64)
    if not len(ids): return np.empty((0, 32))
    conv, bn = model.block3.conv1, model.block3.bn1
    assert conv.groups == 1 and conv.stride == (1, 1) and conv.padding == (1, 1)
    assert conv.dilation == (1, 1) and conv.kernel_size == (3, 3)
    patches = F.unfold(residual, 3, padding=1)[0].reshape(32, 9, 1024)
    out = torch.as_tensor(ids // 1024, device=residual.device)
    position = torch.as_tensor(ids % 1024, device=residual.device)
    kernels = conv.weight[out].reshape(-1, 32, 9)
    scale = bn.weight[out] / torch.sqrt(bn.running_var[out] + bn.eps)
    result = (kernels * patches[:, :, position].permute(2, 0, 1)).sum(2) * scale[:, None]
    return result.cpu().numpy()


@torch.no_grad()
def census(model, t):
    """枚举当前端点第一处 ReLU 的位置，汇总全强度下的联合触发情况。"""
    za = t['za'].flatten().cpu().numpy(); zh = t['zh'].flatten().cpu().numpy()
    flip = (za > 0) != (zh > 0)
    robust = flip & (abs(za) > EPS) & (abs(zh) > EPS)
    ids = np.flatnonzero(robust)
    direction = np.where(zh[ids] > 0, 1, -1)
    changes = incoming(model, t['h'] - t['a'], ids)
    error = float(np.max(abs(za[ids] + changes.sum(1) - zh[ids]))) if len(ids) else 0.
    assert error < 1e-9, error
    minimum = minimum_order(za[ids], changes, direction)
    kmap = np.zeros(32768, dtype=np.int16); kmap[ids] = minimum['k']
    joint = kmap >= 2
    # 八个计数不充当独立样本；图像/案例/模型分母另行保留。
    counts = np.array([32768, flip.sum(), robust.sum(), (~minimum['valid']).sum(),
                       (kmap == 1).sum(), joint.sum(), (joint & (zh > 0)).sum(),
                       (joint & (zh < 0)).sum()], dtype=np.int64)
    hist = np.bincount(kmap[kmap > 0], minlength=33)[1:33]
    return dict(counts=counts, histogram=hist, kmap=kmap, pool=joint,
                affine_error=error, ids=ids, changes=changes, minimum=minimum)


def select(t, population, seed):
    """半强度选择：按基准梯度加权余项选32点，并保留两套同池随机对照。"""
    pool = np.flatnonzero(population['pool']); budget = min(32, len(pool))
    direct = t['first'].flatten().cpu().numpy()
    score = np.where(abs(direct) < 1e-12, 0., abs(direct))
    ordered = pool[np.lexsort((pool, -score[pool]))]
    rng = np.random.default_rng(seed)
    lists = [ordered[:budget], rng.permutation(pool)[:budget], rng.permutation(pool)[:budget]]
    padded = np.array([np.pad(v, (0, 32-len(v)), constant_values=-1) for v in lists])
    chosen = int(padded[0, 0]); channels = np.full(32, -1, dtype=np.int64)
    k = 0; direction = 0
    if chosen >= 0:
        row = int(np.flatnonzero(population['ids'] == chosen)[0])
        k = int(population['minimum']['k'][row])
        channels[:k] = population['minimum']['order'][row, :k]
        direction = 1 if float(t['zh'].flatten()[chosen]) > 0 else -1
    return dict(lists=padded, pool=population['pool'], channels=channels,
                half_k=np.array(k), half_direction=np.array(direction), budget=np.array(budget))


@torch.no_grad()
def hold(model, t, coordinate_lists):
    """固定指定第一ReLU到当前A的开关，第二ReLU真实重算。

    当基准开而自然关时，固定网络会传入负的预激活；这是计算反事实，
    并非声称存在自然负ReLU激活。返回三分类logit和两层抵消项。
    """
    outputs = []; direct = []; response = []
    b = model.block3
    for ids in coordinate_lists:
        ids = np.asarray(ids, dtype=np.int64); ids = ids[ids >= 0]
        u = t['uh'].clone(); u.flatten(1)[:, ids] = (t['ga'] * t['zh']).flatten(1)[:, ids]
        v = b.bn2(b.conv2(u)) + t['h']
        outputs.append(model.readout(F.relu(v).mean((2, 3)))[0].cpu().numpy())
        direct.append(float(t['first'].flatten()[ids].sum()))
        new_second = (F.relu(v) - t['gb'] * v) * t['q']
        response.append(float((t['second'] - new_second).sum()))
    return np.array(outputs), np.array(direct), np.array(response)


@torch.no_grad()
def measure(model, a, h, weight, population, selection):
    """同一套冻结名单，在两种强度执行同样的实际干预。"""
    t = trace(model, a, h, weight); w = np.asarray(weight)
    lists = selection['lists']; point = int(lists[0, 0]); pool = np.flatnonzero(selection['pool'])
    groups = [v[:1] for v in lists] + list(lists) + [pool, np.arange(32768)]
    held, direct, response = hold(model, t, groups)
    natural = t['modes'][0, 0].cpu().numpy(); baseline = t['baseline'][0].cpu().numpy()
    assert np.max(abs((natural - held) @ w - direct - response)) < 1e-9
    # 半强度选中坐标在当前强度的机械状态；0表示无可靠跨阈值最小数。
    list_k = np.zeros_like(lists)
    za = t['za'].flatten().cpu().numpy(); zh = t['zh'].flatten().cpu().numpy()
    for i, ids in enumerate(lists):
        valid = ids >= 0; list_k[i, valid] = population['kmap'][ids[valid]]
    # 只测半强度选出的通道组：空、k−1、k、全部32，绝不全强度重选组。
    channels = selection['channels']; channels = channels[channels >= 0]
    n = len(channels)
    fields = a.expand(4, -1, -1, -1).clone()
    if n:
        fields[1, channels[:-1]] = h[0, channels[:-1]]
        fields[2, channels] = h[0, channels]
    fields[3] = h[0]
    b = model.block3
    z = b.bn1(b.conv1(fields)); u = F.relu(z); v = b.bn2(b.conv2(u)) + fields
    patch_logits = model.readout(F.relu(v).mean((2, 3))).cpu().numpy()
    patch_held = patch_logits.copy(); patch_z = np.zeros(4); solo_z = np.zeros(32)
    point_change = np.zeros(32); bounds = np.zeros(2)
    meta = np.zeros(11)  # ZA,ZH,half方向,当前k,固定组翻转,同向端点联合,同向固定组联合,最优界前后,half k,真实单通道误差。
    if point >= 0:
        point_change = incoming(model, h-a, [point])[0]
        # 独立于unfold，实际32个单通道conv+BN检查仿射分项。
        solos = a.expand(32, -1, -1, -1).clone()
        for c in range(32): solos[c, c] = h[0, c]
        solo_z = b.bn1(b.conv1(solos)).flatten(1)[:, point].cpu().numpy()
        solo_error = float(np.max(abs(solo_z - za[point] - point_change)))
        assert solo_error < 1e-9
        patch_z = z.flatten(1)[:, point].cpu().numpy()
        expected = za[point] + np.array([0., point_change[channels[:-1]].sum(), point_change[channels].sum(), point_change.sum()])
        assert np.max(abs(patch_z - expected)) < 1e-9
        held_u = u.clone(); held_u.flatten(1)[:, point] = (za[point] > 0) * z.flatten(1)[:, point]
        held_v = b.bn2(b.conv2(held_u)) + fields
        patch_held = model.readout(F.relu(held_v).mean((2, 3))).cpu().numpy()
        half_direction = int(selection['half_direction'])
        same_endpoint = (-half_direction * za[point] > EPS) and (half_direction * zh[point] > EPS)
        fixed_flips = (-half_direction * za[point] > EPS) and (half_direction * patch_z[2] > EPS)
        single_safe = np.max(half_direction * solo_z) < -EPS
        minimum = minimum_order([za[point]], point_change[None], [half_direction])
        bounds = np.array([minimum['before_margin'][0], minimum['after_margin'][0]])
        meta = np.array([za[point], zh[point], half_direction, population['kmap'][point],
                         fixed_flips, same_endpoint and population['kmap'][point] >= 2,
                         fixed_flips and single_safe, *bounds, n, solo_error])
    # 所有半强度联合候选在当前强度仍联合的比例，包含失效坐标。
    pool_stats = np.array([len(pool), np.sum(population['kmap'][pool] > 0),
                           np.sum(population['kmap'][pool] >= 2)], dtype=np.int64)
    return dict(natural=natural, baseline=baseline, held=held, direct=direct, response=response,
                counts=population['counts'], histogram=population['histogram'], pool_stats=pool_stats,
                list_k=list_k, point_meta=meta, patch_logits=patch_logits, patch_held=patch_held,
                patch_z=patch_z, solo_z=solo_z, point_change=point_change,
                affine_error=np.array(population['affine_error']))
