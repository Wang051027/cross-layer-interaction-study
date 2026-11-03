"""在固定名单内找原C每一成员都必要的上游恢复，几何搜索不读取任务输出。

三种参照分别是移动AI、固定原A1、真实HI保留组外值且C用原A1消融。
后两种只依赖HI与固定A1，因此只改变A/R分解不可能使它们恢复。
所有新控制为半/全强度非负激活的凸组合，不使用负的强制gate激活。
"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F

EPS = 1e-8
GUARD = 1e-6
GRID = np.linspace(0., 1., 33)
STRATEGIES = ('moving_A', 'fixed_actual', 'aligned')
FAMILIES = ('source', 'middle', 'first_activation', 'second_activation')
VARIANTS = ('selected', 'random0', 'random1')
CONTEXTS = ('moving_A', 'fixed_A1', 'actual_context')
STATES = (0, 1, 2, 4, 8, 3, 15)
PREFIXES = ((1, 2, 3, 4), (1, 2, 3, 4), (1, 4, 8, 16), (1, 4, 8, 16))


def rebuild(h, mask):
    """七状态重建A和R；只使用S内单组锚点。"""
    ids = [g+1 for g in range(4) if mask & (1 << g)]
    a = h[..., ids, :, :, :].sum(-4)-(len(ids)-1)*h[..., 0, :, :, :]
    return a, h[..., 5 if mask == 3 else 6, :, :, :]-a


def crop(value, y, x, radius):
    """固定中心裁剪，图外填0；中间虚拟ReLU另用有效掩码置0。"""
    return F.pad(value, (radius, radius, radius, radius))[..., y:y+2*radius+1, x:x+2*radius+1]


@torch.no_grad()
def cache(model, half, full, point, mask, frozen):
    """精确局部支持：11×11源→5×5第一激活→3×3中间输出。

源通道改变对第一仿射层和shortcut的作用分别缓存；搜索时两次ReLU
仍真实重算。这里的加速不假定最终输出线性；独立审计用直接裁剪卷积。
"""
    co, y, x = np.unravel_index(int(point), (32, 32, 32)); b = model.block2
    device, dtype = full['x'].device, full['x'].dtype
    udelta = crop(half['x']-full['x'], 2*y, 2*x, 5)
    single = torch.zeros((16, 7, 16, 11, 11), dtype=dtype, device=device)
    for c in range(16): single[c, :, c] = udelta[:, c]
    source = single.reshape(112, 16, 11, 11)
    scale1 = b.bn1.weight/torch.sqrt(b.bn1.running_var+b.bn1.eps)
    dz = F.conv2d(source, b.conv1.weight, stride=2)*scale1[None, :, None, None]
    sc = b.shortcut; ss = sc[1].weight/torch.sqrt(sc[1].running_var+sc[1].eps)
    ds = F.conv2d(source[:, :, 3:8, 3:8], sc[0].weight, stride=2)*ss[None, :, None, None]
    yi, xi = torch.meshgrid(torch.arange(y-2, y+3, device=device), torch.arange(x-2, x+3, device=device), indexing='ij')
    valid1 = ((yi >= 0) & (yi < 32) & (xi >= 0) & (xi < 32))[None, None]
    valid2 = valid1[..., 1:4, 1:4]
    b3 = model.block3; scale3 = b3.bn1.weight[co]/torch.sqrt(b3.bn1.running_var[co]+b3.bn1.eps)
    w = b3.conv1.weight[co]*scale3
    bias = b3.bn1.bias[co]-scale3*b3.bn1.running_mean[co]
    a1, _ = rebuild(full['h'], mask); ac = crop(a1, y, x, 1)
    C = frozen['channels']; C = C[C >= 0]
    return dict(model=model, mask=mask, y=y, x=x, w=w, bias=bias, C=C, direction=int(frozen['half_direction']),
                dz=dz.reshape(16, 7, 32, 5, 5)*valid1, ds=ds.reshape(16, 7, 32, 3, 3),
                z1=crop(full['z1'], y, x, 2), u1=crop(full['u'], y, x, 2), u0=crop(half['u'], y, x, 2),
                skip=crop(full['skip'], y, x, 1), h1=crop(full['h'], y, x, 1), h0=crop(half['h'], y, x, 1),
                a1=ac, valid1=valid1, valid2=valid2)


def paths(selection):
    """继承旧定向/两随机名单，取预定前缀；同实际长度的重复前缀仅保留一次。"""
    result = []
    for family in range(4):
        lists = selection['source_lists'] if family == 0 else selection['middle_lists'] if family == 1 else selection['gate_lists'][family-2]
        for variant in range(3):
            ids = lists[variant]; ids = ids[ids >= 0]; seen = set()
            for requested in PREFIXES[family]:
                k = min(requested, len(ids)); valid = k > 0 and k not in seen; seen.add(k)
                result.append(dict(family=family, variant=variant, requested=requested, k=k, ids=ids[:k], valid=valid))
    assert len(result) == 48
    return result


@torch.no_grad()
def local_fields(c, family, ids, amounts):
    """返回强度批次×七状态×32×3×3；所有中间激活保持非负。"""
    lam = torch.as_tensor(amounts, device=c['w'].device, dtype=c['w'].dtype)[:, None, None, None, None]
    n = len(amounts); b = c['model'].block2
    if family == 0:
        z1 = c['z1'][None]+lam*c['dz'][ids].sum(0)[None]
        u = F.relu(z1)*c['valid1']; skip = c['skip'][None]+lam*c['ds'][ids].sum(0)[None]
    elif family == 2:
        delta = torch.zeros_like(c['u1'])
        for coordinate in ids:
            ch, yy, xx = np.unravel_index(int(coordinate), (32, 32, 32)); iy, ix = yy-c['y']+2, xx-c['x']+2
            delta[:, ch, iy, ix] = c['u0'][:, ch, iy, ix]-c['u1'][:, ch, iy, ix]
        u = c['u1'][None]+lam*delta[None]; skip = c['skip'][None].expand(n, -1, -1, -1, -1)
    else:
        delta = torch.zeros_like(c['h1'])
        if family == 1: delta[:, ids] = c['h0'][:, ids]-c['h1'][:, ids]
        else:
            for coordinate in ids:
                ch, yy, xx = np.unravel_index(int(coordinate), (32, 32, 32)); iy, ix = yy-c['y']+1, xx-c['x']+1
                delta[:, ch, iy, ix] = c['h0'][:, ch, iy, ix]-c['h1'][:, ch, iy, ix]
        return c['h1'][None]+lam*delta[None]
    z2 = b.bn2(F.conv2d(u.reshape(n*7, 32, 5, 5), b.conv2.weight)).reshape(n, 7, 32, 3, 3)+skip
    return F.relu(z2)*c['valid2']


@torch.no_grad()
def geometry(c, h):
    """三个参照的正余量[基准原侧、全C触发、最强删一成员仍不触发]。

若全C触发且全部删一成员均失败，则每个定向通道贡献为正，任何真
子集都不可能更强。因此不把“单个都不足”误当作每个成员都必要。
"""
    a, _ = rebuild(h, c['mask']); target = h[:, 5 if c['mask'] == 3 else 6]
    linear = lambda v: (v*c['w']).sum((-1, -2, -3))
    b_own = linear(a)+c['bias']; b_fixed = (linear(c['a1'])+c['bias']).expand(len(h))
    own_d = ((target-a)*c['w']).sum((-1, -2))
    fixed_d = ((target-c['a1'])*c['w']).sum((-1, -2))
    total = linear(target)+c['bias']; b_actual = total-fixed_d[:, c['C']].sum(-1)
    bases = torch.stack((b_own, b_fixed, b_actual), 1)
    changes = torch.stack((own_d, fixed_d, fixed_d), 1)
    incoming = c['direction']*changes[:, :, c['C']]
    group = c['direction']*bases+incoming.sum(-1)
    worst_drop = (group[:, :, None]-incoming).max(-1).values
    return torch.stack((-c['direction']*bases, group, -worst_drop), -1)


def objectives(margins):
    """own只要求移动A；physical同时固定A1/实际组外；aligned要求三者。"""
    q = np.min(margins, axis=-1)
    return np.stack((q[..., 0], np.minimum(q[..., 1], q[..., 2]), np.min(q, axis=-1)), -1)


def best_index(scores, budgets, amounts):
    """找到可行时优先少单元再最大安全余量；否则保留最高分失败，不丢案例。"""
    usable = np.flatnonzero(amounts > 0.)
    feasible = usable[scores[usable] >= GUARD]
    if len(feasible): return int(feasible[np.lexsort((amounts[feasible], -scores[feasible], budgets[feasible]))[0]])
    return int(usable[np.lexsort((amounts[usable], budgets[usable], -scores[usable]))[0]])


@torch.no_grad()
def search(c, selection):
    """固定网格后只在已选路径附近细化；读出F、任务梯度都不参与选择。

这是同面板有限搜索，找不到不证明该上游不存在其他强度/名单的恢复。
随机对照有相同前缀、强度网格和最大预算，最优实际预算允许不同。
"""
    pp = paths(selection); coarse = []
    for p in pp: coarse.append(geometry(c, local_fields(c, p['family'], p['ids'], GRID)).cpu().numpy())
    coarse = np.asarray(coarse); score = objectives(coarse)
    refined = []; chosen = []; choices = []; neighbor = []
    for strategy in range(3):
        for family in range(4):
            for variant in range(3):
                eligible = [i for i, p in enumerate(pp) if p['family'] == family and p['variant'] == variant and p['valid']]
                if not eligible:
                    pindex = -1; amounts = np.zeros(17); margins = np.repeat(geometry(c, c['h1'][None]).cpu().numpy(), 17, 0)
                    selected_amount = 0.; selected_margin = margins[0]; valid = False
                else:
                    scores = score[eligible, :, strategy].ravel(); budgets = np.repeat([pp[i]['k'] for i in eligible], 33)
                    amounts0 = np.tile(GRID, len(eligible)); winner = best_index(scores, budgets, amounts0)
                    pindex = eligible[winner//33]; center = amounts0[winner]
                    amounts = np.linspace(max(0., center-1/32), min(1., center+1/32), 17)
                    p = pp[pindex]; margins = geometry(c, local_fields(c, family, p['ids'], amounts)).cpu().numpy()
                    all_margins = np.concatenate((coarse[pindex], margins)); all_amounts = np.concatenate((GRID, amounts))
                    j = best_index(objectives(all_margins)[:, strategy], np.full(50, p['k']), all_amounts)
                    selected_amount = float(all_amounts[j]); selected_margin = all_margins[j]; valid = True
                chosen.append([pindex, selected_amount, valid]); refined.append(margins); choices.append(selected_margin)
                points = np.clip(np.array([selected_amount-1/256, selected_amount, selected_amount+1/256]), 0., 1.)
                ids = [] if pindex < 0 else pp[pindex]['ids']
                neighbor.append(geometry(c, local_fields(c, family, ids, points)).cpu().numpy())
    return dict(coarse_margins=coarse, refine_margins=np.asarray(refined), selected_margins=np.asarray(choices),
                choices=np.asarray(chosen), neighbor_margins=np.asarray(neighbor))


def intervention_specs(selection, result):
    """39个评价端点：自然、36个冻结搜索解、全源/全中间回退阳性控制。"""
    pp = paths(selection); specs = [(-1, np.array([], dtype=int), 0.)]
    for index, amount, valid in result['choices']:
        p = None if index < 0 else pp[int(index)]
        specs.append((-1, np.array([], dtype=int), 0.) if p is None else (p['family'], p['ids'], float(amount)))
    specs += [(0, np.arange(16), 1.), (1, np.arange(32), 1.)]
    return specs


@torch.no_grad()
def full_fields(model, half, full, specs):
    """原block2整图执行与非负激活回退；相同解先合并再展开以节省重复计算。"""
    unique = []; lookup = {}; restore = []
    for family, ids, amount in specs:
        key = (family, tuple(ids), amount)
        if key not in lookup: lookup[key] = len(unique); unique.append((family, ids, amount))
        restore.append(lookup[key])
    values = []
    for family, ids, amount in unique:
        if family < 0 or amount == 0.: h = full['h']
        elif family == 0:
            x = full['x'].clone(); x[:, ids] += amount*(half['x'][:, ids]-x[:, ids]); h = model.block2(x)
        elif family == 2:
            u = full['u'].clone(); u.flatten(1)[:, ids] += amount*(half['u'].flatten(1)[:, ids]-u.flatten(1)[:, ids])
            h = model.block2.finish(u, full['skip'])
        else:
            h = full['h'].clone()
            if family == 1: h[:, ids] += amount*(half['h'][:, ids]-h[:, ids])
            else: h.flatten(1)[:, ids] += amount*(half['h'].flatten(1)[:, ids]-h.flatten(1)[:, ids])
        assert float(h.min()) >= -1e-12; values.append(h)
    h = torch.stack(values)[restore]
    return h, len(unique)
