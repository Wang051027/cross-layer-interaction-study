"""追踪block2如何移动构造基准A，并冻结仅依据基准修复量的上游名单。

A不是自然激活：它由空状态和各单组状态的block2输出重建。
所有上游控制都施加到同一套源状态，再同时重建A与R；本文件
不读取最终分类读出，也不以是否救回原通道组作为排名分数。
"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from audit_dsprites_gradient import affine_adjoint

EPS = 1e-8
MASKS = (3, 15)
STATES = (0, 1, 2, 4, 8, 3, 15)
METHODS = ('natural', 'src_selected4', 'src_random0', 'src_random1', 'src_selected1', 'src_all',
           'mid_selected4', 'mid_random0', 'mid_random1', 'mid_selected1', 'mid_all',
           'g1_selected16', 'g1_random0', 'g1_random1', 'g1_selected1', 'g1_all',
           'g2_selected16', 'g2_random0', 'g2_random1', 'g2_selected1', 'g2_all', 'g12_all')
FAMILIES = ('source', 'middle', 'gate1', 'gate2')


def anchors(mask):
    """返回单组状态在七状态数组中的行号，组外状态不进入该A。"""
    return [g+1 for g in range(4) if mask & (1 << g)]


def reconstruct(h, mask):
    """支持任意前导批轴；末四轴为七状态×32×32×32。"""
    ids = anchors(mask)
    a = h[..., ids, :, :, :].sum(-4)-(len(ids)-1)*h[..., 0, :, :, :]
    return a, h[..., STATES.index(mask), :, :, :]-a


def source_fields(recipient, donor, alpha):
    """显式源组替换，保持原36图和七个已有源状态。"""
    fields = recipient.expand(7, -1, -1, -1).clone()
    for n, mask in enumerate(STATES):
        for g in range(4):
            if mask & (1 << g):
                sl = slice(4*g, 4*g+4)
                fields[n, sl] += alpha*(donor[0, sl]-recipient[0, sl])
    return fields


@torch.no_grad()
def upstream_trace(model, source):
    """原block2的两次预激活、shortcut及自然输出；BN均为eval常量。"""
    b = model.block2
    z1 = b.bn1(b.conv1(source)); u = F.relu(z1); skip = b.shortcut(source)
    z2 = b.bn2(b.conv2(u))+skip
    return dict(x=source, z1=z1, u=u, skip=skip, z2=z2, h=F.relu(z2))


@torch.no_grad()
def target_weight(model, point):
    """冻结block3第一预激活j的线性核；其常量偏置只在算b时加入。"""
    c, y, x = np.unravel_index(int(point), (32, 32, 32))
    b = model.block3
    kernel = b.conv1.weight[c]*(b.bn1.weight[c]/torch.sqrt(b.bn1.running_var[c]+b.bn1.eps))
    w = torch.zeros((1, 32, 32, 32), dtype=kernel.dtype, device=kernel.device)
    for dy in range(3):
        for dx in range(3):
            yy, xx = y+dy-1, x+dx-1
            if 0 <= yy < 32 and 0 <= xx < 32: w[0, :, yy, xx] = kernel[:, dy, dx]
    return w


@torch.no_grad()
def decompose(model, half, full, w):
    """四个单组锚点的精确端点恒等式：线性传播+第一/第二ReLU余项。

每个源状态使用自己的半强度开关。source_linear按16个输入通道分解，
gate_parts按两层各32通道分解；分解项允许互相抵消，不是因果份额。
"""
    b = model.block2; h0 = {k: v[1:5] for k, v in half.items()}; h1 = {k: v[1:5] for k, v in full.items()}
    weight = w.expand_as(h0['h']); g1 = h0['z1'] > 0; g2 = h0['z2'] > 0
    p = affine_adjoint(b.conv2, b.bn2, g2*weight, h0['u'].shape)
    ja = affine_adjoint(b.conv1, b.bn1, g1*p, h0['x'].shape)
    ja += affine_adjoint(b.shortcut[0], b.shortcut[1], g2*weight, h0['x'].shape)
    linear = (ja*(h1['x']-h0['x'])).sum((2, 3))
    first = ((h1['u']-g1*h1['z1'])*p).sum((2, 3))
    second = ((h1['h']-g2*h1['z2'])*weight).sum((2, 3))
    delta = ((h1['h']-h0['h'])*weight).sum((1, 2, 3))
    assert float(abs(delta-linear.sum(1)-first.sum(1)-second.sum(1)).max()) < 1e-9
    flips = []
    for z, g in ((h1['z1'], g1), (h1['z2'], g2)):
        flips.append(torch.stack(((~g & (z > 0)).sum((1, 2, 3)), (g & ~(z > 0)).sum((1, 2, 3))), 1))
    return dict(source_linear=linear.cpu().numpy(), gate_parts=torch.stack((first, second)).cpu().numpy(),
                anchor_delta=delta.cpu().numpy(), upstream_flips=torch.stack(flips).cpu().numpy())


def local_ids(point, radius):
    """卷积的严格空间支持；边缘截断，不把图像外零填充当成可干预开关。"""
    _, y, x = np.unravel_index(int(point), (32, 32, 32))
    return np.array([c*1024+yy*32+xx for c in range(32)
                     for yy in range(max(0, y-radius), min(32, y+radius+1))
                     for xx in range(max(0, x-radius), min(32, x+radius+1))], dtype=np.int64)


@torch.no_grad()
def gate_candidates(model, half, full, mask, point, w, direction):
    """精确局部单开关修复分数；第一层候选实际重算第二ReLU。

利用5×5→3×3支持计算全部候选，避免为每个坐标运行整张特征图。
独立审计另用原finish和裁剪卷积检查本公式及边界处理。
"""
    ids1, ids2 = local_ids(point, 2), local_ids(point, 1)
    rows = anchors(mask); r1 = full['u']-torch.where(half['z1'] > 0, full['z1'], 0.)
    r2 = full['h']-torch.where(half['z2'] > 0, full['z2'], 0.)
    rem1 = r1[rows].flatten(1)[:, ids1]; rem2 = r2[rows].flatten(1)[:, ids2]
    spatial = ids2[:len(ids2)//32] % 1024
    device = w.device; pos = torch.as_tensor(spatial, device=device)
    y2, x2 = pos//32, pos % 32
    p = torch.as_tensor(ids1, device=device); c1, y1, x1 = p//1024, p % 1024//32, p % 32
    ky = y1[:, None]-y2[None]+1; kx = x1[:, None]-x2[None]+1
    valid = (ky >= 0) & (ky < 3) & (kx >= 0) & (kx < 3)
    b = model.block2
    kernel = b.conv2.weight*(b.bn2.weight/torch.sqrt(b.bn2.running_var+b.bn2.eps))[:, None, None, None]
    edge = kernel[:, c1[:, None], ky.clamp(0, 2), kx.clamp(0, 2)].permute(1, 0, 2)*valid[:, None]
    v = full['z2'][rows].flatten(2)[:, :, pos]; weight = w.flatten(2)[0, :, pos]
    delta = -rem1.T[:, :, None, None]*edge[:, None]
    score1 = direction*((F.relu(v[None]+delta)-F.relu(v)[None])*weight[None, None]).sum((1, 2, 3))
    score2 = -direction*(rem2.sum(0)*w.flatten()[ids2])
    pool1 = abs(rem1).max(0).values > EPS; pool2 = abs(rem2).max(0).values > EPS
    return [(ids1, score1.cpu().numpy(), pool1.cpu().numpy()), (ids2, score2.cpu().numpy(), pool2.cpu().numpy())]


def choose(ids, scores, pool, budget, rng, width):
    """只选正修复量；随机来自同一个有效候选池，使用同一实际预算。"""
    eligible = ids[pool]; good = pool & (scores > EPS)
    selected = ids[good][np.lexsort((ids[good], -scores[good]))][:budget]
    n = len(selected)
    lists = [selected, rng.permutation(eligible)[:n], rng.permutation(eligible)[:n]]
    padded = np.array([np.pad(v, (0, width-len(v)), constant_values=-1) for v in lists], dtype=np.int64)
    return padded, np.array([n, len(eligible), int(good.sum())])


@torch.no_grad()
def discover(model, half, full, mask, frozen, seed):
    """只用基准b的定向修复量排名；不读取C的残差或任务logit。"""
    point = int(frozen['lists'][0, 0]); direction = int(frozen['half_direction'])
    w = target_weight(model, point); a0, _ = reconstruct(half['h'], mask); a1, _ = reconstruct(full['h'], mask)
    source_score = np.zeros(16); eligible = np.array([bool(mask & (1 << (c//4))) for c in range(16)])
    candidates = []
    for c in range(16):
        value = full['x'][c//4+1:c//4+2].clone(); value[:, c] = half['x'][c//4+1:c//4+2, c]; candidates.append(value)
    h = model.block2(torch.cat(candidates))
    for c in range(16):
        source_score[c] = direction*float(((h[c:c+1]-full['h'][c//4+1:c//4+2])*w).sum()) if eligible[c] else 0.
    middle_score = (direction*(a0-a1)*w[0]).sum((1, 2)).cpu().numpy()
    gates = gate_candidates(model, half, full, mask, point, w, direction)
    rng = np.random.default_rng(seed)
    sl, sc = choose(np.arange(16), source_score, eligible, 4, rng, 16)
    ml, mc = choose(np.arange(32), middle_score, np.ones(32, dtype=bool), 4, rng, 32)
    output = dict(source_lists=sl, middle_lists=ml, source_scores=source_score, middle_scores=middle_score,
                  source_pool=eligible, **decompose(model, half, full, w))
    lists = []; counts = [sc, mc]
    for stage, (ids, scores, pool) in enumerate(gates):
        selected, count = choose(ids, scores, pool, 16, rng, 16); lists.append(selected); counts.append(count)
        size = 800 if stage == 0 else 288
        output[f'gate{stage+1}_ids'] = np.pad(ids, (0, size-len(ids)), constant_values=-1)
        output[f'gate{stage+1}_scores'] = np.pad(scores, (0, size-len(ids)))
        output[f'gate{stage+1}_pool'] = np.pad(pool, (0, size-len(ids)))
    output.update(gate_lists=np.asarray(lists), selection_counts=np.asarray(counts))
    b = model.block3.bn1(model.block3.conv1(torch.stack((a0, a1)))).flatten(1)[:, point]
    output['base_z'] = b.cpu().numpy()
    assert abs(float(b[1]-b[0])-output['anchor_delta'][np.array(anchors(mask))-1].sum()) < 1e-9
    return output


def method_spec(selection, index, mask):
    """一处声明冻结方法，便于逐方法审计；所有编号为物理通道或坐标。"""
    if index == 0: return 'natural', np.array([], dtype=np.int64)
    if index == 21: return 'both', np.arange(32768)
    family = (index-1)//5; slot = (index-1) % 5
    if slot == 4:
        ids = np.flatnonzero(selection['source_pool']) if family == 0 else np.arange(32 if family == 1 else 32768)
    else:
        key = ('source_lists', 'middle_lists', 'gate_lists', 'gate_lists')[family]
        lists = selection[key] if family < 2 else selection[key][family-2]
        ids = lists[0, :1] if slot == 3 else lists[slot]
        ids = ids[ids >= 0]
    return FAMILIES[family], ids


@torch.no_grad()
def intervene(model, half, full, mask, selection):
    """22种控制同时作用于七状态；第一开关变动后真实重算第二层。

开关固定使用半强度对应状态的gate乘以当前预激活。这是内部计算
反事实，关→开时可能出现负的强制激活，不能称为自然ReLU输出。
"""
    count = len(METHODS); x = full['x'].repeat(count, 1, 1, 1)
    for n in range(count):
        kind, ids = method_spec(selection, n, mask)
        if kind == 'source': x.reshape(count, 7, 16, 64, 64)[n, :, ids] = half['x'][:, ids]
    b = model.block2; z1 = b.bn1(b.conv1(x)); u = F.relu(z1)
    gates1 = (half['z1'] > 0).repeat(count, 1, 1, 1)
    for n in range(count):
        kind, ids = method_spec(selection, n, mask)
        if kind in ('gate1', 'both'):
            uf = u[n*7:(n+1)*7].flatten(1); zf = z1[n*7:(n+1)*7].flatten(1)
            uf[:, ids] = gates1[n*7:(n+1)*7].flatten(1)[:, ids]*zf[:, ids]
    z2 = b.bn2(b.conv2(u))+b.shortcut(x); h = F.relu(z2)
    for n in range(count):
        kind, ids = method_spec(selection, n, mask)
        view = h[n*7:(n+1)*7]
        if kind in ('gate2', 'both'):
            view.flatten(1)[:, ids] = (half['z2'] > 0).flatten(1)[:, ids]*z2[n*7:(n+1)*7].flatten(1)[:, ids]
        if kind == 'middle': view[:, ids] = half['h'][:, ids]
    h = h.reshape(count, 7, 32, 32, 32)
    assert float(abs(h[:, 0]-half['h'][0]).max()) < 1e-10
    a, r = reconstruct(h, mask)
    a0, r0 = reconstruct(half['h'], mask); a1, r1 = reconstruct(full['h'], mask)
    assert float(abs(a[0]-a1).max()) < 1e-10 and float(abs(r[0]-r1).max()) < 1e-10
    for n in (5, 10):
        assert float(abs(a[n]-a0).max()) < 1e-10 and float(abs(r[n]-r0).max()) < 1e-10
    # 两层参考gate固定后的∆b恰好为线性项；不宣称跨源状态的R为零。
    return a, r
