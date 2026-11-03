"""用低阶状态预测三组联合响应。训练和打分在独立入口中进行。

Q 是由七个真子集构成的二阶预测。这里仅重算少量选定通道的阈值和
卷积校正。最终输出预测使用空组处的读出导数。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np
import dsprites_teacher as teacher
import torch
from torch.nn import functional as functional
from cross_layer_mechanism import source_activation, tail_from_middle
from shape_donor_mechanism import replace_groups

TRIPLES = (7, 11, 13, 14)
LOW = tuple(i for i in range(16) if i.bit_count() <= 2)
METHODS = ('quadratic', 'rule', 'ridge', 'threshold_only', 'carry_only', 'class_mean')
PATCHES = ('chosen4', 'strongest1', 'weakest1', 'random0', 'random1')
EPS = 1e-9


def corners(triple):
    """空、三个单组、三组两两组合、三组全选；最后一个只在验证时读取。"""
    a, b, c = [g for g in (1, 2, 4, 8) if triple & g]
    return (0, a, b, c, a | b, a | c, b | c, triple)


def order_two(values):
    """轴0依LOW编号，返回对应四个三组目标的二阶预测。"""
    results = []
    for triple in TRIPLES:
        ids = [LOW.index(i) for i in corners(triple)[:7]]
        zero, a, b, c, ab, ac, bc = values[ids]
        results.append(ab + ac + bc - a - b - c + zero)
    return torch.stack(results)


def mixed_sources(recipient, donor, masks, strength):
    """沿用封存实验的四个互不重叠的源通道组。"""
    bits = np.array([[(mask >> bit) & 1 for bit in range(4)] for mask in masks])
    bits = torch.as_tensor(bits, dtype=recipient.dtype, device=recipient.device)
    return replace_groups(recipient, donor, bits, strength)


def as_numpy(tensor):
    return tensor.detach().cpu().numpy()


@torch.no_grad()
def source_bank(model, panel, cases, device):
    """只为指定一侧的图像计算前缀，不从这里取得自然输出。"""
    image_ids = sorted({int(i) for case in cases for i in panel['pairs'][case]})
    images = torch.as_tensor(panel['images'][image_ids, None], dtype=torch.float64, device=device)
    return dict(zip(image_ids, source_activation(model, images)))


def split_panel(panel):
    """划分单位是整个因素格；每格仅取一个循环方向，排除反向互补重用。"""
    factors, labels, pairs = (panel[k] for k in ('factors', 'labels', 'pairs'))
    sizes = sorted(set(factors[:, 2].tolist()))
    angles = sorted(set(factors[:, 3].tolist()))
    assert len(sizes) == 6 and len(angles) == 2 and len(pairs) == 72
    splits = {'development': [], 'test': []}
    images = {name: set() for name in splits}
    for case, (r, d) in enumerate(pairs):
        assert np.array_equal(factors[r, 2:], factors[d, 2:])
        if int(labels[d]) != (int(labels[r]) + 1) % 3:
            continue
        size, angle = map(int, factors[r, 2:4])
        name = 'development' if (sizes.index(size) + angles.index(angle)) % 2 == 0 else 'test'
        splits[name].append(case)
        images[name].update([int(r), int(d)])
    assert all(len(cases) == 18 for cases in splits.values())
    assert images['development'].isdisjoint(images['test'])
    assert len(images['development'] | images['test']) == 36
    used = {tuple(pairs[i]) for cases in splits.values() for i in cases}
    assert all((d, r) not in used for r, d in used)
    return splits


def empty_jacobian(model, empty, channels):
    """只在该配对的空组状态求三个类别导数；三组真实状态不参与读出。"""
    with torch.enable_grad():
        leaf = empty[None].detach().requires_grad_(True)
        outputs = tail_from_middle(model, leaf)[0]
        grads = [torch.autograd.grad(outputs[k], leaf, retain_graph=k < 2)[0][0, channels]
                 for k in range(3)]
    return torch.stack(grads).detach()


@dataclass
class LowStates:
    first_residual: torch.Tensor
    base_v: torch.Tensor
    base_h: torch.Tensor
    quad_v: torch.Tensor
    quad_h: torch.Tensor
    quad_logits: torch.Tensor
    empty_logits: torch.Tensor


@torch.no_grad()
def low_states(model, r, d, strength, first_channels):
    """唯一的预测特征入口。只执行11个低阶状态，同一三组只取其七个角。"""
    inputs = mixed_sources(r, d, LOW, strength)
    block = model.block2
    z = block.bn1(block.conv1(inputs))
    u = z.relu()
    v = block.bn2(block.conv2(u)) + block.shortcut(inputs)
    h = v.relu()
    logits = tail_from_middle(model, h)
    z = z[:, first_channels]
    predicted_z = []
    for triple in TRIPLES:
        a, b, c = [LOW.index(g) for g in (1, 2, 4, 8) if g & triple]
        predicted_z.append(z[a] + z[b] + z[c] - 2*z[0])
    first_residual = torch.stack(predicted_z).relu() - order_two(u[:, first_channels])
    return LowStates(first_residual, v[0], h[0], order_two(v), order_two(h),
                     order_two(logits), logits[0])


def propagated(model, first_residual, first_channels, middle_channels):
    """只保留K到C的卷积核，并把推理BN斜率乘入；三阶差分消去偏置。"""
    block = model.block2
    scale = block.bn2.weight / torch.sqrt(block.bn2.running_var + block.bn2.eps)
    kernel = block.conv2.weight[middle_channels][:, first_channels]
    kernel = kernel * scale[middle_channels, None, None, None]
    return functional.conv2d(first_residual, kernel, padding=1)


def middle_rules(low, drive, channels):
    """物理规则、两个消融，以及使用相同低阶观测拟合的线性控制。"""
    qv, qh = low.quad_v[:, channels], low.quad_h[:, channels]
    v0 = low.base_v[channels][None].expand_as(qv)
    h0 = low.base_h[channels][None].expand_as(qh)
    features = torch.stack((v0, h0, qv-v0, qh-h0, drive), dim=-1)
    physical = (qv + drive).relu() - qh
    threshold_only = qv.relu() - qh
    carry_only = (v0 > 0) * drive
    return physical, threshold_only, carry_only, features


def rank_four(scores):
    """同分时先选物理编号小的通道。"""
    return np.lexsort((np.arange(32), -np.asarray(scores)))[:4].tolist()


def fit_site_ridge(features, targets):
    """对每个中间通道共享空间回归系数，无图像或坐标标识特征。"""
    weight, bias = [], []
    for channel in range(4):
        x = features[:, channel].reshape(-1, 5)
        y = targets[:, channel].reshape(-1)
        mean = x.mean(0)
        sd = x.std(0)
        sd[sd < 1e-12] = 1.
        normalized = (x - mean) / sd
        gram = normalized.T @ normalized + 0.01*len(y)*np.eye(5)
        standardized = np.linalg.solve(gram, normalized.T @ (y - y.mean()))
        raw = standardized / sd
        weight.append(raw.tolist())
        bias.append(float(y.mean() - mean @ raw))
    return weight, bias


@torch.no_grad()
def fit_model(model, panel, development, device, seed):
    """只在开发格全强度拟合K、C、线性对照和类别均值。"""
    bank = source_bank(model, panel, development, device)
    cached = []
    energy = np.zeros(32)
    all_channels = list(range(32))
    for case in development:
        r, d = panel['pairs'][case]
        low = low_states(model, bank[r], bank[d], 1., all_channels)
        truth_h = model.block2(mixed_sources(bank[r], bank[d], TRIPLES, 1.))
        target = truth_h - low.quad_h
        truth_logits = tail_from_middle(model, truth_h) - low.quad_logits
        cached.append((int(panel['labels'][r]), int(panel['labels'][d]), low, target, truth_logits))
        energy += as_numpy(low.first_residual.square().sum((0, 2, 3)))
    first = rank_four(energy)
    gain = np.zeros(32)
    for _, _, low, target, _ in cached:
        drive = propagated(model, low.first_residual[:, first], first, all_channels)
        estimated = (low.quad_v + drive).relu() - low.quad_h
        gain += as_numpy((2*estimated*target-estimated.square()).sum((0, 2, 3)))
    middle = rank_four(gain)
    features, targets, predicted_strength = [], [], []
    classes = [[] for _ in range(3)]
    for recipient, donor, low, target, truth_logits in cached:
        drive = propagated(model, low.first_residual[:, first], first, middle)
        correction, _, _, x = middle_rules(low, drive, middle)
        jac = empty_jacobian(model, low.base_h, middle)
        y = torch.einsum('tchw,kchw->tk', correction, jac)
        predicted_strength.extend(as_numpy(abs(y[:, donor]-y[:, recipient])).tolist())
        features.append(as_numpy(x))
        targets.append(as_numpy(target[:, middle]))
        classes[recipient].append(as_numpy(truth_logits))
    weights, intercept = fit_site_ridge(np.concatenate(features), np.concatenate(targets))
    outside = np.setdiff1d(np.arange(32), middle)
    rng = np.random.default_rng(seed + 2026103100)
    controls = [rng.choice(outside, size=4, replace=False).tolist() for _ in range(2)]
    return dict(first=first, middle=middle, energy=energy.tolist(), gain=gain.tolist(),
                ridge_weights=weights, ridge_bias=intercept,
                class_logits=[np.mean(rows, axis=0).tolist() for rows in classes],
                strong_threshold=float(np.median(predicted_strength)), random_channels=controls)


@torch.no_grad()
def predict_pair(model, r, d, recipient, donor, strength, fit):
    """先作出四个三组预测以及所有阻断名单，再由另一阶段打开真值。"""
    first, middle = fit['first'], fit['middle']
    low = low_states(model, r, d, strength, first)
    drive = propagated(model, low.first_residual, first, middle)
    rule, threshold, carry, x = middle_rules(low, drive, middle)
    weights = x.new_tensor(fit['ridge_weights'])[None, :, None, None, :]
    bias = x.new_tensor(fit['ridge_bias'])[None, :, None, None]
    ridge = (x * weights).sum(-1) + bias
    corrections = torch.stack((rule, ridge, threshold, carry), dim=1)
    jac = empty_jacobian(model, low.base_h, middle)
    output = torch.einsum('tmchw,kchw->tmk', corrections, jac)
    class_logits = x.new_tensor(fit['class_logits'][recipient])
    output = torch.cat((torch.zeros_like(output[:, :1]), output, class_logits[:, None]), dim=1)
    per_channel = torch.einsum('tchw,kchw->tck', rule, jac)
    scores = as_numpy(per_channel[:, :, donor] - per_channel[:, :, recipient])
    physical_ids = np.array(middle)
    patch_ids = np.full((4, 5, 4), -1, dtype=np.int16)
    for t, values in enumerate(scores):
        patch_ids[t, 0] = middle
        patch_ids[t, 1, 0] = physical_ids[np.lexsort((physical_ids, -abs(values)))[0]]
        patch_ids[t, 2, 0] = physical_ids[np.lexsort((physical_ids, abs(values)))[0]]
        patch_ids[t, 3:] = fit['random_channels']
    return dict(qh=as_numpy(low.quad_h), h0=as_numpy(low.base_h),
                qlogits=as_numpy(low.quad_logits), empty_logits=as_numpy(low.empty_logits),
                middle_corrections=as_numpy(corrections), output_corrections=as_numpy(output),
                channel_logits=as_numpy(per_channel), patch_ids=patch_ids)


@torch.no_grad()
def measure_pair(model, r, d, strength, patch_ids):
    """使用原模型执行真实干预。阻断和只保留均在全部八个角上操作。"""
    masks = tuple(range(15))
    inputs = mixed_sources(r, d, masks, strength)
    h = model.block2(inputs)
    natural = tail_from_middle(model, h)
    patches = []
    for t, triple in enumerate(TRIPLES):
        states = h[list(corners(triple))]
        empty = h[:1].expand_as(states)
        fields = []
        for ids in patch_ids[t]:
            chosen = ids[ids >= 0].tolist()
            blocked, kept = states.clone(), empty.clone()
            blocked[:, chosen] = empty[:, chosen]
            kept[:, chosen] = states[:, chosen]
            fields.extend((blocked, kept))
        patches.append(torch.stack([tail_from_middle(model, f) for f in fields]).reshape(5, 2, 8, 3))
    return dict(true_middle=as_numpy(h[list(TRIPLES)]), natural_logits=as_numpy(natural),
                patch_logits=as_numpy(torch.stack(patches)))
