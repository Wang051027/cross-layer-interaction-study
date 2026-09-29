"""跨层交互的逐层计算、固定门控分解与中介恢复/移植。

源点为 block1 的完整输出；block2/block3 的主分支和旁路均自然传播。
这里用原始、未融合的冻结网络取得参考。分解是代数诊断，不等于唯一因果路径。
所有模型计算均为评价模式；本文件不训练权重，也不自行读测试数据。
"""
from __future__ import annotations
import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STAGES = ('b2_mid', 'b2_out', 'b3_mid', 'b3_out')


def design(q=4):
    """返回二元掩码和条件成对差分索引；四列依次是 TAB、TA、TB、T。

    q=4 有 16 个组合、6 对坐标，每对有 4 种其余坐标背景，共 24 个差分。
    1 表示衰减；T=0 的六个差分另作原点背景补充分析。
    """
    bits = ((np.arange(1 << q)[:, None] >> np.arange(q)) & 1).astype(float)
    stencils, metadata = [], []
    for i, j in itertools.combinations(range(q), 2):
        for background in range(1 << q):
            if background & ((1 << i) | (1 << j)):
                continue
            stencils.append([background | (1 << i) | (1 << j),
                             background | (1 << i), background | (1 << j), background])
            metadata.append([i, j, background])
    return bits, np.asarray(stencils), np.asarray(metadata)


def pair_difference(values, stencils):
    """沿第 0 维的组合轴取四项差分，后续维度可以是图像状态或 logits。"""
    ab, a, b, base = stencils.T
    return values[ab] - values[a] - values[b] + values[base]


def target(values, label):
    """固定十类标量目标：真实类别 logit 减去其余九类平均；不改用概率。"""
    return (10 * values[..., int(label)] - values.sum(-1)) / 9


@torch.no_grad()
def source_activation(model, images):
    """执行冻结前缀，输出 [批次,16,28,28]，作为全部实验的同一个源层。"""
    return model.block1(model.stem(images))


def changed_sources(source, nodes, bits, alpha):
    """复制单图源状态，独立衰减选中坐标；原状态与未选坐标保持原值。"""
    altered = source.expand(len(bits), -1, -1, -1).clone()
    altered.flatten(1)[:, nodes] *= 1 - alpha * bits
    return altered


def tail_from_middle(model, middle):
    """block2 输出是完整图切口；从这里原样运行 block3、均值池化和读出。"""
    return model.readout(model.block3(middle).mean((2, 3)))


@torch.no_grad()
def trace(model, altered):
    """记录源干预后的四个预激活/激活和输出，所有残差支路随输入自然变化。"""
    b2, b3 = model.block2, model.block3
    z21 = b2.bn1(b2.conv1(altered))
    h21 = F.relu(z21)
    # block2 是降采样块，旁路为 Conv+BN；不能固定为未干预值。
    z22 = b2.bn2(b2.conv2(h21)) + b2.shortcut(altered)
    h22 = F.relu(z22)
    z31 = b3.bn1(b3.conv1(h22))
    h31 = F.relu(z31)
    z32 = b3.bn2(b3.conv2(h31)) + b3.shortcut(h22)
    h32 = F.relu(z32)
    output = model.readout(h32.mean((2, 3)))
    return {'source': altered, 'z': (z21, z22, z31, z32),
            'h': (h21, h22, h31, h32), 'output': output}


@torch.no_grad()
def original_forward(model, image, nodes, bits, alpha, mode=None, patch_nodes=None):
    """独立参考：在原 model.forward 的模块出口加钩子，直接修改源或中介。

    与 trace 的手写层序列独立；每次调用结束必定移除钩子。
    恢复：M=该图原 block2 输出；移植：仅 M 取自然干预值，其余取原值。
    """
    handles = []
    baseline_middle = []
    if mode is not None:
        h = model.block2.register_forward_hook(lambda m, a, o: baseline_middle.append(o.detach().clone()))
        model(image)
        h.remove()
    def source_hook(module, args, output):
        """只修改指定源坐标，直接让原 forward 继续执行。"""
        changed = output.clone()
        changed.flatten(1)[:, nodes] *= 1 - alpha * bits
        return changed
    def middle_hook(module, args, output):
        """在完整 block2 出口替换状态，作为批量修补计算的独立对照。"""
        baseline = baseline_middle[0].expand_as(output)
        if mode == 'restore':
            patched = output.clone()
            patched.flatten(1)[:, patch_nodes] = baseline.flatten(1)[:, patch_nodes]
        elif mode == 'transplant':
            patched = baseline.clone()
            patched.flatten(1)[:, patch_nodes] = output.flatten(1)[:, patch_nodes]
        else:
            raise ValueError(mode)
        return patched
    try:
        handles.append(model.block1.register_forward_hook(source_hook))
        if mode is not None:
            handles.append(model.block2.register_forward_hook(middle_hook))
        return model(image.expand(len(bits), -1, -1, -1)).detach()
    finally:
        for handle in handles:
            handle.remove()


def conv_bn_linear(conv, bn, values):
    """仅应用 Conv+评价 BN 的线性部分；常数项在四项差分中严格相消。"""
    assert not bn.training
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    out = F.conv2d(values, conv.weight, None, conv.stride, conv.padding,
                   conv.dilation, conv.groups)
    return out * scale[None, :, None, None]


def shortcut_linear(block, values):
    """残差旁路对差分的线性映射，分别处理恒等旁路与降采样 Conv+BN。"""
    if isinstance(block.shortcut, nn.Identity):
        return values
    return conv_bn_linear(block.shortcut[0], block.shortcut[1], values)


@torch.no_grad()
def decomposition(model, traced, stencils):
    """准确分开固定门控传播与非线性修正，并将四阶段修正传到输出。

    D0 取该图无干预状态，不随其他坐标背景 T 改变。N 是四种 ReLU
    余项的差分；来源分量可互相抵消，不能当作四个互斥的因果百分比。
    """
    z, h = traced['z'], traced['h']
    gates = tuple((value[:1] > 0).to(value.dtype) for value in z)
    interaction = tuple(pair_difference(value, stencils) for value in h)
    zi = tuple(pair_difference(value, stencils) for value in z)
    generated = tuple(pair_difference(F.relu(value) - gate * value, stencils)
                      for value, gate in zip(z, gates))
    source_i = pair_difference(traced['source'], stencils)
    b2, b3 = model.block2, model.block3
    linear_z = (
        conv_bn_linear(b2.conv1, b2.bn1, source_i),
        conv_bn_linear(b2.conv2, b2.bn2, interaction[0]) + shortcut_linear(b2, source_i),
        conv_bn_linear(b3.conv1, b3.bn1, interaction[1]),
        conv_bn_linear(b3.conv2, b3.bn2, interaction[2]) + shortcut_linear(b3, interaction[1]))
    checks = {'source_interaction': float(source_i.abs().max()),
              'affine_identity': max(float((a-b).abs().max()) for a, b in zip(zi, linear_z)),
              'relu_identity': max(float((a-d*b-n).abs().max()) for a, d, b, n in
                                   zip(interaction, gates, zi, generated))}
    # 四个槽位分别存四个 ReLU 阶段产生的修正。源干预本身为加性，所以无第五项。
    parts = torch.zeros((4,) + generated[0].shape, dtype=z[0].dtype, device=z[0].device)
    parts[0] = generated[0]
    def conv_parts(value, conv, bn):
        """暂合并来源轴与查询轴，卷积后还原两轴，避免来源互相混合。"""
        shape = value.shape
        flat = conv_bn_linear(conv, bn, value.flatten(0, 1))
        return flat.reshape(shape[:2] + flat.shape[1:])
    p2 = gates[1][None] * conv_parts(parts, b2.conv2, b2.bn2)
    p2[1] += generated[1]
    p3 = gates[2][None] * conv_parts(p2, b3.conv1, b3.bn1)
    p3[2] += generated[2]
    # block3 的旁路把 block2 输出的交互一并送到最后 ReLU 前。
    bypass = shortcut_linear(b3, p2.flatten(0, 1)).reshape_as(p2)
    p4 = gates[3][None] * (conv_parts(p3, b3.conv2, b3.bn2) + bypass)
    p4[3] += generated[3]
    contributions = p4.mean((-2, -1)) @ model.readout.weight.T
    final_i = pair_difference(traced['output'], stencils)
    checks['output_decomposition'] = float((contributions.sum(0)-final_i).abs().max())
    layer = []
    for state, pre, gate, inter, incoming, extra in zip(h, z, gates, interaction, zi, generated):
        flips = ((pre > 0) != (pre[:1] > 0)).any(0)
        layer.append([float(inter.square().mean().sqrt()),
                      float((gate*incoming).square().mean().sqrt()),
                      float(extra.square().mean().sqrt()),
                      int(flips.sum()), int(flips.numel())])
    return contributions, np.asarray(layer), checks


@torch.no_grad()
def patch_outputs(model, middle, selections, chunk=128):
    """批量做中介恢复/移植，返回 [选择方案,操作2,组合16,类别10]。

    所有方案使用同一张图的无源干预中介作基线。这个人工混合内部状态
    对应限定的计算干预，不保证属于自然图像分布。
    """
    batches = []
    baseline = middle[:1].expand_as(middle)
    for nodes in selections:
        restored = middle.clone()
        restored.flatten(1)[:, nodes] = baseline.flatten(1)[:, nodes]
        transplanted = baseline.clone()
        transplanted.flatten(1)[:, nodes] = middle.flatten(1)[:, nodes]
        batches.extend((restored, transplanted))
    stacked = torch.cat(batches)
    output = torch.cat([tail_from_middle(model, part) for part in stacked.split(chunk)])
    return output.reshape(len(selections), 2, len(middle), 10)


@torch.no_grad()
def reachable_middle(model, nodes, device):
    """用连接拓扑界定 block2 输出候选池；不按测试响应或当前 ReLU 符号筛选。

    卷积权重非零的位置视为可能连接。两条残差路径取并集，包含所有结构
    可达坐标；这是保守支持集，不声称每个坐标在当前图上都会产生效应。
    """
    basis = torch.zeros((1, 16, 28, 28), dtype=torch.float64, device=device)
    basis.flatten()[nodes] = 1
    def support_conv(conv, x):
        """传播可能受影响的支持集，不读取当前输入的门控。"""
        return (F.conv2d(x, (conv.weight != 0).to(x.dtype), None,
                        conv.stride, conv.padding, conv.dilation, conv.groups) > 0).to(x.dtype)
    block = model.block2
    main = support_conv(block.conv2, support_conv(block.conv1, basis))
    skip = basis if isinstance(block.shortcut, nn.Identity) else support_conv(block.shortcut[0], basis)
    return torch.nonzero((main+skip).flatten() > 0).flatten().cpu().numpy()


@torch.no_grad()
def select_source_nodes(model, images, device):
    """只按固定训练上下文均值，在源层中央 3×3 窗口选择四个坐标。"""
    total = torch.zeros((16, 28, 28), dtype=torch.float64, device=device)
    for batch in images.split(32):
        total += source_activation(model, batch.to(device)).sum(0)
    ids = np.arange(16*28*28).reshape(16, 28, 28)[:, 13:16, 13:16].ravel()
    means = (total / len(images)).cpu().numpy().ravel()
    return ids[np.argsort(-means[ids], kind='stable')[:4]]


def learn_selection(model, images, labels, nodes, bits, stencils, device):
    """在开发图像上拟合中介排序，不使用评价图像和半强度结果。

    候选分数为平均 |目标梯度 × 条件成对交互|。单独效应基线使用相同
    梯度，但乘单坐标干预变化；另记录激活大小，供随机组近似匹配。
    """
    features = np.zeros((3, 32*14*14), dtype=np.float64)
    single_ids = [1, 2, 4, 8]
    for image, label in zip(images, labels):
        with torch.no_grad():
            source = source_activation(model, image[None].to(device))
            traced = trace(model, changed_sources(source, nodes, bits, 1.))
            middle = traced['h'][1]
        baseline = middle[:1].detach().clone().requires_grad_(True)
        gradient = torch.autograd.grad(target(tail_from_middle(model, baseline), int(label)).sum(), baseline)[0]
        with torch.no_grad():
            inter = pair_difference(middle, stencils)
            single = middle[single_ids] - middle[:1]
            features[0] += (inter*gradient).abs().mean(0).flatten().cpu().numpy()
            features[1] += (single*gradient).abs().mean(0).flatten().cpu().numpy()
            features[2] += middle[0].abs().flatten().cpu().numpy()
    return features / len(images)


def selections_from_features(features, pool, seed, budgets=(8, 16, 32)):
    """固定三种排序及四个近似匹配随机对照，返回方案和匹配质量记录。

    随机对照在开发集的激活大小、单独效应两个特征上近邻匹配；从最近
    五个尚未使用坐标中抽一个。排除该预算的候选 M，且记录匹配距离。
    """
    orders = [pool[np.argsort(-score[pool], kind='stable')] for score in features]
    matching = features[[2, 1]][:, pool].T.copy()
    scales = np.array([np.median(col[col > 0]) if np.any(col > 0) else 1.
                       for col in matching.T])
    matching = np.log1p(matching / scales)
    matching = (matching-matching.mean(0)) / np.maximum(matching.std(0), 1e-12)
    index = {int(node): i for i, node in enumerate(pool)}
    result = []
    for budget in budgets:
        assert len(pool) >= 2*budget+5
        for method, order in zip(('interaction', 'single_effect', 'activation'), orders):
            result.append(dict(method=method, budget=budget, nodes=order[:budget].tolist()))
        chosen = orders[0][:budget]
        for repeat in range(4):
            rng = np.random.default_rng(seed + 2026092800 + budget*10 + repeat)
            available = set(map(int, pool)) - set(map(int, chosen))
            matched, distances = [], []
            for node in chosen:
                candidates = np.array(sorted(available), dtype=int)
                delta = matching[[index[int(v)] for v in candidates]] - matching[index[int(node)]]
                distance = np.square(delta).sum(1)
                best = np.argsort(distance, kind='stable')[:5]
                pick = int(rng.choice(best)); selected = int(candidates[pick])
                available.remove(selected); matched.append(selected); distances.append(float(np.sqrt(distance[pick])))
            result.append(dict(method=f'matched_{repeat}', budget=budget, nodes=matched,
                               mean_match_distance=float(np.mean(distances)), max_match_distance=max(distances)))
    return result
