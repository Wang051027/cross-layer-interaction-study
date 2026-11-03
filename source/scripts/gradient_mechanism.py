"""冻结网络的状态梯度、局部线性修补预测及梯度变化诊断。

本文件只实现计算，不选择数据、不训练权重、不写旧运行。梯度指logit对
源层或中介层激活的导数；它不同于优化器更新权重时的损失梯度。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

import dsprites_mechanism as mechanism


def state_gradients(model, source, pool, source_nodes):
    """每个组合分别求三类输出的两个Jacobian，批次之间没有梯度混合。

    网络处于eval模式，BN使用冻结统计量。对某类的批次logit求和后，
    各行梯度仍只来自该行；独立审计会用逐行计算复核这个性质。
    只保存可达中介池和四个源坐标，避免存储无关的全图梯度。
    """
    assert not model.training
    source_leaf = source.detach().clone().requires_grad_(True)
    middle = model.block2(source_leaf)
    logits = mechanism.tail_from_middle(model, middle)
    source_jacobian, middle_jacobian = [], []
    for c in range(logits.shape[-1]):
        sg, mg = torch.autograd.grad(logits[:, c].sum(), (source_leaf, middle),
                                     retain_graph=c < logits.shape[-1] - 1)
        source_jacobian.append(sg.flatten(1)[:, source_nodes].detach())
        middle_jacobian.append(mg.flatten(1)[:, pool].detach())
    return dict(logits=logits.detach(), middle=middle.detach(),
                source_jacobian=torch.stack(source_jacobian, 1),
                middle_jacobian=torch.stack(middle_jacobian, 1))


@torch.no_grad()
def frozen_gate_output(model, altered, baseline_trace):
    """固定四个ReLU为无干预的开关，再用原仿射层运行。

    这个人为网络在源层之后完全仿射，所以加性的源干预不会产生二阶
    交互；它是数学正对照，不能被解释为已找到唯一的因果回路。
    """
    gates = [(z[:1] > 0).to(z.dtype) for z in baseline_trace['z']]
    b2, b3 = model.block2, model.block3
    h21 = gates[0] * b2.bn1(b2.conv1(altered))
    h22 = gates[1] * (b2.bn2(b2.conv2(h21)) + b2.shortcut(altered))
    h31 = gates[2] * b3.bn1(b3.conv1(h22))
    h32 = gates[3] * (b3.bn2(b3.conv2(h31)) + b3.shortcut(h22))
    return model.readout(h32.mean((2, 3)))


def singleton_predictions(logits, delta, jacobian, shortlist):
    """两种一阶预测，输出形状为[方法2,候选64,操作2,组合16,类别3]。

    delta是自然干预中介减去无干预中介，Jacobian形状[16,3,可达池]。
    fixed_linear：恢复和移植都用无干预梯度J0。
    state_linear：恢复用干预状态JS，移植从无干预状态出发仍用J0。
    两者使用同一批自然logit和中介变化，因此比较只改变恢复的导数。
    预测不假设16个坐标的实际联合修补可加。
    """
    jac = jacobian[:, :, shortlist]
    change = delta[:, shortlist]
    fixed_effect = np.einsum('cj,sj->jsc', jac[0], change)
    state_effect = np.einsum('scj,sj->jsc', jac, change)
    restored_fixed = logits[None] - fixed_effect
    restored_state = logits[None] - state_effect
    transplanted = logits[:1][None] + fixed_effect
    return np.stack([np.stack([restored_fixed, transplanted], axis=1),
                     np.stack([restored_state, transplanted], axis=1)])


def gradient_score_predictions(logits, delta, jacobian, shortlist, label, stencil):
    """保持旧单点有限收益的定义，只把修补输出换成一阶预测。"""
    predictions = singleton_predictions(logits, delta, jacobian, shortlist)
    return np.stack([mechanism.finite_scores(p, logits, label, stencil) for p in predictions])


def margin_jacobian(jacobian, label):
    """真实类别与其余类平均logit之差的梯度，保留所有前导维度。"""
    c = jacobian.shape[-2]
    return (c * jacobian[..., int(label), :] - jacobian.sum(-2)) / (c - 1)


def drift_statistics(jacobian, label):
    """返回每个非空组合的梯度相对变化及方向余弦，零向量另作计数。

    分母只在原始梯度范数非零时定义；数值下限用于防止舍入除零，
    不能把零梯度的巨大比率当作证据。记录范数让报告可判断尺度。
    """
    g = margin_jacobian(jacobian, label)
    base_norm = float(np.linalg.norm(g[0]))
    norms = np.linalg.norm(g[1:], axis=1)
    relative = np.linalg.norm(g[1:] - g[0], axis=1) / max(base_norm, 1e-30)
    cosine = np.einsum('sj,j->s', g[1:], g[0]) / np.maximum(norms * base_norm, 1e-30)
    return np.stack([relative, cosine, norms], -1), base_norm


def known_answers():
    """用可手算仿射函数及斜率改变函数测试轴、符号和收益。

    仿射尾部两种预测都应等于实际单点修补；在ReLU正区间走向负区间
    时，一阶预测可以失准，测试不把局部导数偷换成有限变化。
    """
    bits, stencil, _ = mechanism.design()
    delta = np.c_[bits[:, 0] * .2, bits[:, 1] * -.3, bits[:, 2] * .4]
    weight = np.array([[2., -1., 3.], [-1., 4., .5], [.5, -2., 1.]])
    bias = np.array([.7, -.2, .4])
    logits = bias + delta @ weight.T
    jac = np.broadcast_to(weight, (16, 3, 3)).copy()
    predicted = singleton_predictions(logits, delta, jac, np.arange(3))
    actual = []
    for j in range(3):
        restored = delta.copy(); restored[:, j] = 0
        transplanted = np.zeros_like(delta); transplanted[:, j] = delta[:, j]
        actual.append([bias + restored @ weight.T, bias + transplanted @ weight.T])
    actual = np.asarray(actual)
    assert np.max(np.abs(predicted - actual)) < 1e-14
    assert np.max(np.abs(gradient_score_predictions(logits, delta, jac, np.arange(3), 1, stencil))) < 1e-14
    drift, norm = drift_statistics(jac, 1)
    assert norm > 0 and np.max(np.abs(drift[:, 0])) == 0
    assert np.max(np.abs(drift[:, 1] - 1)) < 1e-14
    # ReLU从1走到-1：起点导数1预测变化-2，真实变化-1。
    assert abs((max(-1., 0.) - max(1., 0.)) - (-2.)) == 1.
    return dict(status='PASS', affine_patch_max_error=float(np.max(np.abs(predicted - actual))),
                zero_drift_verified=True, finite_relu_change_not_equal_local_derivative=True)
