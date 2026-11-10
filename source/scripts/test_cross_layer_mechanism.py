"""用已知差分和模型前向钩子核对跨层干预。"""
import json
import numpy as np
import torch
from torch.nn import functional as F
from cross_layer_mechanism import (design, pair_difference, source_activation,
    changed_sources, trace, decomposition, patch_outputs, original_forward, tail_from_middle)


def known_answers():
    """检查加性、阈值协作、背景相关三阶作用、隐藏交互抵消四类已知例子。"""
    bits, stencil, meta = design()
    linear = bits @ np.arange(1., 5.) + 7
    assert np.max(np.abs(pair_difference(linear, stencil))) == 0
    threshold = np.maximum(1.5-bits[:, 0]-bits[:, 1], 0)
    pair = pair_difference(threshold, stencil)
    assert np.all(pair[(meta[:, 0] == 0) & (meta[:, 1] == 1)] == .5)
    assert np.all(pair[(meta[:, 0] != 0) | (meta[:, 1] != 1)] == 0)
    cubic = np.maximum(bits[:, :3].sum(1)-2, 0)
    ci = pair_difference(cubic, stencil)
    assert ci[np.flatnonzero((meta == [0, 1, 0]).all(1))[0]] == 0
    assert ci[np.flatnonzero((meta == [0, 1, 4]).all(1))[0]] == 1
    hidden = np.stack([threshold, threshold], axis=1)
    assert np.abs(pair_difference(hidden, stencil)).max() == .5
    assert np.abs(pair_difference(hidden @ np.array([1., -1.]), stencil)).max() == 0
    return dict(status='PASS', controls=['affine zero interaction', 'threshold pair = 0.5',
        'third-order effect visible only in conditional background', 'hidden interactions cancel at readout'])


def gradient_projection_check(model, traced, parts, stencil):
    """用自动微分独立检查四阶段代数分量的输出投影。

    每个阶段将其基线激活视为独立输入，保留原网络剩余计算与基线旁路；
    与手写 Conv+BN 差分传播是不同实现。这里只核验代数，不解释为中介实验。
    """
    h, z = traced['h'], traced['z']
    v = torch.linspace(-1., 1., 10, dtype=h[0].dtype, device=h[0].device)
    errors = []
    for stage in range(4):
        state = h[stage][:1].detach().clone().requires_grad_(True)
        if stage == 0:
            b2 = model.block2
            middle = F.relu(b2.bn2(b2.conv2(state)) + b2.shortcut(traced['source'][:1]))
            out = tail_from_middle(model, middle)
        elif stage == 1:
            out = tail_from_middle(model, state)
        elif stage == 2:
            b3 = model.block3
            last = F.relu(b3.bn2(b3.conv2(state)) + b3.shortcut(h[1][:1]))
            out = model.readout(last.mean((2, 3)))
        else:
            out = model.readout(state.mean((2, 3)))
        gradient = torch.autograd.grad((out*v).sum(), state)[0]
        gate = (z[stage][:1] > 0).to(state.dtype)
        remainder = F.relu(z[stage]) - gate*z[stage]
        generated = pair_difference(remainder, stencil)
        direct = (generated*gradient).flatten(1).sum(1)
        errors.append(float((direct-parts[stage]@v).abs().max().detach()))
    return max(errors)


def check_teacher(model, images, nodes, tolerance=1e-9):
    """用固定验证输入检查原模型、逐层展开、分解和三类修补的数值一致性。"""
    raw_bits, stencil, _ = design()
    bits = torch.as_tensor(raw_bits, dtype=torch.float64, device=images.device)
    maxima = {}
    def record(name, value):
        maxima[name] = max(maxima.get(name, 0.), float(value))
    for image in images:
        source = source_activation(model, image[None])
        for alpha in (1., .5):
            traced = trace(model, changed_sources(source, nodes, bits, alpha))
            reference = original_forward(model, image[None], nodes, bits, alpha)
            record('original_hook_vs_trace', (reference-traced['output']).abs().max())
            parts, _, checks = decomposition(model, traced, stencil)
            for key, value in checks.items(): record(key, value)
            record('autograd_projection', gradient_projection_check(model, traced, parts, stencil))
            full = np.arange(traced['h'][1][0].numel())
            subset = np.array([10, 300, 1000, 4000])
            outputs = patch_outputs(model, traced['h'][1], [[], full, subset])
            record('empty_restore_identity', (outputs[0, 0]-reference).abs().max())
            record('empty_transplant_baseline', (outputs[0, 1]-reference[:1]).abs().max())
            record('full_restore_baseline', (outputs[1, 0]-reference[:1]).abs().max())
            record('full_transplant_identity', (outputs[1, 1]-reference).abs().max())
            for op, name in enumerate(('restore', 'transplant')):
                patched = original_forward(model, image[None], nodes, bits, alpha, name, subset)
                record('patch_hook_'+name, (outputs[2, op]-patched).abs().max())
        unchanged = trace(model, changed_sources(source, nodes, bits, 0.))['output']
        record('zero_dose', (unchanged-unchanged[:1]).abs().max())
    assert max(maxima.values()) < tolerance, maxima
    return dict(status='PASS', tolerance=tolerance, images=len(images), errors=maxima)


if __name__ == '__main__':
    print(json.dumps(known_answers(), ensure_ascii=False, indent=2))
