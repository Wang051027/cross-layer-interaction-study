"""由低阶激活预测 block3 的开关、输出和定向干预。"""
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as functional

import dsprites_triple_rule as rule


BUDGETS = (0, 4, 32)
ARMS = ("target", "random1", "random2")
LAYERS = ("first", "second", "both")
POINTS = 64


class Pair(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor


@dataclass
class Endpoint:
    middle: torch.Tensor
    z: Pair
    gates: Pair
    score: torch.Tensor


def score(net, hidden, classes):
    recipient, donor = classes
    logits = net.readout(hidden.mean(dim=(-1, -2)))
    return logits[..., donor] - logits[..., recipient]


def second_input(net, activated, middle):
    block = net.block3
    return block.bn2(block.conv2(activated)) + block.shortcut(middle)


def endpoint(net, middle, classes, first=None, second=None):
    block = net.block3
    z1 = block.bn1(block.conv1(middle))
    d1 = z1 > 0 if first is None else first
    z2 = second_input(net, z1 * d1, middle)
    d2 = z2 > 0 if second is None else second
    return Endpoint(middle, Pair(z1, z2), Pair(d1, d2), score(net, z2 * d2, classes))


def majority(field):
    result = []
    for triple in rule.TRIPLES:
        rows = [rule.LOW.index(i) for i in rule.corners(triple)[:7]]
        result.append((field[rows] > 0).sum(0) >= 4)
    return torch.stack(result)


def packed(gates):
    fields = torch.stack(gates, dim=1).flatten(start_dim=2).cpu().numpy()
    return np.packbits(fields, axis=-1, bitorder="little")


def forecast(net, source, donor, classes, strength, fit):
    # low_states 提供旧规则的中间场。单独记录低阶尾部的两个预激活。
    low = rule.low_states(net, source, donor, strength, fit["first"])
    low_source = rule.mixed_sources(source, donor, rule.LOW, strength)
    observed = endpoint(net, net.block2(low_source), classes)
    qz = Pair(*(rule.order_two(field) for field in observed.z))
    proposed = Pair(*(field > 0 for field in qz))
    vote = Pair(*(majority(field) for field in observed.z))
    base = Pair(*(field[:1] for field in observed.gates))
    quadr_output = rule.order_two(observed.score)
    recipient, donor_class = classes
    jac = rule.empty_jacobian(net, low.base_h, list(range(32)))
    jac = jac[donor_class] - jac[recipient]
    drive = rule.propagated(net, low.first_residual, fit["first"], list(range(32)))
    correction = (low.quad_v + drive).relu() - low.quad_h
    previous = correction.new_tensor(fit["class_logits"][recipient])
    predictions = {"q_output": quadr_output, "class_mean": previous[:, donor_class] - previous[:, recipient]}
    for budget in BUDGETS:
        choose = torch.zeros_like(correction[:1])
        channels = {0: [], 4: fit["middle"], 32: list(range(32))}[budget]
        choose[:, channels] = 1
        change = choose * correction
        middle = low.quad_h + change
        natural = endpoint(net, middle, classes)
        projected = endpoint(net, middle, classes, *proposed)
        predictions[f"fixed_{budget}"] = (change * jac).sum((1, 2, 3))
        predictions[f"live_{budget}"] = natural.score - quadr_output
        predictions[f"predicted_{budget}"] = projected.score - quadr_output
        if budget == 0:
            torch.testing.assert_close(natural.z.first, qz.first, rtol=1e-9, atol=1e-10)
        if budget == 4:
            primary = projected
            predictions["first_4"] = endpoint(net, middle, classes, first=proposed.first).score - quadr_output
            predictions["second_4"] = endpoint(net, middle, classes, second=proposed.second).score - quadr_output
            predictions["empty_masks_4"] = endpoint(net, middle, classes, *base).score - quadr_output
            predictions["majority_4"] = endpoint(net, middle, classes, *vote).score - quadr_output
    return predictions, primary, base, vote, qz.second


def point_lists(net, state, base, classes, seed, strength, case):
    """从预测翻转池选点，随机对照匹配每层预测开启和关闭的数量。"""
    r, d = classes
    height, width = state.middle.shape[-2:]
    readout = (net.readout.weight[d] - net.readout.weight[r])[None, :, None, None] / (height * width)
    block = net.block3
    bn_slope = block.bn2.weight / (block.bn2.running_var + block.bn2.eps).sqrt()
    upstream = functional.conv_transpose2d(readout * state.gates.second * bn_slope[None, :, None, None],
                                           block.conv2.weight, padding=block.conv2.padding)
    scores = Pair(state.z.first * upstream, state.z.second * readout)
    lists = np.full((len(rule.TRIPLES), len(ARMS), 2, POINTS), -1, dtype=np.int32)
    for layer, (field, gates, empty) in enumerate(zip(scores, state.gates, base)):
        magnitudes = np.round(field.abs().flatten(1).cpu().numpy(), 11)
        guesses = gates.flatten(1).cpu().numpy()
        changes = (gates != empty).flatten(1).cpu().numpy()
        for index, triple in enumerate(rule.TRIPLES):
            pool = np.flatnonzero(changes[index])
            order = np.lexsort((pool, -magnitudes[index, pool]))
            selected = pool[order[:POINTS]]
            lists[index, 0, layer, :len(selected)] = selected
            for arm in (1, 2):
                rng = np.random.default_rng([seed, int(strength * 100), case, triple, layer, arm])
                picks = []
                for opened in (False, True):
                    available = pool[guesses[index, pool] == opened]
                    count = int(np.count_nonzero(guesses[index, selected] == opened))
                    picks.extend(rng.choice(available, size=count, replace=False))
                lists[index, arm, layer, :len(picks)] = sorted(picks)
    return lists


def selection_masks(lists, state):
    n, arms = lists.shape[:2]
    dims = state.middle.shape[1:]
    selected = np.zeros((n, arms, 2, int(np.prod(dims))), dtype=bool)
    rows, groups, layers, pos = np.nonzero(lists >= 0)
    selected[rows, groups, layers, lists[rows, groups, layers, pos]] = True
    masks = torch.as_tensor(selected, device=state.middle.device)
    return Pair(*(masks[:, :, layer].reshape(n, arms, *dims) for layer in (0, 1)))


def patch_effects(net, state, base, classes, lists, forecast_second=None):
    """每次阻断的有符号效应等于干预后端点得分减自然端点得分。"""
    mask = selection_masks(lists, state)
    g1 = torch.where(mask.first, base.first[:, None], state.gates.first[:, None])
    first = state.z.first[:, None] * g1
    middle = state.middle[:, None].expand_as(first)
    after = second_input(net, first.flatten(0, 1), middle.reshape_as(first).flatten(0, 1)).reshape_as(first)
    before = state.z.second[:, None]
    if forecast_second is None:
        after_gates = after > 0
    else:
        # 低阶外推负责初始开关。第一处干预的预激活增量经第二卷积传递。
        after_gates = (forecast_second[:, None] + (after - before)) > 0
    second_only = torch.where(mask.second, base.second[:, None], state.gates.second[:, None])
    together = torch.where(mask.second, base.second[:, None], after_gates)
    outputs = [score(net, after * after_gates, classes), score(net, before * second_only, classes),
               score(net, after * together, classes)]
    return torch.stack(outputs, dim=-1) - state.score[:, None, None]


def reference_patches(net, state, base, classes, lists):
    """逐名单直接改写 ReLU 输出，并让未选位置执行原模型的 ReLU。"""
    outputs = []
    block = net.block3
    for index, middle in enumerate(state.middle):
        rows = []
        z1 = block.bn1(block.conv1(middle[None]))
        for group in lists[index]:
            first_ids, second_ids = (points[points >= 0].astype(np.int64) for points in group)
            first = block.mid(middle[None]).expand(3, -1, -1, -1).clone()
            first_value = (z1 * base.first).flatten()[first_ids]
            for mode in (0, 2):
                first[mode].view(-1)[first_ids] = first_value
            residual = block.shortcut(middle[None]).expand_as(first)
            z2 = block.bn2(block.conv2(first)) + residual
            result = block.finish(first, residual)
            for mode in (1, 2):
                result[mode].view(-1)[second_ids] = (z2[mode] * base.second[0]).flatten()[second_ids]
            rows.append(score(net, result, classes) - state.score[index])
        outputs.append(torch.stack(rows))
    return torch.stack(outputs)
