"""在原分类头测量末层整通道阻断后的八状态得分。"""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "remaining_routes_20261126"))
from experiment import Inputs, checksums, js, put, sha, stamp, verify, write_csv
from paths import third

SOURCE_RUN = "dsprites_remaining_routes_20261126"
SELECTION = "66c10e818444793fa8748f40930ad48bdc2118b3400ec3ea7f768f1358bf0746"
MEASUREMENT = "261b63bb334c5ed854dcc3f3933bb2005f4050504e3d79bb55da0caa05ae3c32"
BUDGETS = (1, 2, 4, 8, 14)
EPS = 1e-9


def previous(inputs):
    folder = inputs.previous.parent / SOURCE_RUN
    verify(folder, {"selection_seal.json": SELECTION, "measurement_seal.json": MEASUREMENT})
    verify(folder, js(folder / "selection_seal.json")["files"])
    protocol = js(folder / "protocol.json")
    if any(protocol[k] != v for k, v in inputs.spec.items() if k != "shared_code"):
        raise ValueError("Source, weights, split or gate forecast differs from the frozen experiment")
    # 公开版在来源表中逐文件登记原始与发布指纹。
    actual = inputs.spec["shared_code"]
    original = protocol["shared_code"]
    if actual != original:
        records = js(inputs.frozen.parents[2] / "PROVENANCE.json")["source_files"]
        released = {item["path"]: item for item in records}
        if actual.keys() != original.keys():
            raise ValueError("Shared code list changed")
        for name, digest in actual.items():
            entry = released["source/scripts/" + name]
            if (original[name], digest) != (entry["original_sha256"], entry["published_sha256"]):
                raise ValueError("Unexpected source pair: " + name)
    return folder


def final_maps(state, upstream):
    """轴依次为八个状态、U 自然或已阻断、末层通道、行、列。"""
    fixed = state.first.clone()
    fixed[:, upstream] = state.first[:1, upstream]
    after = state.last_activation(state.second_conv(fixed) + state.skip)
    return torch.stack((state.last, after), dim=1)


def routed_terms(state, maps):
    """开发时用线性读出直接计算各通道承接的 U 作用。"""
    difference = maps[:, 0] - maps[:, 1]
    return (third(difference).mean((-2, -1)) * state.weights).cpu().numpy()


def choose_model(seed, upstream, old_last, terms):
    outside = sorted(set(range(32)) - set(old_last))
    target = terms.sum(axis=1)
    scores = terms.T @ target / (target @ target)
    order = sorted(outside, key=lambda j: (-round(float(scores[j]), 10), j))
    pairs = list(combinations(outside, 2))
    first, second = np.array(pairs).T
    a, b = terms[:, first], terms[:, second]
    overlap = np.minimum(np.abs(a), np.abs(b))
    same = (np.where(a*b > 0, overlap, 0)).mean(0)
    opposite = (np.where(a*b < 0, overlap, 0)).mean(0)
    # 组合按编号字典序生成，舍入后同分时 argmax 取最前一对。
    joined = pairs[int(np.argmax(np.round(same, 10)))]
    cancelled = pairs[int(np.argmax(np.round(opposite, 10)))]
    rng = [np.random.default_rng([20261126, 28, seed, i]) for i in (1, 2)]
    selected = dict(upstream=upstream, old_last=old_last, outside=outside, order=order,
                    negative=[j for j in order if scores[j] < -EPS],
                    reinforce_pair=list(joined), cancel_pair=list(cancelled),
                    random=[g.permutation(outside).tolist() for g in rng])
    channels = [dict(seed=seed, rank=i, channel=j, projection_on_upstream=float(scores[j]))
                for i, j in enumerate(order, 1)]
    pairs_table = [dict(seed=seed, first=i, second=j, same_overlap=s, opposite_overlap=o)
                   for (i, j), s, o in zip(pairs, same, opposite)]
    return selected, channels, pairs_table


def groups(selection):
    order, old = selection["order"], selection["old_last"]
    named = {"base": [], "old4": old, "outside28": selection["outside"], "all32": list(range(32))}
    named.update({f"top{k}": order[:k] for k in BUDGETS})
    named.update(low4=order[-4:], top4_low4=order[:4]+order[-4:], negative=selection["negative"],
                 reinforce_pair=selection["reinforce_pair"], cancel_pair=selection["cancel_pair"],
                 old4_top4=old+order[:4], old4_top8=old+order[:8])
    for replica, permutation in enumerate(selection["random"], 1):
        for k in (4, 8):
            named[f"random{replica}_{k}"] = permutation[:k]
    return named


def make_masks(channel_sets, device):
    matrix = np.zeros((len(channel_sets), 32), dtype=bool)
    for i, channels in enumerate(channel_sets):
        matrix[i, list(channels)] = True
    return torch.as_tensor(matrix, device=device)


def scores(state, pooled):
    logits = state.model.readout(pooled)
    receptor, donor = state.classes
    margin = logits[..., donor] - logits[..., receptor]
    return torch.stack((third(margin), margin[-1]), dim=-1)


def block_channels(state, maps, masks, direct_count):
    # 对每个阻断配置，在两个 U 背景中实际运行原分类头。
    pooled = maps.mean((-2, -1))
    mixed = torch.where(masks[None, :, None, :], pooled[:1, None], pooled[:, None])
    result = scores(state, mixed)
    error = 0.
    # 指定组合与全部单通道另在空间激活图上阻断，再池化、读出。
    for start in range(0, direct_count, 4):
        stop = min(start + 4, direct_count)
        mask = masks[start:stop][None, :, None, :, None, None]
        direct = scores(state, torch.where(mask, maps[:1, None], maps[:, None]).mean((-2, -1)))
        error = max(error, float((direct - result[start:stop]).abs().max()))
        result[start:stop] = direct
    return result.cpu().numpy(), error
