"""八状态阻断背景下的支路、门控和通道干预。"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


def third(x):
    """输入依次为空组、三个单组、三个双组、三组全选。"""
    return x[7] - x[4] - x[5] - x[6] + x[1] + x[2] + x[3] - x[0]


def channel_terms(hidden, weights):
    return third(hidden).mean(dim=(-1, -2)) * weights


def model_third(model, hidden, classes):
    r, d = classes
    logits = model.readout(hidden.mean(dim=(-1, -2)))
    return third(logits[..., d] - logits[..., r])


def site_mask(ids, reference):
    mask = torch.zeros(reference.shape, dtype=torch.bool, device=reference.device)
    mask.view(-1)[ids[ids >= 0].astype(np.int64)] = True
    return mask


@dataclass
class PairStates:
    middle: torch.Tensor
    z1: torch.Tensor
    skip: torch.Tensor
    first_empty: torch.Tensor
    last_empty: torch.Tensor


def pair_states(model, sources):
    middle = model.block2(sources)
    block = model.block3
    z1 = block.bn1(block.conv1(middle))
    skip = block.shortcut(middle)
    last_empty = block.bn2(block.conv2(z1[:1].relu())) + skip[:1]
    return PairStates(middle, z1, skip, z1[0] > 0, last_empty[0] > 0)


class LockedBlock:
    """同一个三组的八个状态共用两份空间位置名单。"""
    def __init__(self, model, pair, corners, sites, classes):
        self.model = model
        self.block = model.block3
        self.classes = classes
        r, d = classes
        self.weights = model.readout.weight[d] - model.readout.weight[r]
        self.middle = pair.middle[corners]
        self.skip = pair.skip[corners]
        self.z1 = pair.z1[corners]
        self.g0_first = pair.first_empty
        self.g0_last = pair.last_empty
        self.first_sites = site_mask(sites[0], self.g0_first)
        self.last_sites = site_mask(sites[1], self.g0_last)
        self.first = torch.where(self.first_sites, self.z1 * self.g0_first, self.z1.relu())
        self.conv = self.second_conv(self.first)
        self.z2 = self.conv + self.skip
        self.last_gates = torch.where(self.last_sites, self.g0_last, self.z2 > 0)
        self.last = self.z2 * self.last_gates
        self.phi = channel_terms(self.last, self.weights)
        self.y = model_third(model, self.last, classes)

    def second_conv(self, first):
        return self.block.bn2(self.block.conv2(first))

    def last_activation(self, z):
        return torch.where(self.last_sites, self.g0_last * z, z.relu())

    def contribution(self, field):
        return channel_terms(field, self.weights).sum(dim=-1)

    def individual_first(self, engine):
        """矩阵 [i,j] 记录恢复第一处通道 i 后，末层通道 j 移除的带符号三阶量。"""
        channels = self.first.shape[1]
        if engine == "direct":
            results = []
            for channel in range(channels):
                restored = self.first.clone()
                restored[:, channel] = self.first[0, channel]
                output = self.last_activation(self.second_conv(restored) + self.skip)
                results.append(self.phi - channel_terms(output, self.weights))
            return torch.stack(results)
        conv, bn = self.block.conv2, self.block.bn2
        slope = bn.weight / (bn.running_var + bn.eps).sqrt()
        # 每个输入通道独立生成全部输出通道的预激活增量，BN 平移在相减中消去。
        kernels = (conv.weight * slope[:, None, None, None]).transpose(0, 1).contiguous()
        kernels = kernels.reshape(channels * len(slope), 1, *conv.kernel_size)
        shift = F.conv2d(self.first[:1] - self.first, kernels, padding=conv.padding, groups=channels)
        shift = shift.reshape(len(self.first), channels, *self.z2.shape[1:])
        final = self.last_activation(self.z2[:, None] + shift)
        return self.phi[None, :] - channel_terms(final, self.weights)

    def group(self, first_ids, last_ids):
        """分别恢复两层及同时恢复，三种得分均经过原池化和分类头。"""
        restored = self.first.clone()
        restored[:, first_ids] = self.first[:1, first_ids]
        final_after_first = self.last_activation(self.second_conv(restored) + self.skip)
        final_after_last = self.last.clone()
        final_after_last[:, last_ids] = self.last[:1, last_ids]
        final_after_both = final_after_first.clone()
        final_after_both[:, last_ids] = self.last[:1, last_ids]
        values = [model_third(self.model, h, self.classes) for h in (final_after_first, final_after_last, final_after_both)]
        transfer = self.phi - channel_terms(final_after_first, self.weights)
        return (*values, transfer)

    def split_routes(self, empty_gradient):
        conv_only = self.contribution(self.last_activation(self.conv + self.skip[:1]))
        skip_only = self.contribution(self.last_activation(self.conv[:1] + self.skip))
        # 两组额外的全层开关钳制仍保留每个子集的传入幅值。
        z_first_fixed = self.second_conv(self.z1 * self.g0_first) + self.skip
        affine = self.contribution(z_first_fixed * self.g0_last)
        first_free = self.contribution(self.z2 * self.g0_last)
        last_free = self.contribution(self.last_activation(z_first_fixed))
        fields = dict(conv_only=conv_only, skip_only=skip_only, branch_mixed=self.y-conv_only-skip_only,
                      gate_affine=affine, gate_first=first_free-affine, gate_last=last_free-affine,
                      gate_mixed=self.y-first_free-last_free+affine, first_only_free=first_free, last_only_free=last_free)
        return fields, (affine-(empty_gradient*third(self.middle)).sum()).abs().item()

    def spatial(self):
        coordinate = third(self.last) * self.weights[:, None, None] / self.last.shape[-2:].numel()
        gates = self.last_gates
        partitions = {"locked": self.last_sites, "always_on": (~self.last_sites) & gates.all(0),
                      "switching": (~self.last_sites) & (gates.any(0) & ~gates.all(0)),
                      "always_off": (~self.last_sites) & ~gates.any(0)}
        totals = {}
        membership = np.empty(coordinate.numel(), dtype=np.int8)
        for index, (name, mask) in enumerate(partitions.items()):
            totals["site_" + name] = coordinate[mask].sum().item()
            totals["count_" + name] = int(mask.sum().item())
            membership[mask.flatten().cpu().numpy()] = index
        totals.update(site_abs=coordinate.abs().sum().item(), channel_abs=self.phi.abs().sum().item(),
                      site_positive=coordinate.clamp_min(0).sum().item(), site_negative=coordinate.clamp_max(0).sum().item())
        values = coordinate.flatten().cpu().numpy()
        ordered = np.round(values, 11)
        extremes = []
        for sign, label in ((1, "positive"), (-1, "negative")):
            candidates = np.flatnonzero(sign * ordered > 0)
            ranked = candidates[np.lexsort((candidates, -sign * ordered[candidates]))[:4]]
            for rank, flat in enumerate(ranked, 1):
                channel, y, x = np.unravel_index(flat, coordinate.shape)
                extremes.append(dict(direction=label, rank=rank, channel=int(channel), y=int(y), x=int(x),
                                     gate_group=tuple(partitions)[membership[flat]], contribution=float(values[flat])))
        return totals, extremes
