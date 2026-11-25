"""在冻结模型里操作 block2 输出、末块第一处 ReLU 和末层开关。"""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "outside_channels_20261126"))
from interventions import Inputs, js, put, verify, sha, checksums, stamp, write_csv, previous
from experiment import read_csv
from paths import third

SEED = 202609291
PAIRS = ((23, 30), (17, 27))
U = (13, 16, 20, 22)
PREVIOUS_SELECT = "9fe5acabec82165dbf47ea15723fd4a86d21a0c5ae62a8ee6a1575df990519a2"
PREVIOUS_MEASURE = "c209009bf9ee1d255fbfa97afec68977191123bf36c2ae5ee74620d09527d8e0"
EPS = 1e-9


def source(inputs):
    earlier = previous(inputs)
    path = earlier.parent / "dsprites_outside_channels_20261126"
    verify(path, {"selection_seal.json": PREVIOUS_SELECT, "measurement_seal.json": PREVIOUS_MEASURE})
    verify(path, js(path/"selection_seal.json")["files"])
    spec = js(path/"selection.json")[str(SEED)]
    if list(U) != spec["upstream"] or [list(p) for p in PAIRS] != [spec["reinforce_pair"], spec["cancel_pair"]]:
        raise ValueError("Target pair or frozen upstream group changed")
    return path


def label(pair):
    return "-".join(map(str, pair))


def balance(a):
    x, y = a[..., 0], a[..., 1]
    return (abs(x+y)-abs(x-y))/2


def signs(b):
    return np.where(b > EPS, 1, np.where(b < -EPS, -1, 0))


def meta(inputs, key):
    ri, di = map(int, inputs.panel["pairs"][key["case"]])
    size, angle = map(int, inputs.panel["factors"][ri, 2:4])
    return dict(strength=key["strength"], case=key["case"], mask=key["mask"], size=size, angle=angle,
                recipient=int(inputs.panel["labels"][ri]), donor=int(inputs.panel["labels"][di]))


@dataclass
class First:
    z: torch.Tensor
    gates: torch.Tensor
    act: torch.Tensor
    skip: torch.Tensor


def natural(s):
    return First(s.z1, torch.where(s.first_sites, s.g0_first, s.z1 > 0), s.first, s.skip)


def middle_block(s, channels, keep=False):
    value = s.middle.clone() if not keep else s.middle[:1].expand_as(s.middle).clone()
    value[:, channels] = s.middle[:1, channels] if not keep else s.middle[:, channels]
    z = s.block.bn1(s.block.conv1(value))
    gates = torch.where(s.first_sites, s.g0_first, z > 0)
    return First(z, gates, z*gates, s.block.shortcut(value))


def restored(act, ref, channels):
    copy = act.clone()
    copy[:, channels] = ref[:, channels]
    return copy


def selected_tail(s, first, skip, dest):
    """只计算给定末层通道，开发阶段给全部候选使用相同的卷积和 BN。"""
    conv, bn = s.block.conv2, s.block.bn2
    v = F.conv2d(first, conv.weight[dest], padding=conv.padding)
    z = F.batch_norm(v, bn.running_mean[dest], bn.running_var[dest], bn.weight[dest], bn.bias[dest],
                     training=False, eps=bn.eps) + skip[:, dest]
    g = torch.where(s.last_sites[dest], s.g0_last[dest], z > 0)
    return z, g, z*g


def candidate_readouts(s, sets, dest, altered=None):
    """在 block2 固定候选，或在受干预的第一处恢复候选，返回每条目标读数。"""
    values = []
    for start in range(0, len(sets), 4):
        chunk = sets[start:start+4]
        mask = np.zeros((len(chunk), 32), dtype=bool)
        for k, channels in enumerate(chunk):
            mask[k, list(channels)] = True
        m = torch.as_tensor(mask, device=s.middle.device)[None, :, :, None, None]
        if altered is None:
            x = torch.where(m, s.middle[:1, None], s.middle[:, None]).flatten(0, 1)
            z = s.block.bn1(s.block.conv1(x))
            a = z * torch.where(s.first_sites, s.g0_first, z>0)
            skip = s.block.shortcut(x)
        else:
            a = torch.where(m, s.first[:, None], altered.act[:, None]).flatten(0, 1)
            skip = altered.skip[:, None].expand(-1, len(chunk), -1, -1, -1).flatten(0, 1)
        _, _, h = selected_tail(s, a, skip, dest)
        pooled = h.mean((-2,-1)).reshape(8, len(chunk), len(dest))
        values.append((third(pooled)*s.weights[dest]).cpu().numpy())
    return np.concatenate(values)


def last(s, first):
    z = s.second_conv(first.act) + first.skip
    gates = torch.where(s.last_sites, s.g0_last, z>0)
    return z, gates


def coordinate_mask(like, *, coords=(), channels=()):
    mask = torch.zeros_like(like, dtype=torch.bool)
    if len(channels):
        mask[list(channels)] = True
    if len(coords):
        ch, y, x = zip(*coords)
        mask[list(ch), list(y), list(x)] = True
    return mask


def replay(z, active, mask, gates):
    return torch.where(mask, z*gates, active)


def bn_kernel(s, dest, src):
    bn = s.block.bn2
    return s.block.conv2.weight[dest][:, src] * (bn.weight[dest]/(bn.running_var[dest]+bn.eps).sqrt())[:,None,None,None]


def site_scores(s, altered, pair, k):
    """第一处按固定末层开关的局部导数筛点，末次按每个位置的确切得分筛点。"""
    dest = list(pair)
    ref = natural(s)
    z, g = last(s, altered)
    change = altered.z[:, k] * (ref.gates[:, k].to(z.dtype)-altered.gates[:, k].to(z.dtype))
    area = z.shape[-2]*z.shape[-1]
    kernel = bn_kernel(s, dest, k)
    first = torch.zeros_like(change[0])
    for i, channel in enumerate(dest):
        slope = g[:, channel:channel+1].to(z.dtype)*s.weights[channel]/area
        grad = F.conv_transpose2d(slope, kernel[i:i+1], padding=1)
        first += third(change*grad).abs()
    diff = s.last_gates[:, dest].to(z.dtype)-g[:, dest].to(z.dtype)
    terminal = third(z[:, dest]*diff).abs()*s.weights[dest,None,None].abs()/area
    counts = ((ref.gates[:, k] != altered.gates[:, k]).sum(0), (diff!=0).sum(0))
    return [(a.cpu().numpy(), b.cpu().numpy()) for a,b in zip((first,terminal), counts)]


def features(s, pair, spec):
    """阈值特征只读取第一处激活、空组末层开关和冻结权重。"""
    k, dest = spec["K"], list(pair)
    g = natural(s).gates
    result = {"readout_product": float(s.weights[dest].prod())}
    for ch in k:
        result[f"opened_{ch}"] = float((g[1:,ch] & ~g[:1,ch]).double().mean())
        result[f"closed_{ch}"] = float((~g[1:,ch] & g[:1,ch]).double().mean())
        result[f"third_mean_{ch}"] = float(third(s.first[:,ch]).mean())
    coords = spec["sites"]["first"]["target"]
    ch,y,x = zip(*coords)
    selected = g[:, list(ch), list(y), list(x)]
    result["selected_opened"] = float((selected[1:] & ~selected[:1]).double().mean())
    result["selected_closed"] = float((~selected[1:] & selected[:1]).double().mean())
    # 两条读出只接收 K 的卷积增量，末层导数取空组值。
    upstream = third(s.first[:, k])[None]
    delta = F.conv2d(upstream, bn_kernel(s,dest,k), padding=1)[0] * s.g0_last[dest]
    drive = (delta.mean((-2,-1))*s.weights[dest]).cpu().numpy()
    result.update(drive_a=float(drive[0]), drive_b=float(drive[1]), drive_balance=float(balance(drive)))
    return {name:round(v,10) for name,v in result.items()}


COLUMNS = ("y", "endpoint", "a", "b", "u_y", "u_endpoint", "t_a", "t_b",
           "cut_y", "cut_endpoint", "other_change_check")


class Tail:
    """在 U 自然及 U 阻断两个背景重新运行末块。复用未改变的卷积结果。"""
    def __init__(self, state, act, skip):
        self.s = state
        fixed = restored(act, act[:1], list(U))
        x = torch.stack((act, fixed), 1)
        z = state.second_conv(x.flatten(0,1)).reshape_as(x) + skip[:,None]
        self.z = z
        self.gates = torch.where(state.last_sites, state.g0_last, z>0)
        self.act = z*self.gates

    def read(self, pair, mask=None, pattern=None):
        s, dest = self.s, list(pair)
        activated = self.act
        if mask is not None:
            # 两个 U 背景均重放同一份已指定的角点开关，空组仍为原值。
            activated = replay(self.z, self.act, mask, pattern[:,None])
        pooled = activated.mean((-2,-1))
        logits = s.model.readout(pooled)
        r,d = s.classes
        margins = logits[:,:,d]-logits[:,:,r]
        by_channel = third(pooled)*s.weights
        a,b = by_channel[0,dest]
        ta,tb = by_channel[0,dest]-by_channel[1,dest]
        # 在真实读出头同时阻断这对通道，独立检查其他 30 通道的贡献。
        cut = pooled[:,0].clone()
        cut[:,dest] = pooled[:1,0,dest]
        cut_logits = s.model.readout(cut)
        cut_margin = cut_logits[:,d]-cut_logits[:,r]
        other = [j for j in range(32) if j not in pair]
        values = (third(margins)[0], margins[-1,0], a, b, third(margins)[1], margins[-1,1], ta,tb,
                  third(cut_margin),cut_margin[-1],third(cut_margin)-by_channel[0,other].sum())
        return torch.stack(values).cpu().numpy()
