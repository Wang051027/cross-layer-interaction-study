"""Activity-only selection controls and exact GAP/linear structural controls.

No original model, fit, runtime or primary analysis is modified.
"""
from __future__ import annotations
from dataclasses import replace
import time
from external_experiment import (BASE, SEEDS, np, torch, F, configure_cpu, load_fit,
    load_model, normalized, choose_nodes, HeadRuntime, LocalModule, source_activation,
    bit_design, low_order, prediction_key, score, sha, now, read_json, dump, csv_write)
from torch import nn

PRIOR = BASE / "runs/fashion_external_20260911"
BASELINES = ["greedy", "weight", "pca_fused", "random", "active_weight", "gate_weight", "activity"]
RANDOM_NAMES = [f"active_random_{i}" for i in range(4)]
ALL_METHODS = BASELINES + RANDOM_NAMES
BUDGETS = [16, 32]
STRENGTHS = [1., .5]


# 从训练门控频率构造活动优先、门控方差、频率和固定活动随机排序。
def activity_orders(mod, fit, seed, layer):
    g = fit.gate
    norm = np.linalg.norm(mod.W.numpy(), axis=1) * np.linalg.norm(mod.U.numpy(), axis=0)
    weight_rank = np.empty(len(g), dtype=int)
    weight_rank[fit.weight_order] = np.arange(len(g))
    result = dict(active_weight=np.lexsort((weight_rank, -(g > 0).astype(int))),
        gate_weight=np.argsort(-(g * (1-g) * norm), kind="stable"),
        activity=np.argsort(-g, kind="stable"))
    for i, name in enumerate(RANDOM_NAMES):
        rng = np.random.default_rng(2026091100 + seed * 10 + layer * 100 + i)
        result[name] = np.concatenate([rng.permutation(np.flatnonzero(g > 0)),
                                       rng.permutation(np.flatnonzero(g == 0))])
    for order in result.values():
        assert np.array_equal(np.sort(order), np.arange(len(g)))
    return result


# 把方法名对应的排序接入统一运行时；完整预算直接使用精确基线。
def head_runtime(mod, fit, orders, method, k):
    if k == 64:
        return HeadRuntime(mod, fit, "exact", 64)
    if method in orders:
        return HeadRuntime(mod, replace(fit, order=orders[method]), "greedy", k)
    return HeadRuntime(mod, fit, method, k)


# 以全局平均池化和线性分类器读出的结构对照网络。
class GapCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.readout = nn.Linear(16, 10)

    # 返回所选干预层位置的激活：第一次 ReLU、第二次 ReLU 或池化之后。
    def source(self, x, layer):
        h = F.relu(self.conv1(x))
        if layer == 1:
            return h
        h = F.relu(self.conv2(h))
        return h if layer == 2 else F.max_pool2d(h, 2)

    # 从对应层继续到池化、空间平均和线性读出。
    def tail(self, h, layer):
        if layer == 1:
            h = F.relu(self.conv2(h))
        if layer < 3:
            h = F.max_pool2d(h, 2)
        return self.readout(h.mean((2, 3)))

    def forward(self, x):
        return self.tail(self.source(x, 1), 1)


# 载入 GAP 结构对照模型，并按参数决定是否转为双精度。
def load_gap(directory, double=True):
    model = GapCNN().eval()
    model.load_state_dict(torch.load(directory / "teacher.pt", weights_only=True, map_location="cpu"))
    return model.double() if double else model


class GapLocal:
    """Exact affected pooling windows followed by a fused linear readout.

    L1: after ReLU1; L2: after ReLU2, before MaxPool; L3: after MaxPool.
    At L3 there is no remaining nonlinear operation and no background is needed.
    """
    def __init__(self, model, shape, nodes, layer):
        self.model, self.shape, self.nodes, self.layer = model, tuple(shape), np.array(nodes), layer
        self.q = len(nodes)
        basis = torch.zeros((self.q,) + tuple(shape), dtype=torch.float64)
        basis.flatten(1)[torch.arange(self.q), self.nodes] = 1
        with torch.no_grad():
            # 第三位置已在池化之后，下游仅有平均和线性读出，因此不再需要背景门状态。
            if layer == 3:
                channels = self.nodes // (shape[1] * shape[2])
                self.linear = model.readout.weight[:, channels].T.detach() / (shape[1] * shape[2])
                self.window_ids = np.empty((0, 4), dtype=int)
                self.pool_ids = np.empty(0, dtype=int)
                return
            if layer == 1:
                c = model.conv2
                mapping = F.conv2d(basis, c.weight, None, c.stride, c.padding, c.dilation, c.groups)
            else:
                mapping = basis
            _, channels, height, width = mapping.shape
            changed = torch.nonzero((mapping != 0).any(0).flatten()).flatten().numpy()
            pool_ids = (changed // (height*width))*(height//2)*(width//2)
            pool_ids += ((changed % (height*width))//width//2)*(width//2) + changed % width//2
            self.pool_ids = np.unique(pool_ids)
            windows = []
            for p in self.pool_ids:
                channel = int(p)//((height//2)*(width//2))
                row = (int(p) % ((height//2)*(width//2)))//(width//2)*2
                col = int(p) % (width//2)*2
                windows.append([channel*height*width+(row+dy)*width+col+dx for dy, dx in [(0,0),(0,1),(1,0),(1,1)]])
            self.window_ids = np.asarray(windows, dtype=int)
            self.A = mapping.flatten(1)[:, self.window_ids.ravel()].detach().contiguous()
            channels = self.pool_ids // ((height//2)*(width//2))
            self.linear = model.readout.weight[:, channels].T.detach() / ((height//2)*(width//2))

    # 提取源值及受影响池化窗口；第三位置没有剩余边界状态。
    @torch.no_grad()
    def prepare(self, source):
        h = source.flatten()[self.nodes]
        if self.layer == 3:
            return h, h[:0].reshape(0, 4)
        pre = self.model.conv2(source) if self.layer == 1 else source
        return h, pre.flatten()[self.window_ids]

    # 按所选方法重算边界或固定原区域；第三位置直接使用线性响应。
    @torch.no_grad()
    def evaluate(self, context, bits, alpha=1., method="exact", mean_window=None):
        h, window = context
        delta = -alpha * bits * h[None]
        if self.layer == 3:
            return delta @ self.linear
        if method == "mean_boundary":
            window = mean_window
        displacement = (delta @ self.A).reshape(len(bits), -1, 4)
        baseline = F.relu(window) if self.layer == 1 else window
        if method == "original_region":
            if self.layer == 1:
                displacement = displacement * (window > 0)[None]
            # 原区域近似固定原池化赢家，无法表示干预后赢家发生变化的情况。
            winner = baseline.argmax(1)
            dp = displacement.gather(2, winner[None, :, None].expand(len(bits), -1, 1)).squeeze(2)
        else:
            changed = window[None] + displacement
            changed = F.relu(changed) if self.layer == 1 else changed
            dp = changed.amax(2) - baseline.amax(1)[None]
        return dp @ self.linear

    # 直接改变原特征图并运行 GAP 教师尾部，验证局部计算是否准确。
    @torch.no_grad()
    def full_evaluate(self, source, bits, alpha=1.):
        intervened = source.expand(len(bits), -1, -1, -1).clone()
        intervened.flatten(1)[:, self.nodes] *= (1-alpha*bits)
        return self.model.tail(intervened, self.layer) - self.model.tail(source, self.layer)


# 按模型、方法和预算记录预测误差。
def metrics(prediction, reference, seed, layer, method, k, alpha, architecture):
    held = bit_design(8).sum(1) > 2
    result = score(prediction, reference, held)
    denominator = np.square(reference[:, held]).sum((1, 2))
    numerator = np.square((prediction-reference)[:, held]).sum((1, 2))
    usable = denominator > 1e-24
    per_input = np.sqrt(numerator[usable] / denominator[usable])
    return dict(seed=seed, layer=layer, architecture=architecture, method=method, k=k, alpha=alpha,
        **result, n_inputs=len(prediction), n_coalitions=219, zero_effect_inputs=int((~usable).sum()),
        input_p95=float(np.quantile(per_input, .95)) if len(per_input) else 0.,
        input_fraction_over_10pct=float((per_input > .1).mean()) if len(per_input) else 0.)


# 随机化方法的运行次序并重复计时，保存逐轮记录以保持配对。
def timed_functions(functions, rng, rounds=15, repetitions=3):
    records = []
    with torch.no_grad():
        for fn in functions.values():
            for _ in range(3):
                fn()
        keys = list(functions)
        for repeat in range(rounds):
            for i in rng.permutation(len(keys)):
                start = time.perf_counter_ns()
                for _ in range(repetitions):
                    functions[keys[i]]()
                records.append(dict(method=keys[i], round=repeat,
                    seconds=(time.perf_counter_ns()-start)/repetitions/1e9))
    return records
