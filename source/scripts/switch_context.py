# 中文阅读说明｜用途：将整个干预范围内始终开启/关闭的 ReLU 合并，仅精算可能跨零的状态。
# 数据流：准备步骤覆盖 alpha∈[0,1]；输出必须与原局部方程一致，不是压缩近似。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Exact local responses after folding states stable over the intervention box.

The interval screens states, not queries. One preparation covers every binary
subset and every alpha in [0, 1]. Float64 with a conservative boundary guard is
used; this is not outward-rounded formal verification.
"""
from scale_context import *

SWITCH_RUN = BASE / "runs/fashion_switch_20260914"
SCALE_RUN = BASE / "runs/fashion_scale_20260914"
GUARD_FACTOR = 64.


# 用途：把稳定开启状态合并为线性项，可能跨零的状态保留完整 ReLU 响应。
# 结构：成员字段和方法共同保存本阶段需要的状态。
class SwitchExact:
    # 用途：建立对象所需的网络层、固定映射或配置。把稳定开启状态合并为线性项，可能跨零的状态保留完整 ReLU 响应。
    # 参数：local、engine；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：初始化对象字段，没有显式返回值。
    def __init__(self, local, engine="exact_crop"):
        self.local = local
        self.engine = engine

    # 用途：计算整个独立衰减范围内的状态上下界，返回稳定开启与可能跨零的掩码。
    # 参数：context；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：(stable_on, candidates)。
    @torch.no_grad()
    def screen(self, context):
        h, z = context
        # 考虑从原值到完全衰减的整个盒范围，一次准备可覆盖两种强度。
        contributions = -self.local.W * h[None]
        low = contributions.clamp(max=0).sum(1)
        high = contributions.clamp(min=0).sum(1)
        guard = GUARD_FACTOR * torch.finfo(z.dtype).eps * (z.abs() + high - low + 1.)
        # 下界仍为正则始终开启，上界仍为负则始终关闭，其余保留精确非线性。
        stable_on = z + low > guard
        stable_off = z + high < -guard
        candidates = ~(stable_on | stable_off)
        return stable_on, candidates

    # 用途：把稳定开启状态收缩成线性矩阵，只缓存候选状态的 W、U、z。
    # 参数：context；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：(h, matrix, self.local.W[idx].contiguous(), self.local.U[:, idx].contiguous(), z[idx].c…。
    @torch.no_grad()
    def cached(self, context):
        h, z = context
        stable_on, candidates = self.screen(context)
        idx = torch.nonzero(candidates).flatten()
        # Charge this full-width contraction and dynamic compaction to prepare.
        # 稳定开启状态合并为 U·diag(on)·W；稳定关闭状态没有响应。
        matrix = ((self.local.U * stable_on[None]) @ self.local.W).contiguous()
        return (h, matrix, self.local.W[idx].contiguous(),
                self.local.U[:, idx].contiguous(), z[idx].contiguous())

    # 用途：从未干预的源激活取得当前方法所需上下文，使同一输入的多个组合可复用。
    # 参数：source、residual；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：self.cached(self.local.states(source, residual, self.engine))。
    @torch.no_grad()
    def prepare(self, source, residual):
        return self.cached(self.local.states(source, residual, self.engine))

    # 用途：对 alpha∈[0,1]，将稳定线性项与候选状态精确 ReLU 输出相加。
    # 参数：context、bits、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：response。
    @torch.no_grad()
    def evaluate(self, context, bits, alpha=1.):
        if not 0. <= alpha <= 1.:
            raise ValueError("Preparation is valid only for alpha in [0, 1]")
        h, matrix, W, U, z = context
        delta = -alpha * bits * h[None]
        response = delta @ matrix.T
        if len(z):
            t = delta @ W.T
            response += (F.relu(z[None] + t) - F.relu(z)[None]) @ U.T
        return response


# 用途：保留所有受影响状态的精确基线，可切换不同状态获取实现。
# 结构：成员字段和方法共同保存本阶段需要的状态。
class ExactAll:
    """Existing exact equation, with no unnecessary context copy at preparation."""
    # 用途：建立对象所需的网络层、固定映射或配置。保留所有受影响状态的精确基线，可切换不同状态获取实现。
    # 参数：local、engine；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：初始化对象字段，没有显式返回值。
    def __init__(self, local, engine="exact_crop"):
        self.local = local
        self.engine = engine

    # 用途：原样返回完整上下文，避免给精确基线额外增加复制开销。
    # 参数：context；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：context。
    def cached(self, context):
        return context

    # 用途：从未干预的源激活取得当前方法所需上下文，使同一输入的多个组合可复用。
    # 参数：source、residual；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：self.local.states(source, residual, self.engine)。
    @torch.no_grad()
    def prepare(self, source, residual):
        return self.local.states(source, residual, self.engine)

    # 用途：对给定上下文和组合掩码计算输出变化；具体方法由对象构造时的配置决定。
    # 参数：context、bits、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：self.local.exact(context, bits, alpha)。
    @torch.no_grad()
    def evaluate(self, context, bits, alpha=1.):
        return self.local.exact(context, bits, alpha)


# 用途：建立本设置要比较的执行器；计时模式同时列出不同精确获取后端。
# 参数：local、timing；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：values。
def runtime_set(local, timing=False):
    values = dict(switch_exact=SwitchExact(local), affine_fused=FusedAffine(local),
                  bounded_affine=BoundedAffine(local))
    if timing:
        values.update({e: ExactAll(local, e) for e in EXACT_ENGINES})
    else:
        values["exact"] = ExactAll(local)
    return values


# 用途：按底层存储地址去重后统计缓存字节，避免视图重复计数。
# 参数：value；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：sum(storage.values())。
def tensor_bytes(value):
    # 用途：递归遍历列表、元组和字典中的张量，供存储去重统计。
    # 参数：x；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：逐次生成 x / 逐次生成 tensors(item)。
    def tensors(x):
        if isinstance(x, torch.Tensor):
            yield x
        elif isinstance(x, (list, tuple)):
            for item in x:
                yield from tensors(item)
        elif isinstance(x, dict):
            for item in x.values():
                yield from tensors(item)
    # 按底层存储去重，多个张量视图共享一块内存时只计一次。
    storage = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors(value)}
    return sum(storage.values())


# 用途：核对被判为稳定的状态，其非线性余项是否为零或仅有浮点误差。
# 参数：local、context、bits、alpha、candidates；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：float(omitted.abs().max()) if omitted.numel() else 0.0。
@torch.no_grad()
def omitted_state_error(local, context, bits, alpha, candidates):
    h, z = context
    t = (-alpha * bits * h[None]) @ local.W.T
    remainder = F.relu(z[None] + t) - F.relu(z)[None] - t * (z > 0)[None]
    omitted = remainder[:, ~candidates]
    return float(omitted.abs().max()) if omitted.numel() else 0.
