"""Exact local responses after folding states stable over the intervention box.

The interval screens states, not queries. One preparation covers every binary
subset and every alpha in [0, 1]. Float64 with a conservative boundary guard is
used; this is not outward-rounded formal verification.
"""
from scale_context import *

SWITCH_RUN = BASE / "runs/fashion_switch_20260914"
SCALE_RUN = BASE / "runs/fashion_scale_20260914"
GUARD_FACTOR = 64.


# 把稳定开启状态合并为线性项，可能跨零的状态保留完整 ReLU 响应。
class SwitchExact:
    def __init__(self, local, engine="exact_crop"):
        self.local = local
        self.engine = engine

    # 计算整个独立衰减范围内的状态上下界，返回稳定开启与可能跨零的掩码。
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

    # 把稳定开启状态收缩成线性矩阵，只缓存候选状态的 W、U、z。
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

    # 从未干预的源激活取得当前方法所需上下文，使同一输入的多个组合可复用。
    @torch.no_grad()
    def prepare(self, source, residual):
        return self.cached(self.local.states(source, residual, self.engine))

    # 对 alpha∈[0,1]，将稳定线性项与候选状态精确 ReLU 输出相加。
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


class ExactAll:
    """Existing exact equation, with no unnecessary context copy at preparation."""
    def __init__(self, local, engine="exact_crop"):
        self.local = local
        self.engine = engine

    # 原样返回完整上下文，避免给精确基线额外增加复制开销。
    def cached(self, context):
        return context

    # 从未干预的源激活取得当前方法所需上下文，使同一输入的多个组合可复用。
    @torch.no_grad()
    def prepare(self, source, residual):
        return self.local.states(source, residual, self.engine)

    # 对给定上下文和组合掩码计算输出变化；具体方法由对象构造时的配置决定。
    @torch.no_grad()
    def evaluate(self, context, bits, alpha=1.):
        return self.local.exact(context, bits, alpha)


# 建立本设置要比较的执行器；计时模式同时列出不同精确获取后端。
def runtime_set(local, timing=False):
    values = dict(switch_exact=SwitchExact(local), affine_fused=FusedAffine(local),
                  bounded_affine=BoundedAffine(local))
    if timing:
        values.update({e: ExactAll(local, e) for e in EXACT_ENGINES})
    else:
        values["exact"] = ExactAll(local)
    return values


# 按底层存储地址去重后统计缓存字节，避免视图重复计数。
def tensor_bytes(value):
    # 递归遍历列表、元组和字典中的张量，供存储去重统计。
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


# 核对被判为稳定的状态，其非线性余项是否为零或仅有浮点误差。
@torch.no_grad()
def omitted_state_error(local, context, bits, alpha, candidates):
    h, z = context
    t = (-alpha * bits * h[None]) @ local.W.T
    remainder = F.relu(z[None] + t) - F.relu(z)[None] - t * (z > 0)[None]
    omitted = remainder[:, ~candidates]
    return float(omitted.abs().max()) if omitted.numel() else 0.
