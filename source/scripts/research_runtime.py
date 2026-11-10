"""Matched CPU runtimes for the unchanged v0.7 estimators.

The small teacher architectures reproduce the audited upstream class definitions
without importing its unrelated sklearn/pandas analysis dependencies.
PCA is fitted to fc1 PREACTIVATIONS; its encoder can therefore be fused into fc1.
The full reconstructed head and full response matrices remain in PCA evaluation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.dont_write_bytecode = True

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[1]
UPSTREAM = BASE / "upstream" / "CAT_MT_phase7_v0.7"
sys.path[:0] = [str(UPSTREAM / "vendor"), str(UPSTREAM / "code")]
from context_module import Fit, fit_module, predict  # noqa: E402
from local_modules import LocalModule, bit_design, low_order, source_activation  # noqa: E402


# 较早 CNN 教师：两层卷积、池化、非线性全连接头与分类读出。
class SmallCNN(nn.Module):
    def __init__(self, channels, side, head, classes, padding_mode="zeros"):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, padding_mode=padding_mode)
        self.fc1 = nn.Linear(c2 * (side // 2) ** 2, head)
        self.fc2 = nn.Linear(head, classes)
        self.transition_padding_mode = padding_mode

    # 执行前向并返回各层预激活、激活与最终输出，供后续状态分析。
    def trace(self, x):
        z1 = self.conv1(x)
        a1 = F.relu(z1)
        z2 = self.conv2(a1)
        a2 = F.relu(z2)
        z3 = self.fc1(F.max_pool2d(a2, 2).flatten(1))
        a3 = F.relu(z3)
        return dict(z1=z1, a1=a1, z2=z2, a2=a2, z3=z3, a3=a3, logits=self.fc2(a3))

    def forward(self, x):
        return self.trace(x)["logits"]


# 按 digits、synthetic、fashion 选择与旧检查点一致的 CNN 尺寸。
def make_teacher(dataset):
    if dataset == "digits":
        return SmallCNN((8, 16), 8, 64, 10)
    if dataset == "synthetic":
        return SmallCNN((6, 12), 16, 48, 2, "circular")
    if dataset == "fashion":
        return SmallCNN((8, 16), 28, 64, 10)
    raise ValueError(dataset)


# 在指定评分组合上汇总预测残差与真实效应，返回相对和绝对误差指标。
def score(prediction, reference, held):
    error = prediction[:, held] - reference[:, held]
    target = reference[:, held]
    sse = float(np.square(error).sum())
    effect_sse = float(np.square(target).sum())
    return dict(nrmse=float(np.sqrt(sse / max(effect_sse, 1e-30))),
                sse=sse, effect_sse=effect_sse,
                mae=float(np.abs(error).mean()), maxerr=float(np.abs(error).max()))


class HeadRuntime:
    """Same boundary calculation, backend, dtype and query batching for all heads.

    Context carries q source values, 4 * affected-pool-count boundary values, and
    k head states/codes (J for exact). PCA reconstruction is an evaluation-time
    temporary. Its full W/U matrices and decoder are explicitly retained/counted.
    """
    def __init__(self, mod, fit, method, k):
        self.mod, self.method, self.k = mod, method, k
        self.J = len(fit.gate)
        self.A = mod.A
        dtype, device = mod.W.dtype, mod.W.device
        self.mean = torch.as_tensor(fit.mean_z, dtype=dtype, device=device)
        self.V = torch.as_tensor(fit.pca_components[:k].copy(), dtype=dtype, device=device)
        self.full_head_W = mod.model.fc1.weight.detach()
        self.full_head_b = mod.model.fc1.bias.detach()
        self.W, self.U = mod.W, mod.U
        self.idx = torch.arange(self.J, device=device)
        self.linear = None
        if method in ("greedy", "weight", "random", "boundary_only"):
            order = fit.order if method in ("greedy", "boundary_only") else (
                fit.weight_order if method == "weight" else fit.random_order)
            self.idx = torch.as_tensor(order[:k].copy(), device=device)
            mask = torch.ones(self.J, dtype=torch.bool, device=device)
            mask[self.idx] = False
            gate = torch.as_tensor(fit.gate, dtype=dtype, device=device)
            self.linear = ((mod.U[:, mask] * gate[mask]) @ mod.W[mask]).contiguous()
            self.W = mod.W[self.idx].contiguous()
            self.U = mod.U[:, self.idx].contiguous()
            self.head_W = self.full_head_W[self.idx].contiguous()
            self.head_b = self.full_head_b[self.idx].contiguous()
        # 把 PCA 编码器提前乘进原头部仿射权重，减少状态获取成本而保持同一近似。
        elif method == "pca_fused":
            self.head_W = (self.V @ self.full_head_W).contiguous()
            self.head_b = (self.V @ (self.full_head_b - self.mean)).contiguous()
        elif method in ("pca_unfused", "exact"):
            self.head_W = self.full_head_W
            self.head_b = self.full_head_b
        elif method == "pool_ridge":
            self.ridge = torch.as_tensor(fit.ridge, dtype=dtype, device=device)
        elif method != "mean_head":
            raise ValueError(method)

    # 把已有完整头部预激活转成方法所需的子集、PCA 码或空背景。
    def _encode_full(self, z):
        if self.method == "exact":
            return z
        if self.method in ("pca_fused", "pca_unfused"):
            return (z - self.mean) @ self.V.T
        if self.method in ("mean_head", "pool_ridge"):
            return z[:0]
        return z[self.idx]

    # 利用已经获取的共同上下文，只进行该方法的状态选择/编码。
    @torch.no_grad()
    def prepare_cached(self, common):
        # The preceding full teacher pass is common and already complete.
        h, window, z = common
        return h, window, self._encode_full(z)

    # 从源激活计算边界与必要的头部状态；融合 PCA 直接获取低维编码。
    @torch.no_grad()
    def prepare(self, source):
        pre = self.mod.model.conv2(source) if self.mod.layer == 1 else source
        activation = F.relu(pre) if self.mod.layer == 1 else pre
        pooled = F.max_pool2d(activation, 2).flatten()
        h = source.flatten()[self.mod.nodes]
        window = pre.flatten()[self.mod.window_ids]
        if self.method in ("mean_head", "pool_ridge"):
            z = pooled[:0]
        else:
            z = F.linear(pooled, self.head_W, self.head_b)
            if self.method == "pca_unfused":
                z = (z - self.mean) @ self.V.T
        return h, window, z

    # 重算受影响池化窗口，再经过方法对应的非线性头部与线性省略项。
    @torch.no_grad()
    def evaluate(self, ctx, bits, alpha=1.0):
        h, window, z = ctx
        zz = window[None] + (-alpha * bits * h[None] @ self.A).reshape(len(bits), -1, 4)
        activation = F.relu(zz) if self.mod.layer == 1 else zz
        baseline = F.relu(window) if self.mod.layer == 1 else window
        dp = activation.amax(2) - baseline.amax(1)[None]
        if self.method == "pool_ridge":
            return dp @ self.ridge
        if self.method in ("pca_fused", "pca_unfused"):
            # 查询时从低维编码重建完整预激活；解码和完整响应矩阵仍有成本。
            z = self.mean + z @ self.V
        elif self.method == "mean_head":
            z = self.mean
        out = dp @ self.linear.T if self.linear is not None else None
        if self.W.shape[0]:
            term = (F.relu(z + dp @ self.W.T) - F.relu(z)) @ self.U.T
            out = term if out is None else out + term
        return out

    # 统计获取、缓存和矩阵参数数量；PCA 仍保留完整响应矩阵与解码器。
    def cost_counts(self):
        tensors = [self.A, self.W, self.U]
        if hasattr(self, "head_W"):
            tensors += [self.head_W, self.head_b]
        if self.linear is not None:
            tensors += [self.linear]
        if self.method.startswith("pca"):
            tensors += [self.mean, self.V]
        if self.method == "mean_head":
            tensors += [self.mean]
        if self.method == "pool_ridge":
            tensors += [self.ridge]
        return dict(
            source_scalars=self.mod.q,
            boundary_scalars=4 * len(self.mod.pool_ids),
            head_context_scalars=self.J if self.method == "exact" else self.k,
            acquired_head_scalars=self.J if self.method in ("exact", "pca_unfused") else self.k,
            runtime_numeric_bytes=sum(t.numel() * t.element_size() for t in tensors)
            + self.idx.numel() * self.idx.element_size()
            + self.mod.nodes.nbytes + self.mod.window_ids.nbytes,
            numeric_bytes_scope="reachable execution arrays only; shared teacher/prefix and Python object overhead excluded",
        )
