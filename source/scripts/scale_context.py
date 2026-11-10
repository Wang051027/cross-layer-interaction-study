"""Compute an affine fallback for different source sizes and positions. Evaluate the real-arithmetic bound in float64."""
from residual_context import *
from audit_residual_affine_runtime import FusedAffine

SIZES = [4, 8, 12]
POSITIONS = {"center": (7, 7), "upper_left": (3, 3)}
BUDGETS = [.25, .5]
THRESHOLD = .10
NEW_RUN = BASE / "runs/fashion_scale_20260914"
PRIOR_RUN = BASE / "runs/fashion_residual_20260913"


# q≤8 枚举所有组合；q>8 保留全部低阶组合并固定抽取 256 个高阶组合。
def query_design(q):
    all_bits = bit_design(q)
    orders = all_bits.sum(1)
    train = np.flatnonzero(orders <= 2)
    high = np.flatnonzero(orders > 2)
    # 规模较大时普通评价只固定抽取部分高阶组合；不要误称为全枚举。
    if q > 8:
        high = np.sort(np.random.default_rng(2026091400 + q).choice(high, 256, replace=False))
    indices = np.sort(np.concatenate([train, high]))
    return all_bits[train], all_bits[indices], indices


# 按指定空间位置的 3×3 窗口和训练平均激活，选出前 q 个源坐标。
def choose_source_nodes(sources, position, q):
    _, channels, height, width = sources.shape
    row, col = POSITIONS[position]
    ids = np.arange(channels * height * width).reshape(channels, height, width)
    window = ids[:, row-1:row+2, col-1:col+2].ravel()
    means = sources.mean(0).numpy().ravel()
    # 稳定排序固定并列处理，因此 q=4/8/12 是同一排序的嵌套前缀。
    order = window[np.argsort(-means[window], kind="stable")]
    return order[:q]


# 从训练上下文和低阶查询拟合平均门控、PCA 及各状态排序，并返回诊断统计。
def fit_scale(local, contexts, train_bits, seed):
    h = torch.stack([c[0] for c in contexts]).numpy()
    z = torch.stack([c[1] for c in contexts]).numpy()
    gate = (z > 0).mean(0)
    delta = (-h[:, None] * train_bits[None]).reshape(-1, local.q)
    t = delta @ local.W.numpy().T
    zz = np.repeat(z, len(train_bits), axis=0)
    # 这里的余项相对训练平均门控定义，用于选状态，而非语义贡献分配。
    residual = np.maximum(zz+t, 0) - np.maximum(zz, 0) - t*gate
    order, gains = greedy_order(residual, local.U.numpy())
    # 用奇异值分解提取低维方向；中心化对象和保留维数由本段数据与预算确定。
    _, singular, components = np.linalg.svd(z-z.mean(0), full_matrices=False)
    norm = np.linalg.norm(local.W.numpy(), axis=1) * np.linalg.norm(local.U.numpy(), axis=0)
    values = dict(gate=gate, mean_z=z.mean(0), components=components, greedy=order, gains=gains,
                  weight=np.argsort(-norm, kind="stable"),
                  gate_weight=np.argsort(-(gate*(1-gate)*norm), kind="stable"),
                  activity=np.argsort(-gate, kind="stable"))
    for i in range(4):
        rng = np.random.default_rng(2026091400 + seed*10 + i)
        values[f"active_random_{i}"] = np.concatenate([
            rng.permutation(np.flatnonzero(gate > 0)), rng.permutation(np.flatnonzero(gate == 0))])
    return ResidualFit(**values), dict(
        affected_states=local.J, train_contexts=len(contexts), train_coalitions=len(train_bits),
        training_active_states=int((gate > 0).sum()), always_open_states=int((gate == 1).sum()),
        changing_states=int(((gate > 0) & (gate < 1)).sum()),
        pca_components=len(components), singular_values=singular.tolist())


# 支持可变源坐标数的残差运行时，批量计算多图像、多组合的响应。
class ScaleRuntime(ContextRuntime):
    # 分批计算多张输入的组合响应，返回带输入批次维度的预测数组。
    @torch.no_grad()
    def evaluate_many(self, contexts, bits, alpha=1.):
        outputs = []
        for begin in range(0, len(contexts), 8):
            group = contexts[begin:begin+8]
            h = torch.stack([c[0] for c in group])
            z = torch.stack([c[1] for c in group])
            if self.method == "pca_fused":
                z = self.mean + z @ self.V
            delta = (-alpha*h[:, None]*bits[None]).reshape(-1, self.local.q)
            if self.W.shape[0] == 0:
                value = delta @ self.linear.T
            else:
                t = (delta @ self.W.T).reshape(len(group), len(bits), -1)
                response = F.relu(z[:, None]+t)-F.relu(z)[:, None]
                value = response.reshape(-1, self.W.shape[0]) @ self.U.T
                if self.linear is not None:
                    value += delta @ self.linear.T
            outputs.append(value.reshape(len(group), len(bits), 10))
        return torch.cat(outputs)


# 先用原区域仿射响应和余项上界判断是否接受近似，否则精确回退。
class BoundedAffine(FusedAffine):
    # 预计算原区域响应矩阵及每个固定强度的非线性余项上界，供后续接受判断。
    @torch.no_grad()
    def cached(self, context):
        h, z = context
        _, matrix = super().cached(context)
        contributions = -self.local.W*h[None]
        low = contributions.clamp(max=0).sum(1)
        high = contributions.clamp(min=0).sum(1)
        bounds = {}
        for alpha in STRENGTHS:
            # ReLU's departure from its original tangent is always nonnegative.
            # ReLU 相对原切线的偏离非负，可用区间端点得到每个状态的最大余项。
            b = torch.where(z > 0, F.relu(-z-alpha*low), F.relu(z+alpha*high))
            output_low = self.local.U.clamp(max=0) @ b
            output_high = self.local.U.clamp(min=0) @ b
            beta = torch.maximum(output_low.abs(), output_high.abs()).norm()
            bounds[alpha] = (beta, int((b > 0).sum()))
        return h, z, matrix, bounds

    # 用解析误差上界与预测范数判断是否满足固定阈值；拒绝时调用精确局部方程。
    @torch.no_grad()
    def decision(self, context, bits, alpha=1.):
        h, z, matrix, bounds = context
        approximation = (-alpha*bits*h[None]) @ matrix.T
        # 评分只包含三个及以上源坐标同时被干预的组合，避免混入二阶拟合点。
        held = bits.sum(1) > 2
        if not bool(held.any()):
            held = torch.ones(len(bits), dtype=torch.bool)
        bound = bounds[alpha][0] * math.sqrt(int(held.sum()))
        predicted_norm = approximation[held].norm()
        # 通过三角不等式把相对真实效应的阈值转换为只需预测范数的判定。
        accept = bool(bound <= THRESHOLD/(1+THRESHOLD)*predicted_norm)
        prediction = approximation if accept else self.local.exact((h, z), bits, alpha)
        return prediction, dict(accepted=accept, error_norm_upper=float(bound),
            approximation_norm=float(predicted_norm), potentially_switching_states=bounds[alpha][1])

    # 对给定上下文和组合掩码计算输出变化；具体方法由对象构造时的配置决定。
    @torch.no_grad()
    def evaluate(self, context, bits, alpha=1.):
        return self.decision(context, bits, alpha)[0]


# 列出规模实验的近似预算及零预算、原区域、回退、精确对照。
def scale_specs(local):
    return [(m, f, math.ceil(local.J*f)) for m in ("greedy", "gate_weight", "pca_fused") for f in BUDGETS] + [
        ("mean_gate", 0., 0), ("affine_fused", 1., local.J),
        ("bounded_affine", 1., local.J), ("exact", 1., local.J)]


# 根据方法名实例化融合仿射、带上界回退或常规规模运行时。
def make_runtime(local, fit, method, k, engine="exact_crop"):
    if method == "affine_fused":
        return FusedAffine(local)
    if method == "bounded_affine":
        return BoundedAffine(local)
    return ScaleRuntime(local, fit, method, k, engine)


# 按运行时类型选择批量或逐输入路径，统一返回 NumPy 响应数组。
def evaluate_many(runtime, contexts, bits, alpha):
    if isinstance(runtime, ScaleRuntime):
        return runtime.evaluate_many(contexts, bits, alpha).numpy()
    return torch.stack([runtime.evaluate(c, bits, alpha) for c in contexts]).numpy()


# 在掩码阶数大于二的组合上评分，并保留零效应数量及单输入误差尾部。
def scale_metrics(prediction, reference, bits, **labels):
    held = bits.sum(1) > 2
    result = score(prediction, reference, held)
    den = np.square(reference[:, held]).sum((1, 2))
    num = np.square((prediction-reference)[:, held]).sum((1, 2))
    usable = den > 1e-24
    per = np.sqrt(num[usable]/den[usable])
    return dict(**labels, **result, inputs=len(prediction), scored_coalitions=int(held.sum()),
        zero_effect_inputs=int((~usable).sum()),
        effect_rms=float(np.sqrt(np.square(reference[:, held]).mean())),
        error_rms=float(np.sqrt(np.square((prediction-reference)[:, held]).mean())),
        input_p95=float(np.quantile(per, .95)) if len(per) else 0.,
        input_fraction_over_10pct=float((per > .10).mean()) if len(per) else 0.)
