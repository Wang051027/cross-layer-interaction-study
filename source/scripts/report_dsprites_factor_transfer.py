"""将已完成且独立审计通过的因素迁移实验整理为中文报告和可追溯表格。

只分析保存的输出，不查询模型、不调参、不挑选有利的教师或因素条件。
六教师的配对差异全部列出；补充规则和分层结果保留为探索性证据。
"""
from __future__ import annotations

import os
from pathlib import Path
import csv
import json
import numpy as np
import dsprites_teacher as teacher
import run_dsprites_factor_transfer as run

os.environ['MPLCONFIGDIR'] = str(run.RUN / 'plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPORT = run.BASE / 'reports/dSprites_因素预测与强度迁移_2026-10-07.md'
FIGURE = run.BASE / 'reports/figures/dsprites_factor_transfer_20261007.png'
RULE_NAMES = dict(global_fixed='全局固定', class_fixed='按类固定', class_scale='按类与大小查表',
    class_angle_ridge='按类与角度回归', class_factor_ridge='按类与大小角度联合回归',
    class_factor_knn='按类与因素邻居查表', input_finite='逐图半强度查询',
    random_mean='随机四次均值', all_shortlist='全部64候选', all_reachable='全部可达坐标')


def method_values(rows, target, rule, alpha, stratum='all', value=-1):
    """每个教师一个比率对，随机重复先在教师内部平均，避免扩大样本数。"""
    names = [f'random_{i}' for i in range(4)] if rule == 'random_mean' else [rule] if rule.startswith('all_') else [f'{target}_{rule}']
    result = []
    for seed in run.SEEDS:
        chosen = [r for r in rows if r['seed'] == seed and r['target'] == target and r['alpha'] == alpha
                  and r['stratum'] == stratum and r['value'] == value and r['method'] in names]
        assert len(chosen) == len(names), (seed, target, rule, alpha, stratum, value)
        result.append([np.mean([r[key] for r in chosen]) for key in ('restore_ratio', 'transplant_error_ratio')])
    return np.asarray(result)


def paired(rows, target, rule, alpha, stratum='all', value=-1):
    """正值表示比按类固定集合少留下交互误差，单位为百分点。"""
    baseline = method_values(rows, target, 'class_fixed', alpha, stratum, value)
    candidate = method_values(rows, target, rule, alpha, stratum, value)
    gain = 100 * (baseline - candidate)
    return dict(target=target, rule=rule, alpha=alpha, stratum=stratum, value=value,
        baseline_mean_percent=(100 * baseline.mean(0)).tolist(), candidate_mean_percent=(100 * candidate.mean(0)).tolist(),
        gain_mean_pp=gain.mean(0).tolist(), gain_by_teacher_pp=gain.tolist(),
        both_improved_teachers=int(np.all(gain > 0, axis=1).sum()),
        restore_improved_teachers=int((gain[:, 0] > 0).sum()), transplant_improved_teachers=int((gain[:, 1] > 0).sum()))


def forecast_diagnostics():
    """半强度的新图分数预测误差和名单重合度仅作为选择器诊断。"""
    records = []
    for seed in run.SEEDS:
        pred = np.load(run.RUN / f'fitted/seed_{seed}/predictions.npz', allow_pickle=False)
        selected = np.load(run.RUN / f'input_selection/seed_{seed}/selection.npz', allow_pickle=False)
        for ti, target in enumerate(('margin', 'vector')):
            for ri, rule in enumerate(run.factors_model.RULES):
                actual = selected['finite_scores'][:, ti]
                mse = float(np.mean((pred['scores'][:, ti * 6 + ri] - actual) ** 2))
                overlap = []
                for x, y in zip(pred['selections'][:, ti * 6 + ri], selected['selections'][:, 12 + ti, :16]):
                    overlap.append(len(set(x) & set(y)) / len(set(x) | set(y)))
                counts = [len({frozenset(x) for x in pred['selections'][pred['factors'][:, 1] == c, ti * 6 + ri]}) for c in range(3)]
                records.append(dict(seed=seed, target=target, rule=rule, singleton_prediction_mse=mse,
                                    input_reference_jaccard_mean=float(np.mean(overlap)), distinct_sets_per_class=counts))
    return records


def development_cv_baselines():
    """补充开发折中的按类均值对照，判断因素回归是否已经出现过拟合。"""
    records = []
    factors = np.load(run.DATA / 'development_factors.npy', allow_pickle=False)
    for seed in run.SEEDS:
        dev = np.load(run.OLD / f'development/seed_{seed}/discovery.npz', allow_pickle=False)
        cv = np.load(run.RUN / f'fitted/seed_{seed}/cv_predictions.npz', allow_pickle=False)
        cfg = teacher.read(run.RUN / f'fitted/seed_{seed}/cv.json')
        for ti, target in enumerate(('margin', 'vector')):
            y = dev['finite_scores'][:, ti]; baseline = np.empty_like(y)
            for angle in np.unique(factors[:, 3]):
                for c in range(3):
                    train = (factors[:, 3] != angle) & (factors[:, 1] == c)
                    test = (factors[:, 3] == angle) & (factors[:, 1] == c)
                    baseline[test] = y[train].mean(0)
            denominator = float(np.mean((baseline - y) ** 2))
            for rule in ('class_angle_ridge', 'class_factor_ridge'):
                chosen = cfg[target][rule]['lambdas'].index(cfg[target][rule]['chosen'])
                error = float(np.mean((cv[f'{target}_{rule}'][chosen] - y) ** 2))
                records.append(dict(seed=seed, target=target, rule=rule, class_mean_mse=denominator,
                    ridge_mse=error, relative_mse=error / denominator, penalty=cfg[target][rule]['chosen']))
    return records


def task_scale_diagnostics(summary):
    """比较绝对交互与分类logit间隔，防止很小的交互被相对百分比放大。"""
    records = []
    for seed in run.SEEDS:
        for alpha in (.25, .5, .75, 1.):
            raw = np.load(run.RUN / f'evaluation/seed_{seed}/alpha_{alpha}/responses.npz', allow_pickle=False)
            output, labels = raw['natural'], raw['labels']
            correct = np.take_along_axis(output, np.broadcast_to(labels[:, None, None], (72, 16, 1)), axis=-1)[..., 0]
            margin = (3 * correct - output.sum(-1)) / 2
            other = output[:, 0].copy(); other[np.arange(72), labels] = -np.inf
            decision_gap = correct[:, 0] - other.max(-1)
            base = next(b for b in summary['baseline'] if b['seed'] == seed and b['alpha'] == alpha
                        and b['target'] == 'margin' and b['stratum'] == 'all')
            records.append(dict(seed=seed, alpha=alpha, interaction_rms=base['interaction_rms'],
                baseline_true_class_margin_mean=float(margin[:, 0].mean()),
                baseline_runnerup_gap_mean=float(decision_gap.mean()), baseline_runnerup_gap_min=float(decision_gap.min()),
                maximum_margin_change=float(np.max(np.abs(margin - margin[:, :1]))),
                interaction_to_mean_runnerup_gap=base['interaction_rms'] / float(decision_gap.mean()),
                natural_class_changes=base['source_class_changes']))
    return records


def make_figure(rows):
    """展示预先指定的主要比较及同一名单的强度曲线，不按结果隐藏方法。"""
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font)); plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    methods = ('class_fixed', 'class_scale', 'class_angle_ridge', 'class_factor_ridge', 'class_factor_knn', 'input_finite')
    labels = ('按类固定', '加入大小', '加入角度', '大小角度回归', '因素邻居', '逐图查询')
    for ti, target in enumerate(('margin', 'vector')):
        means = np.stack([method_values(rows, target, m, 1.).mean(0) * 100 for m in methods])
        ax = axes[0, ti]; positions = np.arange(len(methods))
        ax.bar(positions - .18, means[:, 0], .36, label='恢复剩余')
        ax.bar(positions + .18, means[:, 1], .36, label='移植误差')
        ax.set_xticks(positions, labels, rotation=18, ha='right'); ax.set_ylabel('相对于原交互的 RMS %，越小越好')
        ax.set_title(('类别 margin' if ti == 0 else '中心化输出向量') + '，全强度，预算16')
        ax.legend(); ax.grid(axis='y', alpha=.2)
    alphas = (.25, .5, .75, 1.)
    colors = ('#666666', '#2171b5', '#238b45', '#e6550d')
    for oi, outcome in enumerate(('恢复剩余', '移植误差')):
        ax = axes[1, oi]
        for method, color in zip(('class_fixed', 'class_scale', 'class_factor_ridge', 'input_finite'), colors):
            y = [100 * method_values(rows, 'margin', method, a).mean(0)[oi] for a in alphas]
            ax.plot(alphas, y, 'o-', color=color, label=RULE_NAMES[method])
        ax.axvline(.5, color='#aaaaaa', ls=':', lw=1)
        ax.set_title('同一名单跨强度的 ' + outcome + '，margin')
        ax.set_xlabel('源干预强度 α，0.5为选点条件'); ax.set_ylabel('相对 RMS %，越小越好')
        ax.set_xticks(alphas); ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle('新面板72图与六个冻结教师的均值，因素规则使用已知类别及因素', fontsize=14)
    FIGURE.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIGURE, dpi=180); plt.close(fig)


def build():
    """数值先落入JSON和CSV，中文报告的每个结论均对应明确比较。"""
    audit = teacher.read(run.RUN / 'independent_audit.json'); assert audit['status'] == 'PASS'
    summary = teacher.read(run.RUN / 'summary.json'); rows = summary['methods']
    assert not (run.RUN / 'delivery_manifest.json').exists(), 'Preserve delivered run'
    rules = tuple(run.factors_model.RULES) + ('input_finite', 'random_mean', 'all_shortlist', 'all_reachable')
    paired_records = [paired(rows, target, rule, alpha) for target in ('margin', 'vector')
                      for alpha in (.25, .5, .75, 1.) for rule in rules if rule != 'class_fixed']
    paired_records += [paired(rows, 'margin', rule, 1., 'class', c) for c in range(3)
                       for rule in ('class_scale', 'class_angle_ridge', 'class_factor_ridge', 'class_factor_knn', 'input_finite')]
    forecast = forecast_diagnostics(); development_cv = development_cv_baselines(); task_scale = task_scale_diagnostics(summary)
    primary = paired(rows, 'margin', 'class_factor_ridge', 1.)
    teacher.dump(run.RUN / 'descriptive_analysis.json', dict(utc=teacher.now(), primary=primary,
        paired_comparisons=paired_records, forecast_diagnostics=forecast, development_cv_baselines=development_cv,
        task_scale_diagnostics=task_scale,
        statistics='six paired teacher summaries; deterministic shared input panel; descriptive follow-up'))
    with (run.RUN / 'method_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    make_figure(rows)
    restore, transplant = primary['gain_mean_pp']; both = primary['both_improved_teachers']
    direction = '减少' if restore >= 0 else '增加'
    direction_t = '减少' if transplant >= 0 else '增加'
    opening = (f'新面板上的预设主要比较显示：大小与角度联合回归相对于按类固定集合，'
               f'平均恢复剩余交互{direction} {abs(restore):.2f} 个百分点，移植误差{direction_t} {abs(transplant):.2f} 个百分点；'
               f'六个教师中 {both} 个同时改善。该结论针对预算16、全强度干预和类别 margin。'
               '规则使用已知因素，因此只能检验因素是否有助于预测中介选择，不能据此声称已发现通用语义回路。')
    input_primary = paired(rows, 'margin', 'input_finite', 1.)
    evidence = (f"逐图半强度查询参照在全强度下，相对于按类固定集合平均改善恢复 {input_primary['gain_mean_pp'][0]:.2f}、"
                f"移植 {input_primary['gain_mean_pp'][1]:.2f} 个百分点，六教师中 {input_primary['both_improved_teachers']} 个同时改善。"
                '目前可测的输入适应收益仍需要额外查询，平滑因素回归尚未建立稳定的预测规律。')
    text = ['# dSprites 因素预测与干预强度迁移\n', opening + '\n', evidence + '\n',
      '## 研究问题与实验逻辑\n',
      '同一形状类别内，前层四个坐标的组合干预产生的非线性交互，是否需要由不同后层坐标传递？如果需要，能否只根据大小和角度提前调整中介集合，而不逐图查询后层修补效果？本轮将“名单会变”与“变动有帮助”分开检验。\n',
      '沿用六个已训练的三类形状分类残差网络。源为 block1 完整输出的四个冻结标量坐标，中介为 block2 完整输出。每图执行16种源组合，从中求6对坐标在4种背景下的24个条件差分。最终模型按原卷积、ReLU、残差连接和读出真实运行。\n',
      '对候选集合做恢复：仅将集合内状态还原为该图无干预值；做移植：仅将集合内状态置为源干预后的值，其余保持无干预值。恢复残余和移植重建误差各除以原交互 RMS；两项越小越好。汇总先在同一教师内平均平方量并开方，再平均六教师比率，不能把24个共享组合的差分当24个独立样本。\n',
      '选择器的线性回归只预测候选坐标的修补收益。它没有把被研究网络线性化；联合修补效果仍需完整非线性网络验证，单点收益也不被假定可加。\n',
      '对一个候选坐标 m，开发目标是原交互均方量减去其单点恢复残余与移植误差均方量的平均，即 `s_m = mean(I²) − 0.5 × [mean(R_m²) + mean((T_m−I)²)]`。分数越大表示单点修补越有用。规则预测64个分数，再取前16名；最终检验的是这16个坐标共同修补的实际作用。\n',
      '## 开发数据与新面板\n',
      '拟合仅使用旧开发面板144图在 α=0.5 下的64候选单点恢复与移植分数；该面板是教师训练集的子集。旧评价结果已经影响研究方向，因此本轮另选72张与训练、验证、旧评价及旧平移面板均无像素重复的图。\n',
      '新面板固定位置索引 (15,15)，每种形状使用同样的6种大小与4个角度索引 2、14、19、27。它们由像素去重及方向覆盖规则确定；角度特征读取官方弧度值。大小与角度的组合未用于旧中心面板，但其他位置的相同角度可能出现在训练中，所以这不是全局旋转域外测试。\n',
      '所有教师的因素规则与名单均先冻结，再开启新图半强度的逐图参照查询；参照名单全部冻结后才开启全强度评价。原教师不重训，源坐标与共同64候选不重选。\n',
      '## 提前预测中介的规则\n',
      '| 规则 | 使用的开发信息 | 新图网络查询 |\n|---|---|---|',
      '| 全局固定 | 跨类别平均单点收益 | 0 |',
      '| 按类固定 | 每类48图平均收益 | 0 |',
      '| 按类与大小查表 | 每类每大小8个开发角度的平均收益 | 0 |',
      '| 按类与角度回归 | 三阶周期正弦余弦特征 | 0 |',
      '| 大小角度联合回归 | 大小二次项与三阶周期角度特征及乘积，共20项 | 0 |',
      '| 因素邻居查表 | 同类中大小与周期角度距离最近4个开发条件 | 0 |',
      '| 逐图半强度查询 | 新图64候选的单点修补效果 | 2048个尾部输出向量，另有16个自然组合 |\n',
      '两个回归均按整组开发角度留一交叉验证，标准化仅在各拟合折内进行；正则系数从0.0001至10的预设六值中按单点分数预测MSE选择。全局及按类固定基线精确复用原排序。主要比较的方法最终选16个坐标；全部64候选和整个可达集合属于更大预算的数值参照。\n',
      '各按类规则及查表使用真实类别。中心化输出向量的投影本身不使用标签，但按类选择器仍使用真实类别；不能把整个方法称为标签无关。\n',
      '开发角度留一对照已经显示平滑近似的局限：联合回归的单点收益预测MSE在 margin 的六教师中只有一个低于对应开发折的按类均值对照，在向量目标上六个都高于均值对照。两个回归的正则强度均选到预设网格最大值10。没有据此调整已经冻结的规则；回归失败只能否定本轮近似的有效性，不能否定所有因素驱动机制。\n',
      '## 全强度结果\n',
      '表中为六教师均值，数值均是百分比，越小越好；随机参照先平均每教师四次随机名单。\n',
      '| 规则 | margin 恢复剩余 | margin 移植误差 | 向量恢复剩余 | 向量移植误差 |\n|---|---:|---:|---:|---:|']
    for rule in rules:
        margin = method_values(rows, 'margin', rule, 1.).mean(0) * 100
        vector = method_values(rows, 'vector', rule, 1.).mean(0) * 100
        text.append(f'| {RULE_NAMES[rule]} | {margin[0]:.2f} | {margin[1]:.2f} | {vector[0]:.2f} | {vector[1]:.2f} |')
    text += ['\n预设主要比较的逐教师改善如下。正数表示误差下降，负数表示变差；没有删除结果较差的教师。\n',
      '| 教师种子 | 恢复改善百分点 | 移植改善百分点 |\n|---|---:|---:|']
    for seed, gain in zip(run.SEEDS, primary['gain_by_teacher_pp']): text.append(f'| {seed} | {gain[0]:+.2f} | {gain[1]:+.2f} |')
    text += ['\n## 同一集合跨强度的表现\n',
      '开发分数和逐图参照均来自 α=0.5。下表每个强度都使用原先冻结的同一名单；α=0.5 为选点条件内成绩，α=0.25、0.75、1 才显示强度变化后的行为。\n',
      '| α | 按类固定 恢复和移植 | 联合因素回归 恢复和移植 | 逐图查询 恢复和移植 |\n|---|---:|---:|---:|']
    for alpha in (.25, .5, .75, 1.):
        cells = [method_values(rows, 'margin', rule, alpha).mean(0) * 100 for rule in ('class_fixed', 'class_factor_ridge', 'input_finite')]
        text.append(f'| {alpha:.2f} | ' + ' | '.join(f'{v[0]:.2f} / {v[1]:.2f}' for v in cells) + ' |')
    text += ['\n## 同类内的差异\n', '| 形状 | 联合因素回归 恢复改善 | 联合因素回归 移植改善 | 同时改善教师数 |\n|---|---:|---:|---:|']
    for c, name in enumerate(('正方形', '椭圆', '心形')):
        p = paired(rows, 'margin', 'class_factor_ridge', 1., 'class', c)
        text.append(f"| {name} | {p['gain_mean_pp'][0]:+.2f} | {p['gain_mean_pp'][1]:+.2f} | {p['both_improved_teachers']}/6 |")
    base = [b for b in summary['baseline'] if b['alpha'] == 1. and b['target'] == 'margin' and b['stratum'] == 'all']
    changes = sum(b['source_class_changes'] for b in base)
    absolute = [r for r in task_scale if r['alpha'] == 1.]
    text += ['\n## 证据的边界\n',
      f"新图无干预分类准确率在六教师中为 {min(b['accuracy'] for b in base)*100:.2f}% 至 {max(b['accuracy'] for b in base)*100:.2f}%。全强度下六教师×72图×15种非空源组合共6480次干预中，有 {changes} 次改变了无干预时的预测类别。这里测量的主要是logit交互的传递，不能自动提升为决定形状识别的必要机制。\n",
      f"全强度 margin 交互的模型内RMS在六教师间为 {min(r['interaction_rms'] for r in absolute):.6g} 至 {max(r['interaction_rms'] for r in absolute):.6g}；其与该教师平均原始正确类对第二名logit间隔的比值为 {min(r['interaction_to_mean_runnerup_gap'] for r in absolute)*100:.4f}% 至 {max(r['interaction_to_mean_runnerup_gap'] for r in absolute)*100:.4f}%。该比值只是尺度对照，不是分类效果损失。\n",
      '本轮因素预测结果只针对固定四个源坐标、共同64候选和当前架构。三个形状、四个新角度以及六种已见大小不足以证明通用机制；各教师还共享同一面板。人工混合中间状态可能离开自然图像对应的状态分布。其他因素规则为预先列出的诊断比较，不能按最优结果回头改写预设主要比较。\n',
      '单点修补收益预测正确也不保证16点联合修补正确；这正是需要完整恢复与移植评价的原因。整可达集合的近零误差只验证计算切口的完整性，不构成稀疏机制发现。\n',
      '## 下一阶段的任务对照\n',
      '下一步优先检验任务关联：将同大小、角度和位置的两种不同形状配对，使用真实对照图的前层状态作为替换值，同时测量类别对比、预测类别和组合交互。整个源层换为对照图状态后，下游输出应精确等于对照图输出，这是可直接核验的正对照。\n',
      '随后用开发数据规定源组、中介候选与预算，在另一批像素不重叠的配对图上检查：恢复中介能否阻断形状对比的改变，移植中介能否重现该改变，同时是否消除或重现组合交互。源组和中介预算的单位需要明确记录，不能把整通道的干预与当前单个空间坐标的干预当作同等预算。\n',
      '任务对照若成立，再用下游门控状态（ReLU单元的正值或置零开关）检验半强度集合为何在更强干预下失效；仅凭大小角度查表的改善不足以命名一条语义回路。现有教师、16组合设计、双向修补和逐层分解继续作为基础工具，本轮数据保留为局部干预的边界证据。\n',
      '## 核验与复现\n',
      f"独立审计通过：重算 {audit['metric_values_checked']:,} 个逐图均方量，核对全部交叉验证折、名单映射、恢复移植端点与逐层交互分解。旧交付 {audit['old_delivery_files_unchanged']} 个文件的哈希保持一致。\n",
      '代码依次运行 `run_dsprites_factor_transfer.py` 的 init、fit、freeze、select、seal、evaluate、summarize 阶段，再运行 `audit_dsprites_factor_transfer.py` 与本报告脚本。现有运行目录具有防覆盖检查；重新开展研究应使用新的运行目录并在评价前重新冻结协议。完整复现依赖本地官方dSprites数组、旧教师与旧开发响应。\n',
      '[冻结协议](../runs/dsprites_factor_transfer_20261007/protocol.json) · [逐项结果表](../runs/dsprites_factor_transfer_20261007/method_summary.csv) · [逐教师分析](../runs/dsprites_factor_transfer_20261007/descriptive_analysis.json) · [独立审计](../runs/dsprites_factor_transfer_20261007/independent_audit.json)\n',
      '![全强度比较和强度变化](figures/dsprites_factor_transfer_20261007.png)\n']
    REPORT.write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps(dict(report=str(REPORT), primary=primary, figure=str(FIGURE)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    build()
