"""将已审计的完整结果生成中文数值附表；文字结论在主报告中另行判断。"""
from pathlib import Path
import numpy as np
from dsprites_teacher import BASE,RUN,SEEDS,read

NAMES={'global':'全局固定','class':'类别固定','input':'逐图选择',
       'gradient':'梯度评分','finite':'实际单点干预评分'}


def title(method):
    """为冻结方法名提供稳定中文显示，不改变方法定义。"""
    explicit={'all_shortlist':'全部64候选','all_reachable':'全部结构可达点',
        'random_mean_of_4':'四组同池随机选择的教师内均值',
        'margin_global_single':'全局固定／单独效应',
        'margin_class_single':'类别固定／单独效应','margin_input_single':'逐图／单独效应'}
    if method in explicit:return explicit[method]
    if '_full_' in method:return '逐图梯度／完整可达候选池'
    parts=method.split('_');return NAMES[parts[1]]+'／'+NAMES[parts[2]]


def mean_sd(values,percent=True):
    """对六教师结果求均值与样本SD，不从查询数构造伪精确性。"""
    if any(v is None for v in values):return '含未定义值；见原始表'
    v=np.asarray(values)*(100 if percent else 1)
    suffix='%' if percent else ''
    return f'{v.mean():.2f} ± {v.std(ddof=1):.2f}{suffix}'


def main():
    """所有表由保存统计生成，避免手工抄录百分比和模型胜出数。"""
    assert read(RUN/'independent_audit.json')['status']=='PASS'
    s=read(RUN/'summary.json');a=read(RUN/'descriptive_analysis.json');lines=[]
    def write(text=''):lines.append(text)
    def table(header,rows):
        write('| '+' | '.join(header)+' |');write('|'+ '|'.join(['---']*len(header))+'|')
        for row in rows:write('| '+' | '.join(map(str,row))+' |')
        write()
    def records(mode,method,k=16,alpha=1.,label=-1):
        found=[r for r in s['methods'] if r['frame']=='base' and r['alpha']==alpha and r['target']==mode
               and r['label']==label and r['method']==method and r['budget']==k]
        assert len(found)==6
        return [next(r for r in found if r['seed']==seed) for seed in SEEDS]
    write('# dSprites 迁移评估：完整数值附表\n')
    write('以下是六个独立初始化、相同架构模型的描述性统计，均值 ± 样本标准差。144张因素格点图在各方法和模型间复用，不构成144个独立自然场景。两个中介指标越小越好；百分比不是因果解释比例。\n')
    write('## 1. 模型与信号强度\n')
    rows=[]
    for seed in SEEDS:
        training=read(RUN/f'seed_{seed}'/'teacher_summary.json')
        b=next(r for r in s['baseline'] if r['seed']==seed and r['frame']=='base' and r['alpha']==1 and r['label']==-1 and r['target']=='margin')
        stage=next(r for r in a['stage_diagnostics'] if r['seed']==seed and r['target']=='margin')
        with np.load(RUN/'development'/f'seed_{seed}'/'discovery.npz') as z:pool=len(z['pool'])
        rows.append([seed,training['best_epoch'],f"{100*training['validation']['accuracy']:.2f}%",f"{100*b['accuracy']:.2f}%",
            f"{b['interaction_rms']:.6g}",f"{100*stage['interaction_over_effect']:.2f}%",f"{100*stage['prediction_changed_fraction']:.2f}%",b['active_inputs'],pool])
    table(['教师','检查点轮次','验证准确率','评价准确率','自然交互RMS','交互/干预效应RMS','源干预改变预测的比例','非零输入/144','可达池大小'],rows)
    write('“源干预改变预测”统计每图15个非空干预组合相对于该图无干预预测的变化，不是整体测试错误率，也不是只统计成对交互造成的翻转。\n')
    for mode,caption in [('margin','真实类margin'),('vector','去均值logit向量')]:
        write('## '+('2' if mode=='margin' else '3')+'. '+caption+'：完整强度，主预算16及整组参照\n')
        rows=[]
        for item in a['primary']:
            if item['target']!=mode:continue
            pools=[r['budget'] for r in s['methods'] if r['method']=='all_reachable']
            budget=f'{min(pools)}–{max(pools)}（完整可达池）' if item['method']=='all_reachable' else item['budget']
            rows.append([title(item['method']),budget,
                mean_sd(item['restore_ratio']['values']),mean_sd(item['transplant_error_ratio']['values']),mean_sd(item['joint_loss']['values'],False)])
        table(['方法','预算','恢复剩余','移植误差','联合平方损失'],rows)
    write('## 4. 预定方法的教师内配对比较\n')
    write('负差值表示候选方法误差更小。“胜出”仅是六模型中较好的个数，不是显著性检验。完整池和64候选的比较存在搜索范围差异。\n')
    rows=[]
    for r in a['paired_comparisons']:
        rows.append([r['target'],title(r['candidate'])+' 对 '+title(r['reference']),
            mean_sd(r['restore_ratio']['values']).replace('%','百分点'),mean_sd(r['transplant_error_ratio']['values']).replace('%','百分点'),
            f"{r['wins']['restore_ratio']}/6、{r['wins']['transplant_error_ratio']}/6、{r['wins']['joint_loss']}/6"])
    table(['目标','候选 对 参照','恢复比率差','移植误差差','恢复/移植/联合胜出'],rows)
    write('## 5. 预设预算与强度补充\n')
    write('α=.5也是逐图选择使用的强度，不能作为独立于选择的验证证据。表格按margin目标展示；向量目标的全部原数值见method_summary.csv。\n')
    rows=[]
    for alpha in (1.,.5):
        for k in (16,32):
            for score in ('gradient','finite'):
                row=[alpha,k,NAMES[score]]
                for scope in ('global','class','input'):
                    rec=records('margin',f'margin_{scope}_{score}',k,alpha)
                    row.append(f"{np.mean([r['restore_ratio'] for r in rec])*100:.2f}% / {np.mean([r['transplant_error_ratio'] for r in rec])*100:.2f}%")
                rows.append(row)
    table(['α','k','评分','全局：恢复/移植','按类：恢复/移植','逐图：恢复/移植'],rows)
    write('## 6. 每类结果：margin、完整强度、预算16\n')
    rows=[]
    for c,name in enumerate(('方形','椭圆','心形')):
        for scope in ('global','class','input'):
            rec=records('margin',f'margin_{scope}_finite',label=c)
            rows.append([name,NAMES[scope]+'／实际单点评分',mean_sd([r['restore_ratio'] for r in rec]),mean_sd([r['transplant_error_ratio'] for r in rec])])
    table(['形状','方法','恢复剩余','移植误差'],rows)
    write('逐大小与逐角度的全部配对分层保存在factor_strata.csv。它是描述性后处理，不能把事后较好的某个因素格子提升为原主检验。\n')
    write('## 7. 平移配对对照\n')
    write('同一原图严格向右4像素。绝对坐标组复用原源/中介位置；对齐组同步将源点移动4像素、中介点移动2格。名单不重选。下面是自然交互向量差的RMS/原图交互RMS，越小表示配对交互越接近。\n')
    rows=[]
    for seed in SEEDS:
        row=[seed]
        for mode in ('margin','vector'):
            for frame in ('translation_absolute','translation_aligned'):
                r=next(r for r in s['translation'] if r['seed']==seed and r['target']==mode and r['frame']==frame)
                row.append(f"{r['natural_relative_difference']:.6g}")
        rows.append(row)
    table(['教师','margin绝对','margin对齐','vector绝对','vector对齐'],rows)
    write('同步移动后的一致性首先是局部卷积结构与坐标对应的证据，不自动成为语义不变性的新发现；这项控制旨在避免把几何位移误判成中介改变。\n')
    write('## 8. 非线性修正的逐层输出投影\n')
    write('各分量都在同一个输出目标尺度计算，但可能抵消，所以分量RMS/总交互RMS不能相加成因果百分比。\n')
    rows=[]
    for mode in ('margin','vector'):
        selected=[r for r in a['stage_diagnostics'] if r['target']==mode]
        for j,name in enumerate(('block2第一次ReLU','block2输出ReLU','block3第一次ReLU','block3输出ReLU')):
            rows.append([mode,name,mean_sd([r['projected_stage_ratio'][j] for r in selected],False),
                mean_sd([r['mean_gate_flips'][j] for r in selected],False)])
    table(['目标','阶段','输出投影RMS/总交互RMS','每图平均翻转坐标数'],rows)
    write('## 9. 逐教师主结果\n')
    rows=[]
    for seed in SEEDS:
        row=[seed]
        for method in ('margin_global_gradient','margin_class_gradient','margin_input_gradient','margin_global_finite','margin_class_finite','margin_input_finite'):
            r=next(r for r in records('margin',method) if r['seed']==seed)
            row.append(f"{100*r['restore_ratio']:.2f}% / {100*r['transplant_error_ratio']:.2f}%")
        rows.append(row)
    table(['教师','全局梯度','按类梯度','逐图梯度','全局实际','按类实际','逐图实际'],rows)
    write('## 10. 核验记录\n')
    audit=read(RUN/'independent_audit.json');data=read(RUN/'independent_data_audit.json')
    write(f"原始响应独立核验：{audit['status']}；{audit['cases']}个模型/设置文件夹；{audit['selection_rows']:,}个选择集合。数据来源与划分核验：{data['status']}。旧三份历史交付清单保持哈希。详细误差与限制见对应JSON。\n")
    write('初版训练最后一条日志发生utc重名异常；全部模型、历史和冻结文件在异常前已保存并核验。该日志问题及安全复现入口在运行README说明，没有覆盖冻结源码或重新选择模型。\n')
    path=BASE/'reports/dSprites_逐项结果表_2026-09-29.md';path.write_text('\n'.join(lines),encoding='utf-8');print(path)


if __name__=='__main__':main()
