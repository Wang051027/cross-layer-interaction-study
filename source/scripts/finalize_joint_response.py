# 中文阅读说明｜用途：更新本轮报告入口，并建立交付文件的哈希清单；--check 为只读核验。
# 数据流：输入已完成的分析、报告和核验记录；交付模式会更新 README、RUNBOOK 和状态文件。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Publish local navigation and hash the reviewed analysis artifacts.

--check only reads files. Numerical outputs and frozen analysis code are never
modified by this delivery script; historical root metadata is preserved first.
"""
from pathlib import Path
import argparse,ast,hashlib,json,re,shutil,struct
from datetime import datetime,timezone
from run_joint_response import verify

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/joint_response_20260925'
MANIFEST=BASE/'audit/joint_response_manifest_20260925.json'
REPORT=BASE/'reports/联合干预响应_首轮结果_2026-09-25.md'
FIGURE=BASE/'reports/figures/joint_response_20260925.png'
ADDED_SCRIPTS=['joint_response.py','test_joint_response.py','run_joint_response.py',
               'verify_joint_response.py','report_joint_response.py','finalize_joint_response.py']

# 用途：读取本函数指定的记录文件，返回解析后的对象；JSON/CSV 形式见函数体。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：json.loads(path.read_text(encoding='utf-8'))。
def read(path):return json.loads(path.read_text(encoding='utf-8'))
# 用途：对文件字节计算 SHA-256，用于发现数据、源码或检查点是否改变。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：digest.hexdigest()。
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()
# 用途：把结构化记录序列化为 JSON 并写到指定路径。
# 参数：path、value；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

# 用途：检查报告引用、Python 语法和图像格式，防止交付路径或文件缺失。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：count。
def report_checks():
    text=REPORT.read_text(encoding='utf-8')
    assert '{{' not in text and '\ufffd' not in text
    count=0
    for target in re.findall(r'!?\[[^\]]*\]\(([^\n)]+)\)',text):
        if target.startswith(('https://','http://')):continue
        path=Path(target) if Path(target).is_absolute() else REPORT.parent/target
        assert path.is_file(),str(path)
        count+=1
    for filename in ADDED_SCRIPTS:
        ast.parse((BASE/'scripts'/filename).read_text(encoding='utf-8'))
    with FIGURE.open('rb') as stream:head=stream.read(24)
    assert head[:8]==b'\x89PNG\r\n\x1a\n'
    assert struct.unpack('>II',head[16:24])==(2860,2200)
    return count

# 用途：先验证结果，再备份旧状态、更新文档入口并生成交付哈希；仅适用于原完成目录。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def finalize():
    if MANIFEST.exists():raise FileExistsError('Delivery already finalized; use --check')
    verify();links=report_checks()
    summary=read(RUN/'summary.json');audit=read(RUN/'independent_verification.json')
    # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
    assert audit['status']=='PASS' and audit['summary_sha256']==sha(RUN/'summary.json')
    assert audit['implementation_sha256']==sha(BASE/'scripts/verify_joint_response.py')
    saved=RUN/'delivery_before';saved.mkdir(exist_ok=True)
    for name in ('README.md','RUNBOOK.md','RESEARCH_STATUS.json'):
        if not (saved/name).exists():shutil.copy2(BASE/name,saved/name)
    status=read(saved/'RESEARCH_STATUS.json')
    status['previous_status_date']=status['date'];status['date']='2026-09-25'
    status['previous_next_steps_20260917']=status['next_steps']
    status['last_network_evaluation_run']=status['current_run']
    status['current_run']='runs/joint_response_20260925'
    status['latest_report']='reports/联合干预响应_首轮结果_2026-09-25.md'
    status['current_stage']='review-informed descriptive reanalysis of saved coalition responses completed'
    status['latest_review_informed_analysis']=dict(
        run='runs/joint_response_20260925',status='complete_and_independently_verified',
        prior_literature_review='literature_review/神经元组合与跨层机制_证据综述.md',
        input_game_strength_records=6240,reference_output_vectors_reused=1228800,
        new_teacher_training_runs=0,new_network_forward_runs=0,independent_new_test=False,
        calibration_model_access=False,primary=dict(q=8,alpha=1.,inputs_per_teacher_position=128,
            single_query_count=9,pair_query_count=37,common_held_masks=219,
            center_pair_nrmse_mean=.0119917391798857,upper_left_pair_nrmse_mean=.0178760994587266,
            center_oracle2_nrmse_mean=.002754028364745394,upper_left_oracle2_nrmse_mean=.004401135390889848),
        maximum_state_reconstruction_difference=summary['maximum_reconstruction_difference'],
        independent_audit_maximum_difference=audit['maximum_difference'],
        limitation='Same final residual branch with one affected ReLU stage; q12 uses only four earlier validation inputs',
        interpretation='Current local games are mostly low-degree; prioritize failure boundaries and query coverage over claims of strong general high-order synergy')
    status['next_steps']=[
        'develop and freeze an equal-query-budget dispersed coalition design before any new intervention evaluation',
        'validate low-order prediction failures and prospective state diagnostics on newly fixed intervention inputs',
        'after the local failure question is resolved, test earlier interventions with multiple nonlinear layers and external architectures']
    dump(BASE/'RESEARCH_STATUS.json',status)
    old=(saved/'README.md').read_text(encoding='utf-8')
    old=old.replace('更新：2026-09-17。','更新：2026-09-25。',1)
    insert='''
<!-- JOINT_RESPONSE_LATEST -->

最新进展是文献综述后的 [联合干预响应首轮诊断](reports/联合干预响应_首轮结果_2026-09-25.md)。复用六个教师的历史完整响应，分析单独/成对查询的组合预测能力，并用全表最佳低阶投影区分查询覆盖与真正高阶残差。没有新训练或网络前向，原实验指标保留。

主要 8 单元、完全置零设置中，成对预测的误差为中心 1.20%、左上 1.79%；全表最佳二阶残差为 0.275%、0.440%，但后者使用全部响应，不是预测成绩。个别输入仍有明显相对误差。目前优先研究低阶模型的失效边界和查询覆盖，不把局部加速或分组消融直接当作语义机制创新。

本轮 6,240 条指标、72 条教师汇总及独立直接投影检查均通过。结果目录为 `runs/joint_response_20260925`，协议、源码、逐图像数据、图表及最终交付哈希均保存；复核命令见 RUNBOOK。12 单元使用不同来源的四张验证输入，仅作补充描述。5000 张校准图仍未用于模型评价。

<!-- END_JOINT_RESPONSE_LATEST -->
'''
    old=old.replace('更新：2026-09-25。','更新：2026-09-25。\n'+insert,1)
    old=old.replace('最新结果见 [可翻转状态筛选与精确组合计算]','9 月 17 日阶段结果见 [可翻转状态筛选与精确组合计算]',1)
    old=old.replace('当前状态以本报告和 RESEARCH_STATUS 为准。','本节保留当时的加速验证结果，当前研究状态见页首与 RESEARCH_STATUS。',1)
    (BASE/'README.md').write_text(old,encoding='utf-8')
    runbook=(saved/'RUNBOOK.md').read_text(encoding='utf-8')
    runbook+='''

## 2026-09-25 文献综述后的联合干预诊断

最新分析为 `runs/joint_response_20260925`。它读取此前 switch 阶段的完整响应，不运行 PyTorch 模型。主分析 q=8，完全置零；q=4/8 用既有 128 张测试输入，q=12 用四张既有验证输入，不把两者作纯规模比较。

以下命令已执行完成，列出供审计；init、run 和独立核验器拒绝覆盖完成输出，请勿原地重跑生成命令。

```powershell
& $researchPython -B -X utf8 "$researchScripts\\test_joint_response.py"
& $researchPython -B -X utf8 "$researchScripts\\run_joint_response.py" --stage init
& $researchPython -B -X utf8 "$researchScripts\\run_joint_response.py" --stage run
& $researchPython -B -X utf8 "$researchScripts\\verify_joint_response.py"
& $researchPython -B -X utf8 "$researchScripts\\report_joint_response.py"
& $researchPython -B -X utf8 "$researchScripts\\finalize_joint_response.py"
```

日常只读核验使用：

```powershell
& $researchPython -B -X utf8 "$researchScripts\\run_joint_response.py" --stage verify
& $researchPython -B -X utf8 "$researchScripts\\finalize_joint_response.py" --check
```

后一个命令验证新交付清单、结果文件及输入/源码冻结哈希，不重新拟合模型。原始三份计算源码已固定到运行目录的 source 子目录，报告生成器与独立核验器另由最终清单记录。修改分析问题须另建运行目录与协议；不能修改已冻结的程序再续算。

单独/成对查询是在空联盟附近拟合的多项式，统一评价 |T|>2 的留出组合；Walsh 最佳低阶投影使用完整响应，只是诊断。旧测试响应在此次研究前已经用过，因此新的数值协议不构成独立预注册或新测试数据。

交付清单为 `audit/joint_response_manifest_20260925.json`。更新前的 README、RUNBOOK 和 RESEARCH_STATUS 另存到本次运行目录的 delivery_before；旧模型、响应数组及其数值结论均保留。
'''
    (BASE/'RUNBOOK.md').write_text(runbook,encoding='utf-8')
    paths=set(p for p in RUN.rglob('*') if p.is_file() and 'mpl_cache' not in p.parts)
    paths.update(BASE/'scripts'/filename for filename in ADDED_SCRIPTS)
    paths.update([REPORT,FIGURE,BASE/'reports/联合干预响应_分析协议_2026-09-25.md',
                  BASE/'README.md',BASE/'RUNBOOK.md',BASE/'RESEARCH_STATUS.json'])
    entries=[dict(path=p.relative_to(BASE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
    dump(MANIFEST,dict(utc=datetime.now(timezone.utc).isoformat(),status='PASS',scope='Joint-response descriptive reanalysis and local delivery',
        files=len(entries),bytes=sum(p['bytes'] for p in entries),local_report_links_checked=links,
        visual_review=dict(file=FIGURE.relative_to(BASE).as_posix(),dimensions=[2860,2200],status='visually inspected: labels, legend, axes and footnotes readable'),
        numerical_audit='runs/joint_response_20260925/independent_verification.json',
        source_input_files_preserved=148,new_model_evaluations=0,entries=entries))
    check()

# 用途：只读回查最终清单、原输入和报告引用，文件变化时停止。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def check():
    verify();links=report_checks();manifest=read(MANIFEST)
    for entry in manifest['entries']:
        p=BASE/entry['path']
        assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],entry['path']
    print(json.dumps(dict(status='PASS',delivery_files=manifest['files'],local_report_links=links,
        immutable_input_files=148,manifest=str(MANIFEST)),ensure_ascii=False))

# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=='__main__':
    # 解析运行阶段/路径等命令行参数；具体允许的阶段由 choices 或分派逻辑限定。
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    check() if args.check else finalize()
