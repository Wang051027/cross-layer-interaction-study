"""封存下游定位交付；不覆盖旧交付或自动发布GitHub。"""
from __future__ import annotations
import json
import shutil
import dsprites_teacher as teacher
from run_dsprites_downstream_routes import BASE,RUN,verify_protocol,verify_selection,verify_folder
from report_dsprites_downstream_routes import REPORT
from verify_dsprites_downstream_routes_delivery import CODE


def finalize():
    """全部核验通过才复制来源快照并生成SHA清单；进度本体保留可变。"""
    assert not (RUN/'delivery_manifest.json').exists()
    protocol=verify_protocol();verify_selection()
    for name in ('analysis','claim_checks','report_checks','independent_audit','table_verification','visual_review'):
        assert teacher.read(RUN/(name+'.json'))['status']=='PASS'
    assert not (RUN/'.plot_cache').exists()
    for rec in protocol['cohort']:
        for stage in ('discovery','evaluation'):verify_folder(RUN/f"{stage}/record_{rec['record']:03d}")
    figures=teacher.read(RUN/'report_checks.json')['figures']
    copies=[(BASE/'scripts'/name,RUN/'source/scripts'/name) for name in CODE]
    copies += [(REPORT,RUN/'source/reports'/REPORT.name),(BASE/'CURRENT_RESEARCH.md',RUN/'source/CURRENT_RESEARCH.md')]
    # 保留reports下的图目录层级，来源快照中的报告图片也能正确解析。
    copies += [(BASE/name,RUN/'source'/name) for name in figures]
    for origin,target in copies:
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert teacher.sha(target)==teacher.sha(origin)
        else:shutil.copy2(origin,target)
    files=[p for p in RUN.rglob('*') if p.is_file()]
    files += [BASE/'scripts'/name for name in CODE]+[REPORT]+[BASE/name for name in figures]
    files += [BASE/'reports/dsprites_downstream_routes_preflight_20261025.json']
    records=[]
    for path in sorted(set(files)):
        assert path.resolve().is_relative_to(BASE.resolve())
        records.append(dict(path=str(path.relative_to(BASE)).replace('\\','/'),bytes=path.stat().st_size,sha256=teacher.sha(path)))
    teacher.dump(RUN/'delivery_manifest.json',dict(status='PASS',utc=teacher.now(),files=records,old_delivery_entries=1711,
                 mutable_progress='live CURRENT_RESEARCH.md excluded;dated snapshot included'))
    digest=teacher.sha(RUN/'delivery_manifest.json');(RUN/'delivery_manifest.sha256').write_text(digest+'\n',encoding='ascii')
    return dict(status='PASS',files=len(records),bytes=sum(r['bytes'] for r in records),manifest_sha256=digest)


if __name__=='__main__':print(json.dumps(finalize()),flush=True)
