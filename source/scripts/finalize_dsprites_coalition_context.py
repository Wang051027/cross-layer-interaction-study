"""封存新四状态交付；已完成旧目录及旧根说明保持原样。"""
from __future__ import annotations
import json
import shutil
import dsprites_teacher as teacher
from run_dsprites_coalition_context import BASE,RUN,verify_protocol,verify_folder
from report_dsprites_coalition_context import REPORT
from verify_dsprites_coalition_context_delivery import CODE


def finalize():
    """数值、逐行表格、图形检查全部通过才封存；不包含可变进度本体。"""
    assert not (RUN/'delivery_manifest.json').exists();verify_protocol()
    for name in ('analysis.json','claim_checks.json','report_checks.json','independent_audit.json','table_verification.json','visual_review.json'):
        assert teacher.read(RUN/name)['status']=='PASS'
    assert not (RUN/'.plot_cache').exists();assert REPORT.exists() and (RUN/'README.md').exists()
    for seed in teacher.SEEDS:verify_folder(RUN/f'evaluation/seed_{seed}')
    figs=teacher.read(RUN/'report_checks.json')['figures']
    copies=[(BASE/'scripts'/n,RUN/'source/scripts'/n) for n in CODE]
    copies += [(REPORT,RUN/'source/reports'/REPORT.name),(BASE/'CURRENT_RESEARCH.md',RUN/'source/CURRENT_RESEARCH.md')]
    copies += [(BASE/name,RUN/'source/reports/figures'/name.split('/')[-1]) for name in figs]
    for origin,target in copies:
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert teacher.sha(target)==teacher.sha(origin)
        else:shutil.copy2(origin,target)
    files=[p for p in RUN.rglob('*') if p.is_file()]+[BASE/'scripts'/n for n in CODE]+[REPORT]+[BASE/name for name in figs]+[BASE/'reports/dsprites_coalition_context_preflight_20261025.json']
    records=[]
    for p in sorted(set(files)):
        assert p.resolve().is_relative_to(BASE.resolve());records.append(dict(path=str(p.relative_to(BASE)).replace('\\','/'),bytes=p.stat().st_size,sha256=teacher.sha(p)))
    teacher.dump(RUN/'delivery_manifest.json',dict(status='PASS',utc=teacher.now(),files=records,old_delivery_entries=1650,mutable_progress='live CURRENT_RESEARCH.md excluded;dated snapshot included'))
    digest=teacher.sha(RUN/'delivery_manifest.json');(RUN/'delivery_manifest.sha256').write_text(digest+'\n',encoding='ascii')
    return dict(status='PASS',files=len(records),bytes=sum(r['bytes'] for r in records),manifest_sha256=digest)


if __name__=='__main__':print(json.dumps(finalize()),flush=True)
