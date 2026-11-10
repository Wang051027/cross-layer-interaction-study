"""用稠密差分矩阵和输出方向的内积，从保存的响应重算跨层指标。"""
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RUN = BASE/'runs/cross_layer_20260928'


def read(path):
    """读取 UTF-8 审计记录。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    """核验保存后的原始字节，不仅检查文件存在。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(value):
    """用欧氏范数除元素数平方根，独立实现总体 RMS。"""
    return float(np.linalg.norm(value.ravel())/np.sqrt(value.size))


def quotient(a, b):
    """与协议一致处理不可定义的近零分母，保留为 None。"""
    return a/b if b>1e-9 else None


def main():
    """检查冻结、全部数组/逐输入指标与旧原件，写独立核验记录。"""
    protocol=read(RUN/'protocol.json')
    assert digest(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for name, expected in protocol['sources'].items():
        assert digest(BASE/name)==expected and digest(RUN/'source'/name)==expected
    for name, expected in {**protocol['old_sources'],**protocol['input_hashes']}.items():
        assert digest(BASE/name)==expected
    frozen=read(RUN/'freeze.json')
    assert digest(RUN/'freeze.json')==(RUN/'freeze.sha256').read_text().strip()
    assert frozen['utc']<read(RUN/'evaluation_opened.json')['utc']
    for name, expected in frozen['files'].items(): assert digest(RUN/name)==expected
    for entry in protocol['preserved_old_delivery']: assert digest(BASE/entry['path'])==entry['sha256']
    summary=read(RUN/'summary.json')
    methods={(r['seed'],r['alpha'],r['method'],r['budget']):r for r in summary['methods']}
    bases={(r['seed'],r['alpha']):r for r in summary['baseline']}
    gens={(r['seed'],r['alpha'],r['stage']):r for r in summary['generation']}
    with (RUN/'per_input.csv').open(encoding='utf-8-sig',newline='') as stream:
        per_input={(int(r['seed']),float(r['alpha']),r['method'],int(r['budget']),int(r['input_id'])):r
                   for r in csv.DictReader(stream)}
    errors=[]; primitive_errors=[]; ratio_errors=[]; amplified_ratios=[]; checks=0
    arithmetic_scale=1.
    def compare(actual, expected, denominator=None, context=None):
        """null 与空 CSV 单元格必须对应；数值按严格绝对容差比较。"""
        nonlocal checks
        checks+=1
        if expected is None:
            assert actual is None or actual=='', (actual,expected)
        else:
            error=abs(float(actual)-float(expected)); errors.append(error)
            # 不同运算次序的微小 logit 舍入误差在极小交互分母下会被放大。
            # 比率另以已核验原始 RMS 严格复核除法；这里记录独立路径的放大差。
            tolerance=1e-9
            if denominator is not None and denominator>1e-9:
                tolerance+=128*np.finfo(float).eps*arithmetic_scale*(1+abs(float(expected)))/denominator
                ratio_errors.append(error)
                if error>=1e-9:
                    amplified_ratios.append(dict(context=context,difference=error,denominator=denominator,bound=tolerance))
            else:
                primitive_errors.append(error)
            assert error<tolerance, (actual,expected,error,tolerance,context)
    count=0
    for seed in protocol['seeds']:
        chosen=read(RUN/'development'/f'seed_{seed}'/'selections.json')['selections']
        for alpha in protocol['strengths']:
            folder=RUN/'evaluation'/f'seed_{seed}'/f'alpha_{alpha}'
            completion=read(folder/'completion.json')
            for name, expected in completion['files'].items(): assert digest(folder/name)==expected
            with np.load(folder/'responses.npz',allow_pickle=False) as saved:
                y=saved['natural']; patched=saved['patched']; labels=saved['labels']; ids=saved['ids']
                metadata=saved['pair_metadata']; components=saved['contributions']
            arithmetic_scale=max(1.,float(np.abs(y).max()),float(np.abs(patched).max()))
            assert np.array_equal(ids,np.arange(1024,1088))
            matrix=np.zeros((24,16))
            for row,(i,j,t) in enumerate(metadata):
                matrix[row,int(t)|1<<int(i)|1<<int(j)]+=1
                matrix[row,int(t)|1<<int(i)]-=1
                matrix[row,int(t)|1<<int(j)]-=1
                matrix[row,int(t)]+=1
            weights=np.full((len(ids),10),-1/9)
            weights[np.arange(len(ids)),labels]=1
            yi=np.einsum('pm,nmd->npd',matrix,y)
            scalar=np.einsum('nmd,nd->nm',y,weights)
            interaction=np.einsum('pm,nm->np',matrix,scalar)
            delta=scalar[:,1:]-scalar[:,:1]
            den=norm(interaction); vector_den=norm(yi)
            base=bases[seed,alpha]
            expected_base=dict(interaction_rms=den,effect_rms=norm(delta),
                interaction_to_effect=quotient(den,norm(delta)),origin_interaction_rms=norm(interaction[:,metadata[:,2]==0]),
                vector_interaction_rms=vector_den,active_interaction_inputs=int((np.linalg.norm(interaction,axis=1)/np.sqrt(24)>1e-9).sum()))
            for name,value in expected_base.items(): compare(base[name],value)
            compare(np.abs(components.sum(1)-yi).max(),0.)
            for stage_index,stage in enumerate(protocol['stages']):
                projected=np.einsum('npd,nd->np',components[:,stage_index],weights)
                compare(gens[seed,alpha,stage]['output_component_rms'],norm(projected))
                compare(gens[seed,alpha,stage]['component_to_total'],quotient(norm(projected),den))
            for j,spec in enumerate(chosen):
                restored=patched[:,j,0]; transplanted=patched[:,j,1]
                rv=np.einsum('pm,nmd->npd',matrix,restored)
                tv=np.einsum('pm,nmd->npd',matrix,transplanted)
                rs=np.einsum('nmd,nd->nm',restored,weights)
                ts=np.einsum('nmd,nd->nm',transplanted,weights)
                ri=np.einsum('pm,nm->np',matrix,rs)
                ti=np.einsum('pm,nm->np',matrix,ts)
                key=(seed,alpha,spec['method'],spec['budget'])
                expected=dict(natural_interaction_rms=den,restored_interaction_rms=norm(ri),
                    transplant_interaction_error_rms=norm(ti-interaction),restore_ratio=quotient(norm(ri),den),
                    transplant_error_ratio=quotient(norm(ti-interaction),den),
                    vector_restore_ratio=quotient(norm(rv),vector_den),vector_transplant_error_ratio=quotient(norm(tv-yi),vector_den),
                    restored_effect_rms=norm(rs[:,1:]-rs[:,:1]),
                    transplant_response_error_rms=norm((ts[:,1:]-ts[:,:1])-delta))
                for name,value in expected.items(): compare(methods[key][name],value)
                for n,input_id in enumerate(ids):
                    inp=per_input[(*key,int(input_id))]
                    nr=norm(interaction[n]); dr=norm(delta[n])
                    expected_input=dict(interaction_rms=nr,effect_rms=dr,interaction_to_effect=quotient(nr,dr),
                        restored_interaction_rms=norm(ri[n]),transplant_error_rms=norm(ti[n]-interaction[n]),
                        restore_ratio=quotient(norm(ri[n]),nr),transplant_error_ratio=quotient(norm(ti[n]-interaction[n]),nr))
                    divisions={'interaction_to_effect':('interaction_rms','effect_rms'),
                        'restore_ratio':('restored_interaction_rms','interaction_rms'),
                        'transplant_error_ratio':('transplant_error_rms','interaction_rms')}
                    for name,value in expected_input.items():
                        if name in divisions:
                            numerator_name,denominator_name=divisions[name]
                            # 首先检查 CSV 的比率确实由其原始数值算出；这项不放宽容差。
                            compare(inp[name],quotient(float(inp[numerator_name]),float(inp[denominator_name])))
                            compare(inp[name],value,dr if name=='interaction_to_effect' else nr,
                                    f'{key}/{int(input_id)}/{name}')
                        else:
                            compare(inp[name],value)
                    count+=1
    assert count==len(per_input)==16128
    result=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),scalar_comparisons=checks,
        per_input_rows=count,teacher_method_rows=len(methods),maximum_absolute_difference=max(errors),
        maximum_primitive_or_direct_division_difference=max(primitive_errors),
        maximum_independent_ratio_difference=max(ratio_errors),
        amplified_ratio_comparisons=len(amplified_ratios),
        largest_amplified_ratio_cases=sorted(amplified_ratios,key=lambda r:-r['difference'])[:8],
        ratio_check='raw quantities and recorded numerator/denominator division verified at1e-9; independent ratio arithmetic allows explicit128eps*logit_scale*(1+ratio)/denominator roundoff bound; experimental floor unchanged',
        old_delivery_files_preserved=len(protocol['preserved_old_delivery']),calibration_model_access=False,
        implementation='dense24x16 pair-difference matrix; class projection by vector dot product; no runner math imported')
    (RUN/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
