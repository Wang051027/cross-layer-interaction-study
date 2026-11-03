"""在预测与实测封存之后，用原模块、独立容斥和手算尾部伴随重建数值。"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as functional
import dsprites_teacher as teacher
from run_dsprites_triple_rule import (DEFAULT,SHAPE,STRENGTHS,panel_arrays,verify_protocol,
                                      verify_seal,stage_path,load_model,atomic_json,manifest_entries)

TRIPLES = (7,11,13,14)
SIGNS = (-1,1,1,1,-1,-1,-1,1)


def cube(mask):
    return sorted((i for i in range(16) if (i & mask)==i),key=lambda i:(i.bit_count(),i))


def q_from_all(array, mask):
    """独立用容斥符号重算二阶预测；只取严格子集项。"""
    return sum((-1)**(2-i.bit_count())*array[i] for i in cube(mask)[:-1])


def independent_jacobian(model, empty):
    """以转置卷积手工传播三个logit的梯度，独立核验预测使用的J0。"""
    block = model.block3
    z = block.bn1(block.conv1(empty))
    v = block.bn2(block.conv2(z.relu())) + empty
    head = model.readout.weight[:,:,None,None] / (empty.shape[-1]*empty.shape[-2])
    v_adj = head*(v>0)
    def backward(adj,bn,conv):
        slope = bn.weight / (bn.running_var+bn.eps).sqrt()
        return functional.conv_transpose2d(adj*slope[None,:,None,None],conv.weight,padding=1)
    u_adj = backward(v_adj,block.bn2,block.conv2)
    return v_adj+backward(u_adj*(z>0),block.bn1,block.conv1)


@torch.no_grad()
def audit(run,device):
    protocol = verify_protocol(run)
    for name,prev in (('fit_freeze','protocol'),('prediction_freeze','fit_freeze'),
                      ('measurement_freeze','prediction_freeze')):
        verify_seal(run,name,prev)
    panel = panel_arrays()
    cases = protocol['split']['test']['cases']
    development_images = set(protocol['split']['development']['images'])
    assert set(panel['pairs'][cases].flatten()).isdisjoint(development_images)
    history = manifest_entries(SHAPE)
    maximum = {}
    counts = dict(triples=0, raw_logit_states=0, patch_states=0)
    def compare(key,actual,expected):
        a = torch.as_tensor(actual,device=device)
        b = torch.as_tensor(expected,device=device)
        assert a.shape==b.shape,(key,a.shape,b.shape)
        err = float((a-b).abs().max())
        assert err<1e-9, (key,err)
        maximum[key] = max(maximum.get(key,0.),err)
    for seed in teacher.SEEDS:
        fit = teacher.read(run/'fit'/f'{seed}.json')
        first,middle = fit['first'],fit['middle']
        assert len(set(first))==len(set(middle))==4
        model = load_model(seed,device)
        block = model.block2
        images = torch.as_tensor(panel['images'][:,None],device=device,dtype=torch.float64)
        sources = model.block1(model.stem(images))
        for strength in STRENGTHS:
            legacy = SHAPE/f'evaluation/seed_{seed}/alpha_{strength:.1f}/data.npz'
            assert teacher.sha(legacy)==history[legacy.relative_to(teacher.BASE).as_posix()]
            with np.load(legacy,allow_pickle=False) as f:
                legacy_logits = f['natural'][cases,:15]
            with np.load(stage_path(run,'predictions',seed,strength),allow_pickle=False) as f:
                pred = {k:f[k] for k in f.files}
            with np.load(stage_path(run,'measurements',seed,strength),allow_pickle=False) as f:
                data = {k:f[k] for k in f.files}
            compare('archived_original_logits',data['natural_logits'],legacy_logits)
            assert pred['case'].tolist()==data['case'].tolist()==cases
            for row,case in enumerate(cases):
                r,d = panel['pairs'][case]
                src = sources[r].repeat(15,1,1,1)
                for mask in range(15):
                    for g in range(4):
                        if mask & (1<<g):
                            c = slice(4*g,4*g+4)
                            src[mask,c]=(1-strength)*sources[r,c]+strength*sources[d,c]
                u = block.mid(src)
                h = block(src)
                jac = independent_jacobian(model,h[:1])[:,middle]
                compare('empty_middle',pred['h0'][row],h[0])
                compare('true_middle',data['true_middle'][row],h[list(TRIPLES)])
                label_r,label_d = int(panel['labels'][r]),int(panel['labels'][d])
                for t,mask in enumerate(TRIPLES):
                    ids = cube(mask)
                    qh,qu = q_from_all(h,mask),q_from_all(u,mask)
                    compare('QH',pred['qh'][row,t],qh)
                    compare('Qlogits',pred['qlogits'][row,t],q_from_all(legacy_logits[row],mask))
                    # 在所有通道的二阶u背景上，仅把K替换为已实测的三组u。
                    # 全卷积+BN+残差的重新执行，与预测器的4×4核实现独立。
                    hybrid_u = qu.clone()
                    hybrid_u[first] = u[mask,first]
                    selected = block.finish(hybrid_u[None],block.shortcut(src[mask:mask+1]))[0,middle]
                    correction = selected-qh[middle]
                    compare('sparse_threshold_rule',pred['middle_corrections'][row,t,0],correction)
                    by_channel = (correction[None]*jac).sum((-1,-2)).T
                    compare('channel_to_logit',pred['channel_logits'][row,t],by_channel)
                    compare('forecast_logit_residual',pred['output_corrections'][row,t,1],by_channel.sum(0))
                    score = (by_channel[:,label_d]-by_channel[:,label_r]).cpu().numpy()
                    hi = np.lexsort((middle,-abs(score)))[0]
                    lo = np.lexsort((middle,abs(score)))[0]
                    expected = [middle,[middle[hi]],[middle[lo]],*fit['random_channels']]
                    selected_masks = []
                    for p,chosen in enumerate(expected):
                        frozen = pred['patch_ids'][row,t,p]
                        # 数值在极近并列处，以封存预测本身依规则排序为准。
                        if list(frozen[frozen>=0]) != chosen:
                            assert p in (1,2)
                            assert abs(abs(score[middle.index(int(frozen[0]))])-abs(score[hi if p==1 else lo]))<1e-11
                            chosen = [int(frozen[0])]
                        flag = torch.zeros((1,32,1,1),dtype=torch.bool,device=device)
                        flag[:,chosen] = True
                        selected_masks.append(flag)
                    # 一次广播构造5套阻断和反向保留，重新执行完整原block3。
                    flags = torch.stack(selected_masks)
                    states = h[ids][None]
                    blocked = torch.where(flags,h[:1],states)
                    kept = torch.where(flags,states,h[:1])
                    fields = torch.stack((blocked,kept),1).reshape(-1,32,32,32)
                    readout = torch.cat([model.readout(model.block3(batch).mean((2,3))) for batch in fields.split(32)])
                    compare('all_patch_logits',data['patch_logits'][row,t],readout.reshape(5,2,8,3))
                    counts['triples'] += 1
                    counts['patch_states'] += 80
                counts['raw_logit_states'] += 15
        print('audit',seed,flush=True)
    assert counts==dict(triples=864,raw_logit_states=3240,patch_states=69120)
    result = dict(status='PASS',utc=teacher.now(),counts=counts,maximum_absolute_errors=maximum)
    atomic_json(run/'independent_audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT)
    args=parser.parse_args()
    assert not (args.run_dir/'delivery_manifest.json').exists()
    print(audit(args.run_dir,teacher.configure()),flush=True)
