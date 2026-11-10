"""在固定的六通道和目标位置上枚举 64 个子集，计算六阶激活和输出差分。"""
from __future__ import annotations
import csv
import numpy as np
import torch
import dsprites_teacher as teacher
from run_dsprites_residual_routes import RUN,BASE,load,verify_seal
from test_dsprites_residual_routes import original_holds


@torch.no_grad()
def verify():
    assert not (RUN/'threshold_sixth_order.json').exists();verify_seal();check=teacher.read(RUN/'threshold_check.json');assert check['status']=='PASS'
    device=teacher.configure();model=teacher.load_teacher(teacher.SEEDS[0],device)
    for p in model.parameters():p.requires_grad_(False)
    fields=load(BASE/'runs/dsprites_layer_transfer_20261013/evaluation/seed_202609291/alpha_1.0/example_fields.npz')
    a=torch.as_tensor(fields['additive_middle'][15:16],device=device);h=torch.as_tensor(fields['natural_middle'][15:16],device=device)
    channels=check['chosen_channels'];assert len(channels)==6;coordinate=check['coordinate'];flat=coordinate[0]*1024+coordinate[1]*32+coordinate[2]
    values=a.expand(64,-1,-1,-1).clone()
    for mask in range(64):
        for j,c in enumerate(channels):
            if mask&(1<<j):values[mask,c]=h[0,c]
    b=model.block3;z=b.bn1(b.conv1(values));u=torch.relu(z);v=b.bn2(b.conv2(u))+values
    logits=model.readout(torch.relu(v).mean((2,3)));w=torch.tensor([-1.,1.,0.],device=device,dtype=a.dtype)
    ga=(b.bn1(b.conv1(a))>0).expand_as(values);ua=b.mid(a);gb=(b.bn2(b.conv2(ua))+a>0).expand_as(values)
    held=original_holds(model,values,ga,gb,np.array([flat]),None)@w;natural=logits@w
    zz=z[:,coordinate[0],coordinate[1],coordinate[2]].cpu().numpy();activation=np.maximum(zz,0);fn=natural.cpu().numpy();fh=held.cpu().numpy()
    assert np.all(zz[:63]>0) and zz[63]<0 and np.max(abs(fn[:63]-fh[:63]))==0
    signs=np.array([(-1)**(6-mask.bit_count()) for mask in range(64)])
    coefficient=lambda x:float(signs@x)
    sixth_activation=coefficient(activation);assert abs(sixth_activation+zz[63])<1e-9
    output=coefficient(fn);held_output=coefficient(fh);net=float(fn[-1]-fh[-1]);assert abs(output-held_output-net)<1e-9
    expected=next(r['point_net_removed'] for r in check['rows'] if r['group']=='leading_6');assert abs(net-expected)<1e-9
    path=RUN/'tables/six_channel_subsets.csv';assert not path.exists();path.parent.mkdir(exist_ok=True)
    rows=[dict(mask=m,order=m.bit_count(),channels=','.join(str(c) for j,c in enumerate(channels) if m&(1<<j)),preactivation=float(zz[m]),activation=float(activation[m]),
        natural_output=float(fn[m]),point_held_output=float(fh[m]),point_net_removed=float(fn[m]-fh[m])) for m in range(64)]
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='PASS',utc=teacher.now(),scope='post hoc64subsets ofsame6middlechannels atpredefined case/mask15;not source-group sixth order or universal circuit',channels=channels,coordinate=coordinate,
        activation_sixth=sixth_activation,natural_output_sixth=output,point_held_output_sixth=held_output,point_net_sixth=output-held_output,
        point_changes_only_all_six=True,rows=64,table_sha256=teacher.sha(path),threshold_check_sha256=teacher.sha(RUN/'threshold_check.json'))
    teacher.dump(RUN/'threshold_sixth_order.json',result);print(result)


if __name__=='__main__':verify()
