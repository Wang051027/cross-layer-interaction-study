"""开发条件筛选通道，留出条件追踪阻断后的三阶响应。"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gate_forecast_20261125"))
from run import FIT, PROTOCOL, case_key, checksums, js, location, put, read_csv, sha, stamp, verify, write_csv

PRIOR_RUN = "dsprites_gate_forecast_20261125"
PRIOR_PREDICTIONS = "526d749626b97d37d3e50bf371aca9da8d630b939d2d8f30f12073970636e4d4"
PRIOR_EIGHT = "596a9ee8885deaf3fee097efd6ce75f57f2a2b5e119a4c86db61d0feb657cd72"
ARMS = ("target", "random1", "random2")
LAYER_NAMES = ("first", "last")
BUDGETS = (1, 4, 8)


class Inputs:
    def __init__(self, root, device):
        frozen, reference, src = location(root)
        sys.path.insert(0, str(src))
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import dsprites_teacher as teacher
        import gates
        import torch
        teacher.configure()
        torch.set_grad_enabled(False)
        self.gates, self.teacher, self.torch = gates, teacher, torch
        self.rule, self.device = gates.rule, device
        self.frozen, self.reference = frozen, reference
        bundle = root/"reproduction/final" if (root/"reproduction/held_condition").is_dir() else root
        self.previous = bundle/"runs"/PRIOR_RUN
        verify(reference, {"protocol.json":PROTOCOL, "fit_freeze.json":FIT})
        origin, fit_seal = js(reference/"protocol.json"), js(reference/"fit_freeze.json")
        verify(frozen,{name.removeprefix("runs/"):digest for name,digest in origin["inputs"].items()})
        verify(reference,fit_seal["files"])
        verify(self.previous,{"prediction_seal.json":PRIOR_PREDICTIONS,
                              "eight_corners/measurement_seal.json":PRIOR_EIGHT})
        prior_seal = js(self.previous/"prediction_seal.json")
        verify(self.previous,{"protocol.json":prior_seal["protocol"],
                              "forecast_sites.npz":prior_seal["files"]["forecast_sites.npz"]})
        with np.load(frozen/"dsprites_shape_donor_20261010/panel/data.npz",allow_pickle=False) as archive:
            self.panel = {key:archive[key] for key in ("images","pairs","labels","factors")}
        self.split = self.rule.split_panel(self.panel)
        assert self.panel["images"].shape == (36,64,64)
        assert all(self.split[k] == origin["split"][k]["cases"] for k in self.split)
        self.seeds = origin["seeds"]
        code = checksums(HERE, ("DESIGN.md","experiment.py","paths.py"))
        code["prior_gates"] = sha(HERE.parent/"gate_forecast_20261125/gates.py")
        self.spec = dict(seeds=self.seeds,split=self.split,weights_and_panel=origin["inputs"],fits=fit_seal["files"],
                         prior_forecast_seal=PRIOR_PREDICTIONS,prior_eight_state_seal=PRIOR_EIGHT,source=code,
                         shared_code=checksums(src, js(self.previous/"protocol.json")["shared_code"].keys()))

    def model(self, seed):
        net = self.teacher.DSpritesCNN()
        name = self.frozen / f"dsprites_20260929/seed_{seed}/teacher.pt"
        net.load_state_dict(self.torch.load(name,weights_only=True,map_location="cpu"))
        return net.double().to(self.device).eval().requires_grad_(False)

    def samples(self, seed, partition):
        import paths
        net = self.model(seed)
        cases = self.split[partition]
        bank = self.rule.source_bank(net,self.panel,cases,self.device)
        development = partition == "development"
        if development:
            fit = js(self.reference/f"fit/{seed}.json")
        else:
            with np.load(self.previous/"forecast_sites.npz",allow_pickle=False) as archive:
                ids = np.flatnonzero(archive["keys"][:,0] == seed)
                archived_sites = dict(zip(map(tuple,archive["keys"][ids]),archive["sites"][ids,0]))
        for strength in ((1.0,) if development else (1.0,0.5)):
            for case in cases:
                ri, di = map(int,self.panel["pairs"][case])
                classes = tuple(map(int,self.panel["labels"][[ri,di]]))
                if development:
                    _, estimate, empty, _, _ = self.gates.forecast(net,bank[ri],bank[di],classes,strength,fit)
                    dev_sites = self.gates.point_lists(net,estimate,empty,classes,seed,strength,case)[:,0]
                sources = self.rule.mixed_sources(bank[ri],bank[di],tuple(range(15)),strength)
                pair = paths.pair_states(net,sources)
                if not development:
                    jac = self.rule.empty_jacobian(net,pair.middle[0],list(range(32)))
                    jac = jac[classes[1]]-jac[classes[0]]
                native = self.rule.tail_from_middle(net,pair.middle)
                native_score = native[:,classes[1]]-native[:,classes[0]]
                for i, triple in enumerate(self.rule.TRIPLES):
                    corners = list(self.rule.corners(triple))
                    coords = dev_sites[i] if development else archived_sites[seed,int(strength*100),case,triple]
                    state = paths.LockedBlock(net,pair,corners,coords,classes)
                    assert int(state.first_sites.sum())==64 and int(state.last_sites.sum())==64
                    key = dict(seed=seed,strength=strength,case=case,mask=triple)
                    yield key, state, None if development else jac, paths.third(native_score[corners])


def choose(inputs, out, engine):
    out.mkdir(parents=True,exist_ok=False)
    put(out/"protocol.json",dict(utc=stamp(),**inputs.spec,device=inputs.device,engine=engine,
        software=dict(numpy=np.__version__,torch=inputs.torch.__version__,python=sys.version.split()[0])))
    rankings, development = {}, []
    for seed in inputs.seeds:
        gains = {stage:np.zeros(32) for stage in LAYER_NAMES}
        count = 0
        for key, state, _, _ in inputs.samples(seed,"development"):
            first_removed = state.individual_first(engine).sum(1)
            for stage, removed in (("first",first_removed),("last",state.phi)):
                change = (2*state.y*removed-removed.square()).cpu().numpy()
                gains[stage] += change
            count += 1
        assert count == 72
        entry = {}
        for i, stage in enumerate(LAYER_NAMES):
            scores = gains[stage]/count
            ranking = np.lexsort((np.arange(32),-np.round(scores,10))).tolist()
            entry[stage] = {"target":ranking}
            for replica, arm in enumerate(ARMS[1:],1):
                rng = np.random.default_rng([20261126,seed,i,replica])
                entry[stage][arm] = rng.permutation(32).tolist()
            for rank, ch in enumerate(ranking,1):
                development.append(dict(seed=seed,layer=stage,rank=rank,channel=ch,mean_squared_gain=scores[ch]))
        rankings[str(seed)] = entry
        print("selected",seed,flush=True)
    put(out/"rankings.json",rankings)
    write_csv(out/"development.csv",development)
    put(out/"selection_seal.json",dict(utc=stamp(),records_per_model=72,
        files=checksums(out,("rankings.json","development.csv","protocol.json"))))


def measure(inputs, out, engine):
    from paths import third
    if (out/"measurement_seal.json").exists():
        raise FileExistsError("Measurement is already sealed")
    verify(out,js(out/"selection_seal.json")["files"])
    original_protocol = js(out/"protocol.json")
    for key,val in inputs.spec.items():
        if original_protocol[key] != val:
            raise ValueError(f"Changed selection input: {key}")
    orders = js(out/"rankings.json")
    cases, groups, sites, vector_keys = [], [], [], []
    arrays = {name:[] for name in ("terminal","first_removed","first4_to_last","first4_individual_to_last")}
    checks = {}
    def check(name, value):
        gap = abs(float(value))
        if not np.isfinite(gap) or gap > 1e-8:
            raise ArithmeticError((name,gap))
        checks[name] = max(checks.get(name,0.),gap)

    for seed in inputs.seeds:
        order = orders[str(seed)]
        for key,state,jac,natural in inputs.samples(seed,"test"):
            links = state.individual_first(engine)
            removed = links.sum(1)
            components, affine_error = state.split_routes(jac)
            spatial, extreme = state.spatial()
            y = float(state.y)
            check("affine_gradient",affine_error)
            check("channel_readout",state.phi.sum()-state.y)
            check("site_partition",sum(spatial["site_"+k] for k in ("locked","always_on","switching","always_off"))-y)
            check("always_off",spatial["site_always_off"])
            assert sum(spatial["count_"+k] for k in ("locked","always_on","switching","always_off"))==32768
            cases.append(dict(**key,natural=float(natural),remaining=y,**{k:float(v) for k,v in components.items()},**spatial))
            sites.extend(dict(**key,**entry) for entry in extreme)
            vector_keys.append((seed,round(100*key["strength"]),key["case"],key["mask"]))
            arrays["terminal"].append(state.phi.cpu().numpy())
            arrays["first_removed"].append(removed.cpu().numpy())
            arrays["first4_individual_to_last"].append(links[order["first"]["target"][:4]].cpu().numpy())
            configs = [(arm,b) for arm in ARMS for b in BUDGETS] + [("target",32)]
            for arm,budget in configs:
                first_ids, last_ids = [order[stage][arm][:budget] for stage in LAYER_NAMES]
                uf, vl, both, transfer = state.group(first_ids,last_ids)
                overlap = state.y-uf-vl+both
                check("actual_last_intervention",state.y-vl-state.phi[last_ids].sum())
                check("actual_joint_intervention",overlap-transfer[last_ids].sum())
                check("upstream_to_all_last",state.y-uf-transfer.sum())
                if budget == 1:
                    check("actual_first_single",state.y-uf-removed[first_ids[0]])
                if budget == 32:
                    check("all_first_equals_skip",uf-components["skip_only"])
                    check("all_last",vl)
                    check("all_both",both)
                if arm=="target" and budget==4:
                    arrays["first4_to_last"].append(transfer.cpu().numpy())
                groups.append(dict(**key,arm=arm,budget=budget,first_remaining=float(uf),last_remaining=float(vl),
                    both_remaining=float(both),first_removed=float(state.y-uf),last_removed=float(state.y-vl),
                    both_removed=float(state.y-both),overlap=float(overlap),
                    first_nonadditive=float(state.y-uf-removed[first_ids].sum())))
        print("measured",seed,flush=True)
    # 名单和全部观测就绪后再与旧八状态数值逐例对齐。
    previous = inputs.previous/"eight_corners"
    verify(previous,js(previous/"measurement_seal.json")["files"])
    old = {case_key(r):r for r in read_csv(previous/"measurements.csv") if r["arm"]=="target" and r["layer"]=="both"}
    assert set(old)=={case_key(r) for r in cases} and len(cases)==864 and len(groups)==8640
    for row in cases:
        baseline=old[case_key(row)]
        check("prior_patched_third",row["remaining"]-float(baseline["patched_third"]))
        check("prior_natural_third",row["natural"]-float(baseline["natural_third"]))
    write_csv(out/"cases.csv",cases)
    write_csv(out/"group_interventions.csv",groups)
    write_csv(out/"spatial_extremes.csv",sites)
    np.savez_compressed(out/"channels.npz",keys=np.asarray(vector_keys,dtype=np.int32),
                        **{k:np.stack(v) for k,v in arrays.items()})
    put(out/"checks.json",dict(status="PASS",device=inputs.device,engine=engine,records=len(cases),
                               channel_group_records=len(groups),max_abs=checks))
    put(out/"measurement_seal.json",dict(utc=stamp(),selection_seal=sha(out/"selection_seal.json"),
        files=checksums(out,("cases.csv","group_interventions.csv","spatial_extremes.csv","channels.npz","checks.json"))))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("select","measure","all"))
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--engine",choices=("grouped","direct"),default="grouped")
    args=parser.parse_args()
    inputs=Inputs(args.assets.resolve(),args.device)
    if args.phase in ("select","all"):
        choose(inputs,args.out.resolve(),args.engine)
    if args.phase in ("measure","all"):
        measure(inputs,args.out.resolve(),args.engine)
