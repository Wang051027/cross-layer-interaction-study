"""Post-result explanation checks; no fitting, selection, or result replacement."""
from structure_runtime import *
from external_experiment import DATA, csv_read

RUN = BASE / "runs/fashion_structure_20260911"


# 回看结构对照的已有结果，区分头部不活跃、局部线性与精确响应的影响。
def main():
    configure_cpu()
    # Independently check the high-error fixed-branch control against autograd.
    # Four previously fixed validation inputs, first teacher, all three positions.
    images = np.load(DATA / "official_train_images.npy", mmap_mode="r", allow_pickle=False)
    with np.load(DATA / "context_indices.npz", allow_pickle=False) as saved:
        ids = saved["validation"][:4]
    x = torch.tensor(normalized(images[ids]), dtype=torch.float64)
    model = load_gap(RUN / "gap_seed_202609071")
    bits = torch.tensor(bit_design(8), dtype=torch.float64)
    checks = []; maximum = 0.
    for layer in (1, 2, 3):
        with np.load(RUN / f"gap_seed_202609071/context_L{layer}.npz", allow_pickle=False) as saved:
            nodes = saved["nodes"]
        with torch.no_grad():
            sources = model.source(x, layer)
        local = GapLocal(model, tuple(sources.shape[1:]), nodes, layer)
        for index, value in zip(ids, sources):
            source = value[None].detach().clone().requires_grad_(True)
            logits = model.tail(source, layer)
            jacobian = torch.stack([torch.autograd.grad(logits[0, j], source, retain_graph=True)[0].flatten()[nodes]
                                    for j in range(10)], dim=1).detach()
            context = local.prepare(source.detach())
            for alpha in STRENGTHS:
                independent = (-alpha * bits * context[0][None]) @ jacobian
                fixed = local.evaluate(context, bits, alpha, "original_region")
                difference = float((independent - fixed).abs().max())
                assert difference < 1e-10
                maximum = max(maximum, difference)
                checks.append(dict(layer=layer, validation_index=int(index), alpha=alpha, max_difference=difference))
    # Show actual effect and error scales, rather than treating >100% as impossible.
    held = bit_design(8).sum(1) > 2
    scales = []
    for seed in SEEDS:
        case = RUN / f"test/gap_{seed}_L2"
        with np.load(case / "predictions.npz", allow_pickle=False) as pp, np.load(case / "references.npz", allow_pickle=False) as rr:
            reference = rr["1.0"][:, held]
            prediction = pp[prediction_key("original_region", 0, 1.)][:, held]
            scales.append(dict(seed=seed, reference_logit_rms=float(np.sqrt(np.mean(reference**2))),
                error_logit_rms=float(np.sqrt(np.mean((prediction-reference)**2))),
                prediction_logit_rms=float(np.sqrt(np.mean(prediction**2))),
                nrmse=float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference))))
    differences = csv_read(RUN / "analysis/paired_differences.csv")
    directions = []
    for layer in (1, 2):
        for k in BUDGETS:
            d = [100*float(r["greedy_minus_control"]) for r in differences if r["layer"]==str(layer)
                 and r["k"]==str(k) and r["alpha"]=="1.0" and r["control"]=="gate_weight"]
            directions.append(dict(layer=layer,k=k,lower_error=sum(v < -1e-10 for v in d),
                numerical_tie=sum(abs(v) <= 1e-10 for v in d),higher_error=sum(v > 1e-10 for v in d)))
    boundary = [read_json(RUN / f"test/cnn_{seed}_L1/completion.json")["boundary_scalars"] for seed in SEEDS]
    assert boundary == [576]*6
    result = dict(utc=now(),status="passed",scope="post-result implementation and interpretation check, no refitting",
        original_region_autograd=dict(validation_cases=24,max_difference=maximum,details=checks),
        gap_L2_original_region_scales=scales,greedy_vs_gate_weight_directions=directions,
        cnn_L1_background_scalars=dict(boundary=576,head_exact=64,head_selected=32,
            combined_exact=640,combined_selected=608,reduction_fraction=.05,
            excludes="eight source values, weights, workspace and temporary query arrays"),
        calibration_model_access=False,confirmed_test_outputs_replaced=False)
    dump(BASE / "audit/structure_interpretation_20260911.json",result)
    print(json.dumps(dict(status="passed",autograd_max_difference=maximum,paired_directions=directions,
                         gap_L2_mean_nrmse=float(np.mean([r["nrmse"] for r in scales])))))


if __name__ == "__main__":
    import json
    main()
