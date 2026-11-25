"""从固定图像和模型权重复跑 11 月两轮实验，逐项比对发布记录。"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent.parent
NOV10 = HERE.parent / "output_bottleneck_20261110"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, help="含冻结权重和图像的研究目录或公开仓库根目录")
    parser.add_argument("--out", type=Path, help="本次复跑的新目录")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    assets = args.assets or next((p for p in HERE.parents if (p/"reproduction/held_condition").is_dir()), BUNDLE)
    assets = assets.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = (args.out or (assets/"local_output"/("final-"+timestamp))).resolve()
    dest.mkdir(parents=True,exist_ok=False)
    old, new = dest/"nov10", dest/"nov25"
    originals = BUNDLE/"runs"
    environment = dict(os.environ, MPLCONFIGDIR=str(dest/"plot_cache"))
    def run(script, *arguments):
        subprocess.run([sys.executable,"-B","-X","utf8",str(script),*map(str,arguments)],env=environment,check=True)

    run(NOV10/"run_experiment.py","--assets",assets,"--out",old,"--device",args.device)
    run(NOV10/"analyze.py",old)
    run(NOV10/"audit_gate_context.py","--assets",assets,"--run",old,"--device","cpu")
    run(NOV10/"plot_results.py",old)
    run(HERE/"run.py","predict","--assets",assets,"--out",new,"--device",args.device)
    run(HERE/"run.py","measure","--assets",assets,"--out",new,"--device",args.device,"--measurement","indexed")
    run(HERE/"analyze.py",new)
    run(HERE/"eight_corners.py","predict","--assets",assets,"--run",new,"--device",args.device,"--engine","indexed")
    run(HERE/"eight_corners.py","score","--run",new)
    run(HERE/"audit_predictions.py","--assets",assets,"--run",new)
    run(HERE/"figure.py",new)
    run(HERE/"verify_runs.py","--primary",originals/"dsprites_gate_forecast_20261125","--replay",new,
        "--nov10",originals/"dsprites_output_bottleneck_20261110","--nov10-replay",old,"--out",dest/"comparison.json")
    print("PASS",dest/"comparison.json")


if __name__ == "__main__":
    main()
