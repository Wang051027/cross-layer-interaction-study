"""从模型 291 的冻结权重重跑上游组合筛选、符号预判与逐点开关干预。"""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[1]
RUN = "dsprites_pair_origins_20261126"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root=next((p for p in HERE.parents if (p/"reproduction/held_condition").is_dir()), BUNDLE)
    assets=args.assets.resolve() if args.assets else root
    created=datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out=args.out.resolve() if args.out else root/"local_output"/("pair-origins-"+created)
    if out.exists():
        raise FileExistsError(out)
    def execute(script, *argv):
        subprocess.run([sys.executable,"-B","-X","utf8",str(HERE/script),*map(str,argv)],check=True)
    for script in ("select_paths.py","measure.py"):
        execute(script,"--device",args.device,"--assets",assets,"--out",out)
    execute("local_trace.py","--device",args.device,"--assets",assets,"--run",out)
    execute("coverage.py","--device",args.device,"--assets",assets,"--run",out)
    execute("score.py",out)
    execute("compare.py",BUNDLE/"runs"/RUN,out,"--out",out/"replay_comparison.json")
    execute("figure.py",out)
    print(out)


if __name__=="__main__":
    main()
