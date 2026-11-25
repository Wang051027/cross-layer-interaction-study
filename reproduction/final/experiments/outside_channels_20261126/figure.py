"""绘制组外作用的累积份额与单／双通道输出干预。"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def rows(path):
    with path.open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def main(run):
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC") if f in available), None)
    if family:
        plt.rcParams["font.family"] = family
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False})
    def label(chinese, english):
        return chinese if family else english

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 4.7), layout="constrained")
    data = [r for r in rows(run / "cumulative.csv") if float(r["strength"]) == 1.]
    x = np.arange(29)
    curves = []
    for seed in sorted({r["seed"] for r in data}):
        cell = sorted((r for r in data if r["seed"] == seed), key=lambda r: int(r["channels"]))
        y = [0.] + [float(r["cumulative_share"]) for r in cell]
        curves.append(y)
        left.plot(x, y, color="#8db1be", lw=1.15, alpha=.8)
    left.plot(x, np.mean(curves, axis=0), color="#155e77", lw=2.6, label=label("六模型均值", "Mean of six models"))
    left.plot(x, x/28, "--", color="#887055", lw=1.4, label=label("同预算随机期望", "Uniform selection"))
    left.scatter([4,8,14], np.mean(curves, axis=0)[[4,8,14]], color="#155e77", zorder=4)
    left.set(xticks=(0,4,8,14,21,28), ylim=(0,1.055), xlim=(0,28),
             xlabel=label("按开发排名依次纳入的组外通道数", "Outside channels in development rank order"),
             ylabel=label("组外传递投影的累计份额", "Cumulative share of outside projection"))
    left.yaxis.set_major_formatter(PercentFormatter(1))
    left.set_title(label("多少个通道承接了这份作用？", "How concentrated is the routed effect?"), loc="left")
    left.legend(frameon=False, loc="lower right")
    cases = rows(run / "candidate_cases.csv")
    def get(case, mask):
        return next(r for r in cases if r["seed"] == "202609291" and float(r["strength"]) == 1. and
                    r["case"] == "63" and int(r["mask"]) == mask and r["candidate"] == case)
    first, second = get("reinforce_pair",14), get("cancel_pair",7)
    positions = np.array([0,1,2,3.6,4.6,5.6])
    values = [float(r[k]) for r in (first,second) for k in ("output_first","output_second","output_both")]
    colors = ["#56958b","#56958b","#175f55","#b46847","#56958b","#175f55"]
    bars = right.bar(positions, values, width=.72, color=colors)
    right.bar_label(bars, labels=[f"{v:+.3f}" for v in values], padding=3, fontsize=9)
    right.axhline(0, color="#607078", lw=.8)
    right.set_xticks(positions, ["23","30","23 + 30","17","27","17 + 27"])
    right.set(ylim=(-.42,1.47), ylabel=label("阻断移除的三阶得分（logit）", "Removed third-order margin (logit)"),
              xlabel=label("模型 291，配对 63，左组三组掩码 14，右组掩码 7",
                           "Model 291, pair 63. Left mask 14, right mask 7"))
    right.set_title(label("同向叠加与反向抵消的两次实测", "Same-sign addition and opposite-sign cancellation"), loc="left")
    fig.savefig(run / "outside_channels.png", dpi=175, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run)
