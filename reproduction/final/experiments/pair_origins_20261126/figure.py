"""绘制上游阻断、恢复及单个 ReLU 位置的实测结果。"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np


def read(run, name):
    with (run/(name+".csv")).open(encoding="utf8",newline="") as f:
        return list(csv.DictReader(f))


def main(run):
    fonts={f.name for f in font_manager.fontManager.ttflist}
    family=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","PingFang SC") if f in fonts),None)
    if family:
        plt.rcParams["font.family"]=family
    def t(zh,en):
        return zh if family else en
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.unicode_minus":False})
    fig,(left,right)=plt.subplots(1,2,figsize=(12,4.8),layout="constrained",gridspec_kw={"width_ratios":[1.1,1]})
    rows=[r for r in read(run,"arm_scores") if r["strength"]=="1.0"]
    operations=("h_pair","rescue_k_pair","h_both_all","h_both_target")
    x=np.arange(4)
    for i,(pair,color) in enumerate((("23-30","#276781"),("17-27","#a65f35"))):
        y=[float(next(r["total_change_rms"] for r in rows if r["pair"]==pair and r["arm"]==op)) for op in operations]
        bars=left.bar(x+(i-.5)*.36,y,.34,label=pair.replace("-"," + "),color=color)
        left.bar_label(bars,fmt="%.2f",padding=2,fontsize=9)
    left.set_xticks(x,[t("上游双阻断","Upstream block"),t("恢复中继激活","Restore relay"),t("两处各 2 通道回放开关","Replay both channel pairs"),t("两处各 8 点回放","8 sites per stage")],
                    rotation=13,ha="right")
    left.set(ylabel=t("距原三阶输出的 RMS（logit）","RMS distance to original output (logit)"),ylim=(0,1.1))
    left.legend(frameon=False,ncol=2)
    left.set_title(t("上游阻断后，整网输出距原值多远？","Distance from original output after intervention"),loc="left")
    case=[r for r in read(run,"case_effects") if (r["pair"],r["strength"],r["case"],r["mask"])==("17-27","1.0","9","13")]
    local=next(r for r in read(run,"local_cases") if (r["pair"],r["strength"],r["case"],r["mask"],r["arm"])==("17-27","1.0","9","13","minimal"))
    data=[next(r for r in case if r["arm"]==name) for name in ("base","h_pair")]+[local]
    for ch,key,color,offset in ((17,"a","#447e70",-.18),(27,"b","#855780",.18)):
        values=[float(r[key]) for r in data]
        bars=right.bar(np.arange(3)+offset,values,.34,color=color,label=str(ch))
        right.bar_label(bars,fmt="%.4f",padding=2,fontsize=9)
    right.set_xticks(range(3),[t("原状态","Original"),t("阻断上游 27、30","Block 27, 30"),t("回放末层一个开关","Replay one final gate")],rotation=12,ha="right")
    right.axhline(0,color="#666",lw=.8)
    right.set(ylabel=t("单通道的三阶得分贡献（logit）","Channel contribution (logit)"),ylim=(-.012,.057))
    right.set_title(t("配对 9，源掩码 13，位置 (27, 14, 13)","Pair 9, source mask 13, site (27, 14, 13)"),loc="left")
    right.legend(frameon=False,ncol=2)
    fig.savefig(run/"pair_origins.png",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    main(parser.parse_args().run)
