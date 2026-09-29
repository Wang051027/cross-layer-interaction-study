# 中文阅读说明｜用途：绘制残差块背景选择、基线和成本比较结果。
# 数据流：读取残差阶段保存数据；输出静态图，不重新计算网络参考输出。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Presentation-only scientific figure for the completed residual experiment."""
from pathlib import Path
import os
import csv
BASE=Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"]=str(BASE / "runs/residual_figure_cache")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 用途：读取本函数指定的记录文件，返回解析后的对象；JSON/CSV 形式见函数体。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：list(csv.DictReader(f))。
def read(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


# 用途：绘制残差块背景选择、基线和成本比较结果。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    run=BASE / "runs/fashion_residual_20260913"
    data=read(run / "analysis/seed_nrmse.csv")
    timing=read(run / "timing/summary.csv");exact=read(run / "timing/fastest_exact.csv")
    fig,axes=plt.subplots(2,2,figsize=(12.1,8.7))
    methods=[("greedy","Residual selection","#147a68"),("gate_weight","Gate-weight score","#705ab0"),
             ("weight","Weight norm","#ce803d"),("pca_fused","Fused PCA","#488eb5"),
             ("active_random_mean","Active random (4-run mean)","#8f7186")]
    fractions=[.125,.25,.5];ax=axes[0,0]
    for offset,(method,label,color) in zip(np.linspace(-.3,.3,5),methods):
        for pos,f in enumerate(fractions):
            v=np.array([100*float(r["nrmse"]) for r in data if r["method"]==method and float(r["fraction"])==f and r["alpha"]=="1.0"])
            assert len(v)==6
            ax.errorbar(pos+offset,v.mean(),yerr=v.std(ddof=1),fmt="o",color=color,capsize=3,ms=4,label=label if pos==0 else None)
    ax.axhline(10,color="#a44848",ls="--",lw=1)
    ax.set_xticks([0,1,2],["1/8","1/4","1/2"])
    ax.set_xlabel("Fraction of affected context states retained")
    ax.set_ylabel("Intervention NRMSE (%)")
    ax.set_title("A. Accuracy at fixed context budgets",loc="left",fontweight="bold")
    ax.legend(fontsize=8,frameon=False,loc="best")
    ax=axes[0,1]
    for q,workload,label,color,style in [(37,"from_image","Q=37, from image","#ce803d","-"),
                                      (256,"from_image","Q=256, from image","#147a68","-"),
                                      (37,"cached","Q=37, cached context","#ce803d","--"),
                                      (256,"cached","Q=256, cached context","#147a68","--")]:
        means=[];sd=[]
        for fraction in fractions:
            ratios=[]
            for seed in range(202609071,202609077):
                same=lambda r:r["seed"]==str(seed) and r["queries"]==str(q) and r["workload"]==workload
                t=next(float(r["seconds"]) for r in timing if same(r) and r["method"]==f"greedy_f{fraction}")
                e=next(float(r["seconds"]) for r in exact if same(r))
                ratios.append(e/t)
            means.append(np.mean(ratios));sd.append(np.std(ratios,ddof=1))
        ax.errorbar(range(3),means,yerr=sd,marker="o",color=color,ls=style,capsize=3,label=label)
    ax.axhline(1,color="#747b85",ls=":",lw=1)
    ax.set_xticks([0,1,2],["1/8","1/4","1/2"])
    ax.set_xlabel("Fraction of context states retained")
    ax.set_ylabel("Best measured exact time / selection time")
    ax.set_title("B. Pre-test selection timing",loc="left",fontweight="bold")
    ax.legend(fontsize=8,frameon=False)
    ax=axes[1,0]
    for seed in range(202609071,202609077):
        vals=[100*next(float(r["nrmse"]) for r in data if r["seed"]==str(seed) and r["method"]==method and r["fraction"]=="0.25" and r["alpha"]=="1.0") for method in ("greedy","gate_weight","pca_fused")]
        ax.plot([0,1,2],vals,"o-",lw=1,ms=4,alpha=.7,label=str(seed)[-3:])
    ax.axhline(10,color="#a44848",ls="--",lw=1)
    ax.set_xticks([0,1,2],["Residual selection","Gate-weight score","Fused PCA"])
    ax.set_ylabel("Intervention NRMSE (%)")
    ax.set_title("C. Individual teachers at the 1/4 budget",loc="left",fontweight="bold")
    ax.legend(title="Seed suffix",fontsize=7.5,title_fontsize=8,ncol=3,frameon=False)
    ax=axes[1,1]
    groups=[read(BASE / "runs/fashion_external_20260911/test/teacher_classification.csv"),
            read(BASE / "runs/fashion_structure_20260911/test/gap_classification.csv"),
            read(run / "test/classification.csv")]
    for pos,(group,color) in enumerate(zip(groups,["#647c99","#aa9890","#147a68"])):
        values=np.array([100*float(r["accuracy"]) for r in group])
        ax.bar(pos,values.mean(),width=.52,color=color,alpha=.75)
        ax.scatter(pos+np.linspace(-.12,.12,6),values,s=17,color=color,edgecolor="white",linewidth=.4)
        ax.errorbar(pos,values.mean(),yerr=values.std(ddof=1),color=color,capsize=4)
        label=f"{values.mean():.3f}%" if pos==2 else f"{values.mean():.2f}%"
        ax.text(pos,values.max()+2,label,ha="center",fontsize=10)
    ax.set_xticks([0,1,2],["Dense-head CNN","Shallow GAP CNN","Residual CNN"])
    ax.set_ylim(0,103);ax.set_ylabel("Official test classification accuracy (%)")
    ax.set_title("D. Task quality of the architecture control",loc="left",fontweight="bold")
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    fig.suptitle("Context selection inside a residual block",fontsize=17,y=.98)
    fig.text(.5,.017,"Six teachers; mean +/- SD. Interventions: official IDs 512-767, alpha=1, 219 high-order coalitions.\n"
        "Timing: four validation inputs, CPU float64, one thread; best of three exact engines. Post-test affine audit reported separately.",ha="center",fontsize=8.5,color="#526273")
    fig.tight_layout(rect=(0,.065,1,.945),h_pad=2.6,w_pad=2)
    fig.savefig(BASE / "reports/residual_context_results.png",dpi=165)
    fig.savefig(BASE / "reports/residual_context_results.pdf")
    plt.close(fig)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":main()
