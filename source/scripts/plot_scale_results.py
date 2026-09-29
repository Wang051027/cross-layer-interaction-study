# 中文阅读说明｜用途：绘制不同源坐标规模、位置和精确回退策略的已有结果。
# 数据流：输入规模/位置阶段统计文件；输出误差与成本图。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Scientific figure from completed, frozen evaluation and timing records."""
from pathlib import Path
import os
import json
import numpy as np

BASE=Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"]=str(BASE/"runs/residual_figure_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 用途：绘制不同源坐标规模、位置和精确回退策略的已有结果。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    data=json.loads((BASE/"audit/scale_interpretation_20260914.json").read_text(encoding="utf-8"))
    sizes=[4,8,12]
    fig,axes=plt.subplots(2,2,figsize=(12,8.8))
    methods=[("greedy",.25,"Selection, 1/4","#147a68"),
             ("gate_weight",.25,"Gate-weight, 1/4","#ab7b35"),
             ("pca_fused",.25,"Fused PCA, 1/4","#488eb5"),
             ("affine_fused",1.,"Fused affine","#aa5268")]
    for ax,position,title in zip(axes[0],["center","upper_left"],["A. Center window","B. Upper-left window"]):
        for method,fraction,label,color in methods:
            values=[next(r for r in data["summaries"] if r["position"]==position and int(r["q"])==q
                         and r["method"]==method and float(r["fraction"])==fraction and r["alpha"]=="1.0") for q in sizes]
            ax.errorbar(sizes,[100*float(r["nrmse_mean"]) for r in values],
                        yerr=[100*float(r["nrmse_sd"]) for r in values],fmt="o-",capsize=3,color=color,label=label)
        ax.axhline(10,color="#888888",ls=":",lw=1)
        ax.set_title(title,loc="left",fontweight="bold")
        ax.set_xticks(sizes);ax.set_ylim(0,20)
        ax.set_xlabel("Number of source coordinates")
        ax.set_ylabel("Intervention NRMSE (%)")
        if position=="center":ax.legend(frameon=False,fontsize=8,loc="upper left",ncol=2)
    ax=axes[1,0]
    for position,color,name in [("center","#147a68","Center"),("upper_left","#be783e","Upper left")]:
        values=[next(r for r in data["bound_summary"] if r["position"]==position and r["q"]==q and r["alpha"]==1.) for q in sizes]
        ax.errorbar(sizes,[100*r["acceptance_usable_mean"] for r in values],
                    yerr=[100*r["acceptance_usable_sd"] for r in values],fmt="o-",capsize=3,color=color,label=f"{name}: bound accepts")
        ax.plot(sizes,[100*r["hindsight_affine_usable_rate_mean"] for r in values],"s--",color=color,alpha=.6,
                label=f"{name}: actually within 10%")
    ax.set_title("C. Conservative acceptance of affine queries",loc="left",fontweight="bold")
    ax.set_xticks(sizes);ax.set_ylim(-4,116);ax.set_yticks(np.arange(0,101,20))
    ax.set_xlabel("Number of source coordinates")
    ax.set_ylabel("Fraction of inputs with nonzero effects (%)")
    ax.legend(frameon=False,fontsize=7.5,loc="upper center",ncol=2)
    ax=axes[1,1]
    for position,color,name in [("center","#147a68","Center"),("upper_left","#be783e","Upper left")]:
        for method,style,label in [("affine_fused_f1.0","--","affine"),("bounded_affine_f1.0","-","bound + fallback")]:
            values=[next(r for r in data["timing"] if r["position"]==position and r["q"]==q and r["method"]==method
                         and r["query_set"]=="evaluation" and r["workload"]=="from_image") for q in sizes]
            ax.errorbar(sizes,[r["speed_mean"] for r in values],yerr=[r["speed_sd"] for r in values],
                        marker="o",ls=style,capsize=3,color=color,label=f"{name}: {label}")
    ax.axhline(1,color="#888888",ls=":",lw=1)
    ax.set_title("D. Complete cost, starting from each image",loc="left",fontweight="bold")
    ax.set_xticks(sizes);ax.set_ylim(.5,1.92)
    ax.set_xlabel("Number of source coordinates")
    ax.set_ylabel("Fastest exact time / method time")
    ax.legend(frameon=False,fontsize=8,loc="upper left")
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.16);ax.set_axisbelow(True)
    fig.suptitle("Intervention size, spatial position, and exact fallback",fontsize=16,y=.98)
    fig.text(.5,.02,
        "Six frozen teachers; 128 official inputs (IDs 768-895); alpha=1; mean +/- seed SD. q=12 uses 256 sampled high-order masks.\n"
        "Affine uses current gates; selection/PCA use 1/4 dimension budget. Timing: CPU float64, includes prefix, bound check and fallback.",
        ha="center",fontsize=8.3,color="#526273")
    fig.tight_layout(rect=(0,.07,1,.95),h_pad=2.8,w_pad=2.4)
    fig.savefig(BASE/"reports/scale_position_results.png",dpi=165)
    fig.savefig(BASE/"reports/scale_position_results.pdf")
    plt.close(fig)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":main()
