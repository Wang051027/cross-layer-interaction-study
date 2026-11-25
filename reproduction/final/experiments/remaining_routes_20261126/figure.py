"""绘制全强度的支路干预、通道预算及通道对作用。"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from experiment import js, read_csv


def main(run):
    installed={f.name for f in font_manager.fontManager.ttflist}
    family=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","PingFang SC") if f in installed),None)
    if family:
        plt.rcParams["font.family"]=family
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.unicode_minus":False})
    def label(zh,en):
        return zh if family else en
    def sample(table):
        return [r for r in read_csv(run/(table+".csv")) if float(r["strength"])==1.]
    summary=js(run/"summary.json")["strengths"]["1.0"]
    figures=plt.figure(figsize=(11.6,7),layout="constrained")
    grid=figures.add_gridspec(2,2,height_ratios=[1,0.85],width_ratios=[1,1.15])
    ax=figures.add_subplot(grid[0,0])
    colors=["#187568","#496c98","#a26631"]
    components=sample("components")
    for i,term in enumerate(("conv_only","skip_only","branch_mixed")):
        rows=[r for r in components if r["term"]==term]
        v=np.array([float(r["ratio"]) for r in rows])
        ax.scatter(v,i+np.linspace(-.12,.12,len(v)),s=22,color=colors[i],alpha=.55)
        ax.scatter(v.mean(),i,marker="D",color=colors[i],s=52,edgecolor="white")
    ax.set_yticks(range(3),[label("卷积分支单独","Convolution only"),label("捷径单独","Shortcut only"),label("两支交互","Interaction")])
    ax.set_ylim(2.55,-.55); ax.set_xlim(0,max(1.06,1.04*max(float(r["ratio"]) for r in components if r["term"]=="conv_only")))
    ax.set_xlabel(label("该项 RMS / 余量 RMS","Term RMS / residual RMS"))
    ax.set_title(label("余量经哪条支路传递？","Which branch carries the residual?"),loc="left")
    ax=figures.add_subplot(grid[0,1])
    series=summary["channel_interventions"]
    x=[0,1,4,8]
    for n,(stage,col) in enumerate(zip(("first","last","both"),colors)):
        y=[1.]+[series[f"{stage}_{b}"]["remaining"] for b in x[1:]]
        ax.plot(x,y,marker="o",color=col,label=[label("第一处","First"),label("末次","Last"),label("两处同时","Both")][n])
    for name,style in (("random1","--"),("random2",":")):
        ax.plot(x,[1.]+[series[f"both_{b}"][name] for b in x[1:]],style,linewidth=1.2,color="#747b82",
                label=label("两组随机，两处同时","Both, randomized") if name=="random1" else None)
    ax.set(xticks=x,ylim=(0,1.08),xlabel=label("每处恢复的通道数","Channels restored per stage"),
           ylabel=label("干预后 RMS / 余量 RMS","Remaining / starting RMS"))
    ax.legend(frameon=False,fontsize=9,ncol=2,loc="lower left")
    ax.set_title(label("开发条件定名单，留出条件阻断","Restoring development-selected channels"),loc="left")
    ax=figures.add_subplot(grid[1,:])
    ranking=js(run/"rankings.json")["202609291"]
    sources=ranking["first"]["target"][:4]
    matrix=np.empty((4,32))
    for r in sample("link_scores"):
        if r["seed"]=="202609291":
            matrix[sources.index(int(r["first_channel"])),int(r["last_channel"])]=float(r["effect_rms"])
    im=ax.imshow(matrix,cmap="Blues",aspect="auto",vmin=0,interpolation="nearest")
    for dst in ranking["last"]["target"][:4]:
        ax.add_patch(Rectangle((dst-.48,-.48),.96,3.96,fill=False,edgecolor="#b36e34",linewidth=1.1))
    ax.set_xticks(range(32));ax.tick_params(axis="x",labelsize=8)
    ax.set_yticks(range(4),[str(v) for v in sources])
    ax.set(xlabel=label("末层通道编号","Final channel"),ylabel=label("第一处通道","First-stage channel"))
    ax.set_title(label("模型 291 的单条通道连接干预","Isolated channel interventions in model 291"),loc="left")
    figures.colorbar(im,ax=ax,shrink=.84,label=label("作用 RMS（logit）","Effect RMS (logit)"))
    figures.savefig(run/"remaining_routes.png",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(figures)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    main(parser.parse_args().run)
