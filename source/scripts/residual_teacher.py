# 中文阅读说明｜用途：定义并训练小型残差 CNN，按验证损失选择六个教师的检查点。
# 数据流：训练阶段使用 CUDA；数据/配置由冻结协议提供，干预阶段载入后不更新权重。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Small residual CNN and locked train/validation-only development stage."""
from __future__ import annotations
import argparse
import math
import platform
import random
import shutil
import time
from pathlib import Path
from external_experiment import (BASE, DATA, SEEDS, np, torch, F, normalized,
    configure_cpu, teacher_validation, now, sha, dump, read_json, event)
from torch import nn

RUN = BASE / "runs/fashion_residual_20260913"


# 用途：两次卷积与 BatchNorm 的残差块；源干预位置位于第一次 ReLU 之后。
# 结构：继承 nn.Module；成员字段和方法共同保存本阶段需要的状态。
class ResidualBlock(nn.Module):
    # 用途：定义块内两个卷积及旁路。
    # 输入：cin/cout 是输入/输出通道数，stride 是第一次卷积的步长。
    # 输出：通道或尺寸改变时，旁路用 1×1 卷积对齐；否则使用恒等映射。
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = (nn.Identity() if cin == cout and stride == 1 else
                         nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    # 用途：执行块内第一次卷积、BatchNorm 和 ReLU。
    # 输入：x[批次,输入通道,高,宽]；卷积步长由块的配置决定。
    # 输出：返回该块内部的源激活；最后一个块中形状为 [批次,32,14,14]。
    def mid(self, x):
        return F.relu(self.bn1(self.conv1(x)))

    # 用途：执行第二次卷积与 BatchNorm，加上原残差旁路后经过 ReLU。
    # 输入：h 为第一次 ReLU 后的特征图；residual 为与输出形状相同的旁路张量。
    # 输出：返回残差块输出特征图；干预实验中 residual 保持原值。
    def finish(self, h, residual):
        return F.relu(self.bn2(self.conv2(h)) + residual)

    # 用途：执行完整残差块，返回空间特征图；分类读出由外部 SmallResidualCNN 完成。
    # 参数：x；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：self.finish(self.mid(x), self.shortcut(x))。
    def forward(self, x):
        return self.finish(self.mid(x), self.shortcut(x))


# 用途：Fashion-MNIST 残差教师；prefix/tail 把干预点前后的计算明确拆开。
# 结构：继承 nn.Module；成员字段和方法共同保存本阶段需要的状态。
class SmallResidualCNN(nn.Module):
    # 用途：建立对象所需的网络层、固定映射或配置。Fashion-MNIST 残差教师；prefix/tail 把干预点前后的计算明确拆开。
    # 参数：无额外参数；读取或初始化对象状态。
    # 返回：初始化对象字段，没有显式返回值。
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.block1 = ResidualBlock(16, 16)
        # 这一残差块将空间尺寸减半；最后块的源特征图为 32×14×14。
        self.block2 = ResidualBlock(16, 32, 2)
        self.block3 = ResidualBlock(32, 32)
        self.readout = nn.Linear(32, 10)

    # 用途：执行图像到最后残差块内部干预点的前缀。
    # 输入：x[批次,1,28,28] 为已归一化图像。
    # 输出：返回 source 和 residual，二者均为 [批次,32,14,14]；后者是原旁路。
    def prefix(self, x):
        residual = self.block2(self.block1(self.stem(x)))
        # 同时返回旁路原值，使之后的源干预不会悄悄改变残差分支。
        return self.block3.mid(residual), residual

    # 用途：从指定源激活继续执行残差块、空间均值和线性分类读出。
    # 输入：source[批次,32,14,14]；residual 同形或沿批次广播。
    # 输出：返回 [批次,10] logits；实验比较干预前后之差。
    def tail(self, source, residual):
        return self.readout(self.block3.finish(source, residual).mean((2, 3)))

    # 用途：执行该网络的前向计算并返回分类 logits；训练时反向传播由外部训练循环控制。
    # 参数：x；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：self.tail(*self.prefix(x))。
    def forward(self, x):
        return self.tail(*self.prefix(x))


# 用途：载入已选中的残差教师，可转 float64 和指定设备；这里不训练。
# 参数：directory、double、device；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(model.double() if double else model).to(device)。
def load_teacher(directory, double=True, device="cpu"):
    model = SmallResidualCNN().eval()
    model.load_state_dict(torch.load(directory / "teacher.pt", weights_only=True, map_location="cpu"))
    return (model.double() if double else model).to(device)


# 用途：建立新的实验目录并保存数据、模型配置及源码的可核对协议。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def initialize(run):
    run.mkdir(parents=True, exist_ok=False)
    prior = read_json(BASE / "runs/fashion_structure_20260911/protocol.json")
    source_names = ["scripts/residual_teacher.py", "scripts/external_experiment.py", "scripts/research_runtime.py",
                    "upstream/CAT_MT_phase7_v0.7/code/context_module.py", "upstream/CAT_MT_phase7_v0.7/vendor/local_modules.py"]
    protocol = dict(utc=now(),version="residual-context-followup-v1",
        purpose="Test context selection inside a task-competent residual block, beyond an inactive dense head",
        scope="development-informed same-dataset follow-up; not standard ResNet-18 and not a new external dataset",
        prior_knowledge="Original CNN and GAP results were inspected; prior intervention batches 0..511 and full official classification pool were accessed; calibration outputs never accessed",
        seeds=SEEDS, architecture="stem conv1->16 BN ReLU; residual blocks 16->16/s1,16->32/s2,32->32/s1; GAP; linear32->10",
        teacher=dict(batch=128,optimizer="AdamW",lr=.002,min_lr=.00002,weight_decay=.0001,
            schedule="cosine at start of each epoch, 20 total epochs",max_epochs=20,
            selection="lowest validation cross entropy, stable earliest tie",augmentation="none",
            precision="CUDA float32, deterministic algorithms, TF32 disabled",discard_seeds=False),
        quality_rule="Report all six seeds. Call the architecture task-competent only if mean selected-checkpoint validation accuracy >=90% and minimum >=88%; no test-based tuning",
        data=dict(training=50000,validation=5000,calibration=5000,method_train=512,method_validation=128,
            intervention_test_ids="official 512..767, 256 inputs",calibration_model_access=False,
            classification="all 10000 official test images after final freeze"),
        intervention=dict(position="last residual branch after first BN+ReLU and before second convolution; identity path held at its original value",
            source_q=8,coordinates="training mean activation top8 among channels in central3x3",strengths=[1.,.5],
            training_coalitions=37,scored_high_order_coalitions=219,all_coalitions=256),
        context=dict(state="affected pre-ReLU values after conv+frozen-eval BN+identity addition",
            exact="recompute affected last-ReLU outputs and linear readout; keep original teacher weights",
            budgets="ceil(J/8),ceil(J/4),ceil(J/2), plus zero and complete controls",
            selector="training low-order response residual Gram greedy; omitted units use training mean-gate response",
            controls=["weight norm","gate variance times weight norm","activation frequency","four fixed active-random orders","fused preactivation PCA","mean gate","original region","exact"],
            pca="fuse encoder into affine conv+identity map; charge acquisition, reconstruction and all response computation"),
        timing=dict(dtype="CPU float64",threads=1,validation_inputs=4,queries=[37,256],
            workloads=["from_image","cached"],warmup=3,rounds=9,inner=3,random_order=True,
            exact_engines=["indexed conv rows","affected spatial crop convolution","full last convolution"],
            comparison="best measured exact engine per teacher/query/workload; report all engines, not a literal full-tail speedup claim",
            training_concurrent=False),
        reference=dict(device="CUDA float64 original unfused tail for tractable full-coalition validation/test; CPU float64 cross-check on four validation inputs",
            commitment="save predictions and SHA before independently running original tail for reference",
            tolerance=1e-9),
        statistics="Descriptive six-seed means/SD and paired differences, focus quarter budget greedy vs gate-weight at alpha1; no new confirmatory p-values",
        numerical_implementation="Additional intervention and timing sources are frozen before any new official-test model access; development uses training/validation only",
        sources={name:sha(BASE/name) for name in source_names},data_hashes=prior["data_hashes"],
        prior_freeze_sha256=sha(BASE / "runs/fashion_structure_20260911/freeze.json"),
        references=["https://arxiv.org/abs/1512.03385", "https://docs.pytorch.org/docs/2.12/generated/torch.nn.utils.fuse_conv_bn_eval.html"],
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__))
    for name in source_names:
        dest=run / "source" / name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,dest)
    np.save(run / "intervention_ids.npy",np.arange(512,768))
    protocol["intervention_ids_sha256"]=sha(run / "intervention_ids.npy")
    dump(run / "protocol.json",protocol)
    (run / "protocol.sha256").write_text(sha(run / "protocol.json")+"\n",encoding="ascii")
    event(run,"residual_protocol_locked",sha256=sha(run / "protocol.json"))


# 用途：核对协议、源码及依赖文件的原始哈希，确认执行条件未被更换。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：protocol。
def verify_protocol(run):
    protocol=read_json(run / "protocol.json")
    # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
    assert sha(run / "protocol.json")== (run / "protocol.sha256").read_text(encoding="ascii").strip()
    for name,value in protocol["sources"].items():
        assert sha(BASE/name)==value and sha(run / "source" / name)==value,name
    for name,value in protocol["data_hashes"].items():
        assert sha(BASE/name)==value,name
    assert sha(run / "intervention_ids.npy")==protocol["intervention_ids_sha256"]
    assert sha(BASE / "runs/fashion_structure_20260911/freeze.json")==protocol["prior_freeze_sha256"]
    return protocol


# 用途：按六个种子训练 20 轮；每轮验证，按最低验证交叉熵保存检查点，不按测试表现选模型。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def train(run):
    protocol=verify_protocol(run)
    assert not (run / "test_opened.json").exists()
    images=np.load(DATA / "official_train_images.npy",mmap_mode="r",allow_pickle=False)
    labels=np.load(DATA / "official_train_labels.npy",mmap_mode="r",allow_pickle=False)
    with np.load(DATA / "split_indices.npz",allow_pickle=False) as saved:
        train_ids=saved["train"];val_ids=saved["validation"]
    xt=torch.tensor(normalized(images[train_ids]),device="cuda",dtype=torch.float32)
    yt=torch.tensor(labels[train_ids],device="cuda",dtype=torch.long)
    xv=torch.tensor(normalized(images[val_ids]),device="cuda",dtype=torch.float32)
    yv=torch.tensor(labels[val_ids],device="cuda",dtype=torch.long)
    cfg=protocol["teacher"]
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    for seed in SEEDS:
        directory=run / f"seed_{seed}";directory.mkdir(exist_ok=True)
        if (directory / "teacher_summary.json").exists():
            assert sha(directory / "teacher.pt")==read_json(directory / "teacher_summary.json")["checkpoint_sha256"]
            continue
        assert not (directory / "teacher.pt").exists(),"Preserve incomplete training for inspection"
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        model=SmallResidualCNN().cuda()
        # 只在教师训练阶段更新权重；后续组合实验均使用冻结检查点。
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
        generator=torch.Generator().manual_seed(seed)
        history=[];best=float("inf");best_epoch=0
        # 等待 GPU 完成已提交任务，避免计时只统计异步调度开销。
        torch.cuda.synchronize();start=time.perf_counter()
        for epoch in range(1,cfg["max_epochs"]+1):
            lr=cfg["min_lr"]+.5*(cfg["lr"]-cfg["min_lr"])*(1+math.cos(math.pi*(epoch-1)/(cfg["max_epochs"]-1)))
            for group in optimizer.param_groups:group["lr"]=lr
            model.train();order=torch.randperm(len(xt),generator=generator).cuda()
            total=torch.zeros((),device="cuda")
            for offset in range(0,len(xt),cfg["batch"]):
                ix=order[offset:offset+cfg["batch"]]
                optimizer.zero_grad(set_to_none=True)
                # 分类交叉熵是教师学习/评价的任务损失，区别于后续干预响应的 NRMSE。
                # 这里才会计算训练梯度并更新模型；后续干预阶段固定模型权重。
                loss=F.cross_entropy(model(xt[ix]),yt[ix]);loss.backward();optimizer.step()
                total+=loss.detach()*len(ix)
            ce,acc=teacher_validation(model,xv,yv)
            row=dict(epoch=epoch,learning_rate=lr,train_cross_entropy=float(total.item()/len(xt)),
                validation_cross_entropy=ce,validation_accuracy=acc)
            history.append(row)
            # 检查点选择依据验证交叉熵，不依据测试集或干预结果。
            if ce<best:
                best,best_epoch=ce,epoch
                # 把当前选中的模型参数保存为检查点，后续干预从该固定权重载入。
                torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},directory / "teacher.pt")
            dump(directory / "training_history.json",history)
            event(run,"residual_epoch",seed=seed,**row,best_epoch=best_epoch)
        torch.cuda.synchronize()
        summary=dict(seed=seed,best_epoch=best_epoch,best_validation_cross_entropy=best,
            best_validation_accuracy=history[best_epoch-1]["validation_accuracy"],epochs=len(history),
            seconds=time.perf_counter()-start,checkpoint_sha256=sha(directory / "teacher.pt"),
            parameters=sum(p.numel() for p in model.parameters()),test_used_for_selection=False)
        dump(directory / "teacher_summary.json",summary)
        event(run,"residual_teacher_complete",**summary)
        del model,optimizer;torch.cuda.empty_cache()
    accuracy=[read_json(run / f"seed_{s}/teacher_summary.json")["best_validation_accuracy"] for s in SEEDS]
    dump(run / "training_complete.json",dict(utc=now(),seeds=SEEDS,mean_validation_accuracy=float(np.mean(accuracy)),
        minimum_validation_accuracy=min(accuracy),task_competent=bool(np.mean(accuracy)>=.90 and min(accuracy)>=.88),
        calibration_model_access=False,official_test_model_access=False))
    verify_protocol(run)
    event(run,"residual_training_complete",mean_accuracy=float(np.mean(accuracy)))


# 用途：定义并训练小型残差 CNN，按验证损失选择六个教师的检查点。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    # 解析运行阶段/路径等命令行参数；具体允许的阶段由 choices 或分派逻辑限定。
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("init","train"),required=True)
    args=parser.parse_args();configure_cpu()
    if args.stage=="init":initialize(RUN)
    else:train(RUN)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":main()
