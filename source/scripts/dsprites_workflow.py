"""接续 dSprites 的训练、选点、冻结、评价与核验。train-safe 将训练完成时间写入独立日志字段。"""
from pathlib import Path
import argparse,subprocess,sys,time
import dsprites_teacher as teacher


def train_safe():
    """保留冻结版训练算法，仅将额外日志中的 utc 重命名为 artifact_utc。"""
    original=teacher.event
    def log(kind,**details):
        """避免模型冻结记录的时间与事件写入时间撞名。"""
        if 'utc' in details:details['artifact_utc']=details.pop('utc')
        original(kind,**details)
    teacher.event=log
    teacher.train(teacher.configure())


def execute(script,*arguments):
    """采用本解释器执行单个阶段，既保留日志又立即传播失败状态。"""
    command=[sys.executable,'-B','-X','utf8',str(Path(__file__).with_name(script)),*arguments]
    print('RUN '+' '.join(command),flush=True)
    subprocess.run(command,check=True)


def continue_evaluation():
    """等待开发阶段的六个完成标志，最长 45 分钟，然后接续评价。"""
    deadline=time.monotonic()+2700
    while not all((teacher.RUN/'development'/f'seed_{s}'/'completion.json').exists() for s in teacher.SEEDS):
        if time.monotonic()>deadline:raise TimeoutError('Development did not complete; preserve partial artifacts and inspect logs')
        time.sleep(10)
    # 每个冻结/评价阶段本身会核对上一步完成清单及源码哈希。
    for stage in ('freeze','select','seal','evaluate','summarize','verify'):
        execute('run_dsprites_mechanism.py','--stage',stage)
    execute('audit_dsprites_results.py')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['continue','train-safe'],required=True)
    args=parser.parse_args()
    if args.stage=='continue':continue_evaluation()
    else:train_safe()
