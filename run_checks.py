"""无需原研究档案的数学控制与真实 forward 核验；不训练，不生成实验结果。"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'source/scripts'))


def main():
    """先核对已知答案，再在固定随机网络上检查完整计算图。"""
    import torch
    import numpy as np
    from test_joint_response import smoke
    from test_cross_layer_mechanism import known_answers as cross_controls
    from test_adaptive_mediator import known_answers as adaptive_controls
    from test_dsprites_mechanism import known_answers as dsprites_controls, check_model
    from dsprites_teacher import DSpritesCNN
    from dsprites_mechanism import select_source

    torch.set_num_threads(1)
    torch.manual_seed(20260929)
    model = DSpritesCNN().double().eval()
    # 确定性二值输入只检验算法；它不是 dSprites 数据，也不声称有分类语义。
    image = torch.as_tensor(np.random.default_rng(20260929).integers(0, 2, (1, 64, 64)),
                            dtype=torch.float64)
    nodes = select_source(model, image[None], 'cpu')
    result = dict(status='PASS', joint=smoke(), cross_layer=cross_controls(),
                  adaptive=adaptive_controls(), dsprites=dsprites_controls(),
                  random_network_forward=check_model(model, image, 0, nodes, 'cpu'),
                  context='Synthetic correctness control, not a trained model or research result')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
