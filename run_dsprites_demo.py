from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def project(values, label, target):
    """独立显式投影：真实类减其余类均值，或去除各类共同分量。"""
    if target == 'vector':
        return values - values.mean(axis=-1, keepdims=True)
    classes = values.shape[-1]
    weights = np.full(classes, -1.0 / (classes - 1))
    weights[int(label)] = 1.0
    return (values @ weights)[..., None]


def main():
    """重算 44 种方案的 MSE，并与原始结果中相同记录比较。"""
    with np.load(ROOT/'example/dsprites/responses.npz', allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    specs = json.loads((ROOT/'example/dsprites/specs.json').read_text(encoding='utf-8'))
    # 每行四个组合，符号固定为 +1、-1、-1、+1；直接矩阵乘法独立于原切片实现。
    matrix = np.zeros((24, 16))
    for row, indices in enumerate(data['stencil']):
        assert len(set(indices.tolist())) == 4
        matrix[row, indices] = [1, -1, -1, 1]
    errors = {}
    tables = []
    for target in ('margin', 'vector'):
        collected = {key: [] for key in ('natural_mse', 'restore_mse', 'transplant_error_mse')}
        for index, label in enumerate(data['labels']):
            natural = matrix @ project(data['natural'][index], label, target)
            patched = np.einsum('qr,morc->moqc', matrix,
                                project(data['patched'][index], label, target))
            natural_mse = float(np.square(natural).mean())
            restored_mse = np.square(patched[:, 0]).mean(axis=(-2, -1))
            transplant_mse = np.square(patched[:, 1] - natural).mean(axis=(-2, -1))
            collected['natural_mse'].append(natural_mse)
            collected['restore_mse'].append(restored_mse)
            collected['transplant_error_mse'].append(transplant_mse)
            for method in ('global', 'class', 'input'):
                name = f'{target}_{method}_finite'
                choice = next(j for j, spec in enumerate(specs)
                              if spec == {'method': name, 'budget': 16})
                tables.append(dict(target=target, label=int(label), method=name,
                    restore_ratio=float(np.sqrt(restored_mse[choice]/natural_mse)) if natural_mse > 1e-18 else None,
                    transplant_error_ratio=float(np.sqrt(transplant_mse[choice]/natural_mse)) if natural_mse > 1e-18 else None))
        for key, values in collected.items():
            errors[f'{target}_{key}'] = float(np.max(np.abs(np.asarray(values) - data[f'{target}_{key}'])))
    direct = np.einsum('qr,nrc->nqc', matrix, data['natural'])
    errors['sum_layer_contributions'] = float(np.max(np.abs(direct - data['contributions'].sum(axis=1))))
    assert max(errors.values()) < 1e-10, errors
    result = dict(status='PASS', context='Three cached dSprites responses from one model',
                  panel_indices=data['panel_indices'].tolist(), max_abs_errors=errors,
                  ratios_lower_is_better=tables)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
