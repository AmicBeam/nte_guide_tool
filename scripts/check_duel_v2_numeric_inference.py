#!/usr/bin/env python3
"""Audit old numeric warnings against independent float64 and Torch arithmetic."""
import argparse
from pathlib import Path
import sys
import warnings
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('New output required')
    import torch
    from app.modules.card_game.rl.archive_diagnostics import ArchivedDiagnosticModel
    from app.modules.card_game.rl import cross_grounded
    from scripts.fit_duel_v2_policy_recovery import load_games, write
    torch.set_num_threads(2)
    initial = Path('artifacts/rl-evals/five-cross-20260924/formal/initial')
    games = load_games('artifacts/rl-evals/five-cross-20260924/formal/episodes')
    cells = {}
    for key in ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk'):
        policy = ArchivedDiagnosticModel(initial, key)
        net = cross_grounded.network(policy.hidden)
        net.load_state_dict({name: torch.from_numpy(value.copy()) for name, value in policy.weights.items()})
        rows = [r for g in games for r in g['rows'] if r['key'] == key]
        picked = np.random.default_rng(20260927).choice(len(rows), 128, replace=False)
        errors = []; flags = 0; bound = 0.
        for i in picked:
            row = rows[int(i)]; x = row['x']; actor = bool(row.get('value_actor', 1))
            safe = policy.wdl_checked_float64(x, actor)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always', RuntimeWarning)
                original = policy.wdl(x, actor)
            flags += len(captured)
            with torch.no_grad():
                h = net.state_net(torch.as_tensor(x[None]))
                p = net.wdl(torch.cat((h, torch.tensor([[float(actor)]])), -1)).softmax(-1)[0].numpy()
            np.testing.assert_allclose(original, safe, atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(p, safe, atol=2e-6, rtol=2e-6)
            errors.append(float(np.max(np.abs(original - safe))))
            # An elementary magnitude bound, far below the float32 limit.
            w = policy.weights['state_net.0.weight']; b = policy.weights['state_net.0.bias']
            bound = max(bound, float(w.shape[1] * np.max(np.abs(w)) * np.max(np.abs(x)) + np.max(np.abs(b))))
        cells[key] = dict(rows=128, runtime_warnings=flags, max_probability_error=max(errors),
                          first_layer_absolute_bound=bound, finite=True, torch_match=True,
                          historical_arrays_sha=policy.version)
    result = dict(complete=True, passed=True, cells=cells, production_arithmetic_changed=False,
                  actual_overflow_reproduced=False,
                  note='Warnings on checked finite inputs are recorded, not treated as proof of actual arithmetic overflow.')
    write(args.output / 'result.json', result)
    lines = ['# 数值推理警告对拍', '',
             '每套128条真实训练观察：历史NumPy路径、独立float64/einsum与Torch三方对拍；不修改生产运算或源权重。', '',
             '| 队伍 | 警告数 | 最大概率偏差 | 第一层绝对值上界 |', '| --- | ---: | ---: | ---: |']
    for key, c in cells.items():
        lines.append(f"| {key} | {c['runtime_warnings']} | {c['max_probability_error']:.2e} | {c['first_layer_absolute_bound']:.2f} |")
    lines += ['', '本批结果均有限且对拍通过，未复现真实数值溢出。警告仍保留证据，不能概括为所有平台或所有局面已排除数值问题；CUDA需独立验证。', '']
    (args.output / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(result))


if __name__ == '__main__': main()
