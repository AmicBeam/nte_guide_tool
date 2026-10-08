#!/usr/bin/env python3
"""Prepare source/tests and a fail-closed CUDA handoff plan. Does not contact Windows."""
import argparse
from pathlib import Path
import json
import sys
from hashlib import sha256
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXTRA = (
    'app/modules/card_game/rl/recovery_sampler.py','tests/test_duel_v2_recovery_sampler.py',
    'app/modules/card_game/rl/covered_value.py','tests/test_duel_v2_covered_value.py',
    'scripts/prepare_duel_v2_covered_value.py','scripts/verify_duel_v2_covered_value.py',
    'scripts/check_duel_v2_rollout_coverage.py',
    'app/modules/card_game/rl/covered_rollout.py', 'tests/test_duel_v2_covered_rollout.py',
    'app/modules/card_game/rl/teacher_validation.py', 'tests/test_duel_v2_teacher_validation.py',
    'scripts/check_duel_v2_search_teacher.py', 'scripts/verify_duel_v2_teacher_run.py',
    'scripts/check_duel_v2_recovery_full_fit.py',
    'scripts/check_duel_v2_recovery_capacity.py',
    'app/modules/card_game/rl/recovery_capacity.py',
    'app/modules/card_game/rl/recovery_policy.py',
    'app/modules/card_game/rl/recovery_round.py',
    'tests/test_duel_v2_recovery_round.py',
    'app/modules/card_game/rl/archive_diagnostics.py', 'app/modules/card_game/rl/full_hand_diagnostic.py',
    'app/modules/card_game/rl/policy_equivalence.py', 'app/modules/card_game/rl/recovery_search.py',
    'app/modules/card_game/rl/search_stability.py', 'app/modules/card_game/rl/offline_sources.py',
    'app/modules/card_game/rl/recovery_checkpoint.py',
    'scripts/audit_duel_v2_learning_data.py', 'scripts/check_duel_v2_full_hand.py',
    'scripts/check_duel_v2_q_scale.py', 'scripts/check_duel_v2_recovery_chain.py',
    'scripts/check_duel_v2_recovery_value.py', 'scripts/check_duel_v2_search_stability.py',
    'scripts/check_duel_v2_value_calibration.py', 'scripts/prepare_duel_v2_offline_readiness.py',
    'scripts/check_duel_v2_numeric_inference.py',
    'tests/test_duel_v2_learning_audits.py',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    plan = dict(kind='offline_source_and_cuda_preflight_only', windows_contacted=False,
                deployment=False, formal_training_allowed=False, website_approval=False,
                tests=['tests.test_duel_v2_covered_value','tests.test_duel_v2_recovery_sampler',
                       'tests.test_duel_v2_covered_rollout', 'tests.test_duel_v2_teacher_validation',
                       'tests.test_duel_v2_learning_audits', 'tests.test_duel_v2_residual_learning',
                       'tests.test_duel_v2_current_round', 'tests.test_duel_v2_recovery_round',
                       'tests.test_duel_v2_outcome_learning', 'tests.test_duel_v2_gumbel_search',
                       'tests.test_duel_v2_cross_teacher', 'tests.test_duel_v2_information_search',
                       'tests.test_duel_v2_training_recovery'],
                next_required=['Verify source hashes on Windows in an isolated checkout.',
                               'Repeat Torch/NumPy and restore checks on CPU, then CUDA.',
                               'Measure real Gumbel32 full-game throughput with the new Q-scale identity.',
                               'Complete all five value/policy gates; three preserved teachers do not cover zhenhong/murk.',
                               'Use a qualified covered teacher receipt; fixed-build network rollouts are not domain-qualified.',
                               'Price terminal rollouts and selective-best value/search rechecks within the original deadline.',
                               'Reserve fresh seeds and a new dated 09:00 schedule, including independent final evaluation.',
                               'Do not launch formal training or replace service models on the strength of this bundle.'])
    if not args.prepare:
        print(json.dumps(dict(dry_run=True, plan=plan))); return
    if args.output.exists(): raise ValueError('New output required')
    from app.modules.card_game.rl.offline_sources import snapshot
    manifest = snapshot(ROOT, args.output / 'source', EXTRA)
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    commands = 'python -m unittest ' + ' '.join(plan['tests'])
    (args.output / 'preflight-command.txt').write_text(commands + '\n', encoding='utf-8')
    text = ['# Windows上线后的预检材料', '',
            '本包只冻结普通源码、测试和长期操作说明。没有数据库、环境文件、凭证、训练经验或可发布模型；不连接Windows、不安装依赖、不启动训练。', '',
            '在已有仓库的隔离工作副本中逐文件核验source-manifest.json，再使用匹配的Python环境执行：', '',
            '```text', commands, '```', '',
            'CPU测试通过后，还必须做CUDA数值/恢复/梯度短测、正式Gumbel32吞吐测量和五套行为/value闸门；本机三套小链路不代替这些条件。', '',
            '下一次实际任务的种子与带日期09:00截止在启动时重新分配，不沿用已消费的本机确认种子。', '']
    (args.output / 'README.md').write_text('\n'.join(text), encoding='utf-8')
    archive = args.output / 'source-preflight.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as handle:
        for path in sorted((args.output / 'source').rglob('*')):
            if path.is_file(): handle.write(path, path.relative_to(args.output))
        for name in ('plan.json', 'preflight-command.txt', 'README.md'):
            handle.write(args.output / name, name)
    result = dict(files=len(manifest), archive_sha256=sha256(archive.read_bytes()).hexdigest(),
                  archive_bytes=archive.stat().st_size, formal_training_allowed=False)
    (args.output / 'package.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__': main()
