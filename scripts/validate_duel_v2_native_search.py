"""CPU-native full32 and chance parity against the original search, no training."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(args):
    import numpy as np
    from app.modules.card_game.engine.duel_v2 import new_game, acting_side
    from app.modules.card_game.engine.duel_v2.simulation import simulate_action
    from app.modules.card_game.rl import seven_runtime, seven_lineup, cross_teacher_distillation
    from app.modules.card_game.rl.recovery_search import search
    from app.modules.card_game.rl.information_search import sample_world
    from app.modules.card_game.rl.batched_duel.native_array_program import capture_native_array_program
    from app.modules.card_game.rl.batched_duel.native_array_cpu import NativeArrayCPU
    from app.modules.card_game.rl.batched_duel.native_array_reference import NativeArrayReference
    from app.modules.card_game.rl.batched_duel.vm_image import lower_program
    from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
    from app.modules.card_game.rl.batched_search.array_zhenhong_prior import ArrayZhenhongPrior
    from scripts.validate_duel_v2_native_arrays import state_hash
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(scope='same-source CPU-native rule/search parity; not CUDA throughput',
        passed=False, rows=[], interpreter_fallback=False, gpu_qualified=False, learning_updates=0)
    def save():
        report['seconds'] = time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    try:
        if args.teachers:
            cross_teacher_distillation.REPO_TEACHERS = args.teachers
        models = {key: seven_runtime.SevenModel(args.models, key) for key in seven_lineup.KEYS}
        report['model_versions'] = {key: model.version for key, model in models.items()}
        program = capture_native_array_program()
        (args.output/'frozen_program.json').write_text(program.to_json(), encoding='utf8')
        image = lower_program(program)
        with NativeArrayCPU(image, limits=VmLimits(steps=48000000), directory=args.output/'native-code',
                            max_nodes=1048576, max_edges=2097152) as executor:
            runtime = NativeArrayReference(executor, max_batch_rows=8)
            guide = ArrayZhenhongPrior(device='cpu', allow_cpu_reference=True)
            for i in range(args.roots):
                if time.monotonic()-start > args.seconds:
                    raise TimeoutError('Native search parity budget exceeded')
                first = 'ab'[i%2]
                a = seven_lineup.KEYS[(i-i%2)%7]
                b = seven_lineup.KEYS[(i+1-i%2)%7]
                params = dict(seed=args.seed+i, first_side=first, skip_mulligan=True,
                              decks={'a': models[a].serving_deck, 'b': models[b].serving_deck})
                state = runtime.initialize([params])[0]
                expected = new_game(**params)
                if state != expected or state_hash(state) != state_hash(expected):
                    raise AssertionError('Native search initialization mismatch')
                viewer = acting_side(expected)
                seed = (1 << 256) | int.from_bytes(hashlib.sha256(
                    f'native-reference:{args.seed}:{i}:{viewer}'.encode()).digest(), 'big')
                world = runtime.determinize(state, viewer, seed)
                original_world = sample_world(expected, viewer, seed, runtime=seven_runtime)
                if world != original_world or state_hash(world) != state_hash(original_world):
                    raise AssertionError(['Native chance full-state mismatch', i])
                legal, _ = runtime.decision(world, viewer)
                if legal:
                    simulated = runtime.simulate(world, viewer, legal[0])
                    original_simulated = simulate_action(original_world, viewer, legal[0])
                    if simulated != original_simulated or state_hash(simulated) != state_hash(original_simulated):
                        raise AssertionError(['Native hypothetical state mismatch', i])
                policies = {'a': models[a], 'b': models[b]}
                options = dict(seed=seed, simulations=32, gumbel_candidates=16, max_depth=10,
                               noise=True, q_scale='natural_wdl', root_prior=guide, root_trace=True)
                t = time.monotonic()
                native = runtime.search(state, viewer, policies, **options,
                                        deadline=time.time()+args.seconds-(time.monotonic()-start))
                native_seconds = time.monotonic()-t
                t = time.monotonic()
                original = search(expected, viewer, policies, runtime=seven_runtime,
                    terminal_horizon=0, teacher_mode='network', **options,
                    deadline=time.time()+args.seconds-(time.monotonic()-start))
                original_seconds = time.monotonic()-t
                exact = {name: bool(np.array_equal(native[name], original[name]))
                         for name in ('x', 'c', 'visits', 'root_prior', 'root_gumbel', 'prior_shifts')}
                pi_close = bool(np.allclose(native['pi'], original['pi'], rtol=1e-5, atol=1e-6))
                row = dict(root=i, first=first, decks={'a': a, 'b': b},
                    chance_full_state=True, hypothetical_full_state=True,
                    requested=native['requested'], completed=native['simulations'],
                    exact=exact, pi_close=pi_close,
                    same_choice=native['search_choice'] == original['search_choice'],
                    same_value=native['value'] == original['value'],
                    native_seconds=native_seconds, original_seconds=original_seconds,
                    visits=native['visits'].tolist(), pi=native['pi'].tolist(),
                    prior_shifts=native['prior_shifts'].tolist(),
                    native_trace=native['root_trace'], original_trace=original['root_trace'])
                report['rows'].append(row)
                save()
                if (not native['complete'] or not original['complete'] or not all(exact.values())
                        or not pi_close or not row['same_choice'] or not row['same_value']):
                    raise AssertionError(['Native full32 search mismatch', i])
            report['native_dispatches'] = runtime.dispatches
            report['native_rows'] = runtime.rows
            report['passed'] = len(report['rows']) == args.roots
            save()
    except BaseException:
        report['passed'] = False
        report['error'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--teachers', type=Path)
    p.add_argument('--roots', type=int, default=8)
    p.add_argument('--seed', type=int, default=202610162901)
    p.add_argument('--seconds', type=int, default=1800)
    args = p.parse_args()
    args.output = args.output.resolve()
    if not 1 <= args.roots <= 64 or not 1 <= args.seconds <= 7200:
        p.error('Bounded roots and seconds required')
    main(args)
