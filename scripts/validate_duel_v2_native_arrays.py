"""Bounded native-code rule parity and natural games, with explicit CPU oracle.

No learning, no service approval, no host-controlled search sold as GPU
throughput. Windows is not needed for the CPU-native same-source validation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def state_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def differences(left, right, path='', result=None):
    result = [] if result is None else result
    if len(result) >= 12:
        return result
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append([path+'/'+str(key), 'missing on one side'])
            else:
                differences(left[key], right[key], path+'/'+str(key), result)
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            result.append([path, 'length', len(left), len(right)])
        else:
            for i, (a, b) in enumerate(zip(left, right)):
                differences(a, b, path+'/'+str(i), result)
    elif left != right:
        result.append([path, repr(left)[:160], repr(right)[:160]])
    return result


def main(args):
    import numpy as np
    from app.modules.card_game.content.duel_v2 import STARTER_DECKS
    from app.modules.card_game.engine.duel_v2 import new_game, apply_action, acting_side, observe
    from app.modules.card_game.rl import seven_lineup, seven_runtime, cross_teacher_distillation
    from app.modules.card_game.rl.league_rollout import clean
    from app.modules.card_game.rl.batched_duel.native_array_program import capture_native_array_program, load_native_array_program
    from app.modules.card_game.rl.batched_duel.native_array_cpu import NativeArrayCPU
    from app.modules.card_game.rl.batched_duel.native_array_reference import NativeArrayReference
    from app.modules.card_game.rl.batched_duel.vm_image import lower_program
    from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(scope='native array diagnostic queries with canonical post-batch oracle; not closed GPU throughput',
        backend=args.backend,
        passed=False, natural_games=0, requested_games=args.games, physical_decisions=0,
        interpreter_fallback=False, runtime_instruction_fetch=False, learning_updates=0,
        gpu_qualified=False, search_throughput_qualified=False)
    def save(phase):
        report['phase'] = phase
        report['seconds'] = time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    def budget():
        if time.monotonic()-start > args.seconds:
            raise TimeoutError('Native array validation budget exhausted; unfinished games have no result')
    try:
        save('capture')
        program = (load_native_array_program(args.program) if args.program else capture_native_array_program())
        (args.output/'frozen_program.json').write_text(program.to_json(), encoding='utf8')
        image = lower_program(program)
        report.update(functions=len(image.functions), instructions=len(image.instructions),
                      program_identity=program.identity, image_identity=image.identity)
        save('compile')
        config = dict(limits=VmLimits(steps=args.rule_steps), max_nodes=args.scratch_nodes,
                      max_edges=2*args.scratch_nodes, max_payload_bytes=args.scratch_payload)
        if args.backend == 'cuda-native':
            from app.modules.card_game.rl.batched_duel.native_array_cuda_query import NativeArrayCUDAQuery
            prepared_executor = NativeArrayCUDAQuery(image, **config)
        else:
            prepared_executor = NativeArrayCPU(image, directory=args.output/'native-code', **config)
        with prepared_executor as executor:
            actual = getattr(executor, 'executor', executor)
            report['native_source_sha256'] = getattr(actual, 'source_sha256',
                getattr(getattr(actual, 'module', None), 'source_sha', None))
            runtime = NativeArrayReference(executor, max_batch_rows=args.batch_rows)
            report['native_batch_rows'] = args.batch_rows
            decks = {d['id']: d for d in STARTER_DECKS}
            models = None
            if args.models:
                if args.teachers:
                    cross_teacher_distillation.REPO_TEACHERS = args.teachers
                models = {key: seven_runtime.SevenModel(args.models, key) for key in seven_lineup.KEYS}
                decks.update({key: model.serving_deck for key, model in models.items()})
            jobs = []
            for i in range(args.games):
                a = seven_lineup.KEYS[(i//2) % 7]
                b = a if args.mirrors else seven_lineup.KEYS[((i//2)+1) % 7]
                parameters = dict(seed=args.seed+i//2, decks={'a': decks[a], 'b': decks[b]},
                                  first_side='ab'[i%2], skip_mulligan=False, escalation=True)
                jobs.append(dict(parameters=parameters, keys={'a': a, 'b': b},
                                 steps=0, random=random.Random(args.seed+i)))
            states = runtime.initialize([j['parameters'] for j in jobs]) if jobs else []
            expected = [new_game(**j['parameters']) for j in jobs]
            if states != expected or state_hash(states) != state_hash(expected):
                raise AssertionError(['native initialization', differences(states, expected)])
            report['initial_states_verified'] = len(states)
            trace = (args.output/'physical-actions.jsonl').open('w', encoding='utf8')
            with trace:
                while True:
                    budget()
                    live = [i for i, s in enumerate(states) if s['phase'] != 'finished']
                    if not live:
                        break
                    if any(jobs[i]['steps'] >= args.max_decisions for i in live):
                        raise RuntimeError('Unfinished native games reached max_decisions')
                    actors = runtime.call('acting_side', [[states[i]] for i in live])
                    queries = runtime.call('public_legal_decision', [[states[i], a] for i, a in zip(live, actors)])
                    actions = []
                    for i, side, (legal, view) in zip(live, actors, queries):
                        if side != acting_side(expected[i]):
                            raise AssertionError('Native acting player mismatch')
                        legal = [a for a in legal if a['type'] != 'concede']
                        public = observe(expected[i], side, include_previews=False)
                        reference_actions = [e['action'] for e in public['legal_actions'] if e['action']['type'] != 'concede']
                        x, c = seven_lineup.encode(view, legal)
                        xx, cc = seven_lineup.encode(public, reference_actions)
                        if legal != reference_actions or not np.array_equal(x, xx) or not np.array_equal(c, cc):
                            raise AssertionError(['native public/legal encoding', i, jobs[i]['steps']])
                        if not legal:
                            raise AssertionError('Ongoing native game has no legal decision')
                        if models:
                            index = int(np.argmax(models[jobs[i]['keys'][side]].scores(x, c)))
                        else:
                            priority = {'choose': 8, 'choose_cards': 8, 'choose_deck_order': 8,
                                        'mulligan': 7, 'play_card': 5, 'ultimate': 4, 'attack': 3}
                            scores = [priority.get(a['type'], 0)+jobs[i]['random'].random() for a in legal]
                            index = int(np.argmax(scores))
                        actions.append(legal[index])
                    advanced = runtime.call('apply_action', [[states[i], a, action]
                        for i, a, action in zip(live, actors, actions)])
                    for i, side, action, actual in zip(live, actors, actions, advanced):
                        wanted = apply_action(expected[i], side, action)
                        if actual != wanted or state_hash(actual) != state_hash(wanted):
                            raise AssertionError(['native physical full state', i, jobs[i]['steps'],
                                                  action, differences(actual, wanted)])
                        expected[i] = clean(wanted)
                        jobs[i]['steps'] += 1
                        report['physical_decisions'] += 1
                        trace.write(json.dumps(dict(game=i, step=jobs[i]['steps'], side=side,
                            action=action, state_sha256=state_hash(actual)), ensure_ascii=False)+'\n')
                    cleaned = runtime.call('clean', [[s] for s in advanced])
                    for i, actual in zip(live, cleaned):
                        if actual != expected[i] or state_hash(actual) != state_hash(expected[i]):
                            raise AssertionError(['native clean', differences(actual, expected[i])])
                        states[i] = actual
                    report['natural_games'] = sum(s['phase'] == 'finished' for s in states)
                    report['native_dispatches'] = dict(runtime.dispatches)
                    save('natural_games')
            report['games'] = [dict(game=i, first=j['parameters']['first_side'], decks=j['keys'],
                steps=j['steps'], winner=states[i]['winner'], reason=states[i]['reason'])
                for i, j in enumerate(jobs)]
            report['native_dispatches'] = dict(runtime.dispatches)
            report['native_rows'] = dict(runtime.rows)
            if args.mechanism_tests:
                from app.modules.card_game.rl.batched_duel.native_array_conformance import run_rule_conformance
                save('mechanism_tests')
                mechanics = run_rule_conformance(runtime, args.output/'mechanisms', budget=budget)
                report['mechanisms'] = mechanics
                if not mechanics['passed']:
                    raise AssertionError('Native mechanism conformance failed; see mechanisms/tests.log')
            report['passed'] = report['natural_games'] == args.games
            save('complete')
    except BaseException:
        report['passed'] = False
        report['error'] = traceback.format_exc()
        save('failed')
        raise
    finally:
        save(report['phase'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=('cpu-native', 'cuda-native'), default='cpu-native')
    p.add_argument('--program', type=Path)
    p.add_argument('--games', type=int, default=14)
    p.add_argument('--batch-rows', type=int, default=8)
    p.add_argument('--seed', type=int, default=202610162401)
    p.add_argument('--mirrors', action='store_true')
    p.add_argument('--mechanism-tests', action='store_true')
    p.add_argument('--models', type=Path)
    p.add_argument('--teachers', type=Path)
    p.add_argument('--max-decisions', type=int, default=400)
    p.add_argument('--rule-steps', type=int, default=48000000)
    p.add_argument('--scratch-nodes', type=int, default=1048576)
    p.add_argument('--scratch-payload', type=int, default=33554432)
    p.add_argument('--seconds', type=int, default=1800)
    args = p.parse_args()
    args.output = args.output.resolve()
    if sys.version_info[:2] != (3, 10) and args.program is None:
        p.error('--program is required outside CPython 3.10; no runtime recapture')
    if (not 0 <= args.games <= 256 or not 1 <= args.max_decisions <= 2000 or not 1 <= args.seconds <= 7200
            or not 4096 <= args.scratch_nodes <= 4194304
            or not 131072 <= args.scratch_payload <= 134217728
            or not 1 <= args.batch_rows <= 32 or not 1 <= args.rule_steps <= 2**30
            or args.batch_rows * (args.scratch_nodes*32+args.scratch_payload) > 4*1024**3):
        p.error('Bounded games, decisions and seconds required')
    main(args)
