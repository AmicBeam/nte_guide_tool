"""Bounded native SoA software conformance: typed games, mechanisms, Gumbel32.

Default prints a plan. --run writes a new artifacts/rl-evals directory.
CPU NativeSoACPU only; never CUDA throughput, training, or host rule fallback.
Formal oracle compares after actual native results and cannot compute them.
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
    elif type(left) is not type(right) or left != right:
        result.append([path, type(left).__name__, type(right).__name__,
                       repr(left)[:120], repr(right)[:120]])
    return result


def check_aliases(left, right):
    forward, reverse = {}, {}
    def walk(a, b):
        if type(a) is not type(b):
            raise AssertionError('Nested scalar/container/entity type changed')
        if isinstance(a, (dict, list, tuple)):
            if forward.setdefault(id(a), id(b)) != id(b) or reverse.setdefault(id(b), id(a)) != id(a):
                raise AssertionError('Entity alias structure changed')
        if isinstance(a, dict):
            if list(a)!=list(b):
                raise AssertionError('Dictionary field insertion order changed')
            for key in a:
                walk(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            for x, y in zip(a, b):
                walk(x, y)
    walk(left, right)


def compare_boundary(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected or state_hash(actual) != state_hash(expected):
        raise AssertionError([label, differences(actual, expected)])
    check_aliases(actual, expected)


class FixtureRouter:
    """Pack each independent fixture input once; never keep a host Game in the call."""

    def __init__(self, runtime):
        self.runtime = runtime

    def call(self, entry, rows):
        from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states, unpack_game_states
        from app.modules.card_game.rl.batched_duel.record_soa_compact import compact_rows
        output = []
        for state, side, action in rows:
            packed = pack_game_states([state], self.runtime.schema)
            packet = self.runtime.call(entry, packed, [[None, side, action]])
            if packet.decode(path=(0,))[0]:
                actual = unpack_game_states(compact_rows(packet.select_state(path=(1,))))[0]
                output.append((True, actual))
            else:
                output.append(packet.decode()[0])
        return output


def plan(args):
    from app.modules.card_game.content.duel_v2 import PUBLIC_CHARACTER_IDS
    from app.modules.card_game.rl import seven_lineup
    return dict(mode='plan',
        scope='native SoA CPU software conformance over persistent typed states; not CUDA throughput',
        games=args.games, requested_searches=(args.roots if args.models else 0),
        mechanism_tests=bool(args.mechanism_tests), presets=list(seven_lineup.KEYS),
        public_characters=len(PUBLIC_CHARACTER_IDS), gpu_qualified=False,
        interpreter_fallback=False, actual_backend=('cpu_native_soa' if args.backend=='cpu' else 'cuda_native_soa_query'),
        output=None if args.output is None else str(args.output))


def choose_index(legal, view, job, models, side):
    import numpy as np
    from app.modules.card_game.rl import seven_lineup
    if models:
        x, c = seven_lineup.encode(view, legal)
        return int(np.argmax(models[job['keys'][side]].scores(x, c)))
    priority = {'choose': 8, 'choose_cards': 8, 'choose_deck_order': 8,
                'mulligan': 7, 'play_card': 5, 'ultimate': 4, 'attack': 3}
    scores = [priority.get(a['type'], 0)+job['random'].random() for a in legal]
    return int(np.argmax(scores))


def main(args):
    import numpy as np
    from app.modules.card_game.content.duel_v2 import STARTER_DECKS
    from app.modules.card_game.engine.duel_v2 import new_game, apply_action, acting_side, observe
    from app.modules.card_game.rl import seven_lineup, seven_runtime, cross_teacher_distillation
    from app.modules.card_game.rl.league_rollout import clean
    from app.modules.card_game.rl.recovery_search import search as formal_search
    from app.modules.card_game.rl.batched_duel.native_array_program import (
        capture_native_array_program, load_native_array_program)
    from app.modules.card_game.rl.batched_duel.native_array_conformance import run_rule_conformance
    from app.modules.card_game.rl.batched_duel.native_soa_cpu import NativeSoACPU
    from app.modules.card_game.rl.batched_duel.native_soa_reference import NativeSoAReference
    from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
    from app.modules.card_game.rl.batched_duel.record_soa import unpack_game_states
    from app.modules.card_game.rl.batched_duel.vm_image import lower_program
    from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(scope='native SoA software stages with canonical post-run oracle; not closed GPU throughput',
        passed=False, natural_games=0, requested_games=args.games, physical_decisions=0,
        interpreter_fallback=False, runtime_instruction_fetch=False, learning_updates=0,
        gpu_qualified=False, search_throughput_qualified=False,
        software_capture=False, software_initialize=False, software_natural_games=False,
        software_mechanisms=False, software_search=False)
    report.update(actual_rule_backend=args.backend,controller='explicit_host_diagnostic')
    report['requested_stages']=dict(natural_games=args.games,mechanisms=bool(args.mechanism_tests),
                                     search_roots=args.roots if args.models else 0)
    def save(phase):
        report['phase'] = phase
        report['seconds'] = time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    def budget():
        if time.monotonic()-start > args.seconds:
            raise TimeoutError('Native SoA validation budget exhausted; unfinished games have no result')
    try:
        save('capture')
        program = (load_native_array_program(args.program) if args.program else capture_native_array_program())
        (args.output/'frozen_program.json').write_text(program.to_json(), encoding='utf8')
        image = lower_program(program)
        schema = compile_execution_schema()
        report.update(functions=len(image.functions), instructions=len(image.instructions),
                      program_identity=program.identity, image_identity=image.identity,
                      schema_identity=schema.identity, software_capture=True)
        save('compile')
        config = dict(limits=VmLimits(steps=args.rule_steps), max_nodes=args.scratch_nodes,
                      max_edges=2*args.scratch_nodes, max_payload_bytes=args.scratch_payload)
        backend=NativeSoACPU
        if args.backend=='cuda':
            from app.modules.card_game.rl.batched_duel.native_soa_cuda_query import NativeSoACUDAQuery
            backend=NativeSoACUDAQuery
        with backend(image, schema, directory=args.output/'native-code', **config) as executor:
            report['native_source_sha256'] = executor.source_sha256
            (args.output/'source-metadata.json').write_text(json.dumps(dict(
                program_identity=program.identity, image_identity=image.identity,
                schema_identity=schema.identity, source_sha256=executor.source_sha256,
                gpu_qualified=False, interpreter_fallback=False), indent=2), encoding='utf8')
            runtime = NativeSoAReference(executor, max_batch_rows=args.batch_rows, schema=schema)
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
                                 steps=0, random=random.Random(args.seed+i), actions=[]))
            states = runtime.initialize([j['parameters'] for j in jobs]) if jobs else []
            oracles = [new_game(**j['parameters']) for j in jobs]
            for handle, expected in zip(states, oracles):
                actual = unpack_game_states(handle.batch)[0]
                compare_boundary(actual, expected, 'native SoA initialize')
                side = runtime.actor(handle)
                if side != acting_side(expected):
                    raise AssertionError('Native acting player mismatch at ingress')
                legal, view = runtime.public(handle, side)
                legal = [a for a in legal if a['type'] != 'concede']
                public = observe(expected, side, include_previews=False)
                reference = [e['action'] for e in public['legal_actions'] if e['action']['type'] != 'concede']
                x, c = seven_lineup.encode(view, legal)
                xx, cc = seven_lineup.encode(public, reference)
                if legal != reference or not np.array_equal(x, xx) or not np.array_equal(c, cc):
                    raise AssertionError('Native ingress legal/encoding mismatch')
            report.update(initial_states_verified=len(states), software_initialize=bool(states))
            save('initialize')
            with (args.output/'physical-actions.jsonl').open('w', encoding='utf8') as trace:
                while True:
                    budget()
                    live = [i for i, s in enumerate(states) if s['phase'] != 'finished']
                    if not live:
                        break
                    if any(jobs[i]['steps'] >= args.max_decisions for i in live):
                        raise RuntimeError('Unfinished native SoA games reached max_decisions')
                    for i in live:
                        side = runtime.actor(states[i])
                        legal, view = runtime.public(states[i], side)
                        legal = [a for a in legal if a['type'] != 'concede']
                        if not legal:
                            raise AssertionError('Ongoing native SoA game has no legal decision')
                        action = legal[choose_index(legal, view, jobs[i], models, side)]
                        states[i] = runtime.clean(runtime.apply(states[i], side, action))
                        jobs[i]['steps'] += 1
                        jobs[i]['actions'].append((side, action))
                        report['physical_decisions'] += 1
                        trace.write(json.dumps(dict(game=i, step=jobs[i]['steps'], side=side,
                            action=action), ensure_ascii=False) + chr(10))
                    report['natural_games'] = sum(s['phase'] == 'finished' for s in states)
                    report['native_dispatches'] = dict(runtime.dispatches)
                    save('natural_games')
            terminals = []
            for i, job in enumerate(jobs):
                actual = unpack_game_states(states[i].batch)[0]
                expected = new_game(**job['parameters'])
                for side, action in job['actions']:
                    expected = clean(apply_action(expected, side, action))
                compare_boundary(actual, expected, 'native SoA terminal')
                if actual.get('rng') != expected.get('rng'):
                    raise AssertionError('Native SoA RNG mismatch at egress')
                terminals.append(dict(game=i, first=job['parameters']['first_side'], decks=job['keys'],
                    steps=job['steps'], winner=states[i].get('winner'), reason=actual.get('reason')))
            report['games'] = terminals
            report['native_dispatches'] = dict(runtime.dispatches)
            report['native_rows'] = dict(runtime.rows)
            report['software_natural_games'] = args.games>0 and report['natural_games'] == args.games
            if args.games and not report['software_natural_games']:
                raise AssertionError('Requested natural SoA games incomplete')
            if args.mechanism_tests:
                save('mechanism_tests')
                mechanics = run_rule_conformance(FixtureRouter(runtime), args.output/'mechanisms', budget=budget)
                report['mechanisms'] = {k: mechanics[k] for k in (
                    'passed', 'tests', 'skipped', 'failures', 'errors',
                    'native_commits', 'native_rejections', 'tests_with_native_commits') if k in mechanics}
                report['software_mechanisms'] = bool(mechanics['passed'])
                if not mechanics['passed']:
                    raise AssertionError('Native SoA mechanism conformance failed; see mechanisms/tests.log')
            if args.models:
                save('search')
                from app.modules.card_game.rl.batched_search.array_zhenhong_prior import ArrayZhenhongPrior
                guide=ArrayZhenhongPrior(device='cpu',allow_cpu_reference=True)
                report.update(prior_version=guide.version,extra_guidance_simulations=0,
                              teacher_horizon=0,opponent_mode='policy')
                rows = []
                for i in range(args.roots):
                    budget()
                    first = 'ab'[i % 2]
                    a = seven_lineup.KEYS[(i - i % 2) % 7]
                    b = seven_lineup.KEYS[(i + 1 - i % 2) % 7]
                    params = dict(seed=args.seed+1000+i, first_side=first, skip_mulligan=True,
                                  decks={'a': models[a].serving_deck, 'b': models[b].serving_deck})
                    state = runtime.initialize([params])[0]
                    expected = new_game(**params)
                    compare_boundary(unpack_game_states(state.batch)[0], expected, 'native SoA search initialize')
                    viewer = runtime.actor(state)
                    seed = (1 << 256) | int.from_bytes(hashlib.sha256(
                        f'native-soa:{args.seed}:{i}:{viewer}'.encode()).digest(), 'big')
                    policies = {'a': models[a], 'b': models[b]}
                    options = dict(seed=seed, simulations=32, gumbel_candidates=16, max_depth=10,
                                   noise=True, q_scale='natural_wdl',root_prior=guide,root_trace=True)
                    root_start=time.monotonic()
                    native = runtime.search(state, viewer, policies, **options,
                                            deadline=time.time()+args.seconds-(time.monotonic()-start))
                    native_seconds=time.monotonic()-root_start
                    oracle_start=time.monotonic()
                    original = formal_search(expected, viewer, policies, runtime=seven_runtime,
                        terminal_horizon=0, teacher_mode='network', **options,
                        deadline=time.time()+args.seconds-(time.monotonic()-start))
                    exact = {name: bool(np.array_equal(native[name], original[name]))
                             for name in ('x', 'c', 'visits', 'root_prior', 'root_gumbel','prior_shifts')}
                    row = dict(root=i, first=first, decks={'a': a, 'b': b},
                        requested=native['requested'], completed=native['simulations'],
                        exact=exact, same_choice=native['search_choice'] == original['search_choice'],
                        same_value=native['value'] == original['value'],
                        pi_close=bool(np.allclose(native['pi'],original['pi'],rtol=1e-7,atol=1e-9)),
                        trace_equal=native.get('root_trace')==original.get('root_trace'),
                        native_seconds=native_seconds,oracle_seconds=time.monotonic()-oracle_start,
                        visits=native['visits'].tolist(),pi=native['pi'].tolist(),
                        mean_values=native['mean_values'].tolist(),prior_shifts=native['prior_shifts'].tolist(),
                        root_trace=native['root_trace'])
                    np.savez_compressed(args.output/('search-root-%02d.npz'%i),
                        **{name:np.asarray(native[name]) for name in
                           ('x','c','visits','pi','root_prior','root_gumbel','prior_shifts','mean_values')})
                    rows.append(row)
                    report['search_rows']=rows
                    save('search')
                    if (not native['complete'] or not original['complete'] or not all(exact.values())
                            or not row['same_choice'] or not row['same_value'] or not row['pi_close']
                            or not row['trace_equal']):
                        raise AssertionError(['Native SoA full32 search mismatch', i, row])
                report['search_rows'] = rows
                report['software_search'] = len(rows) == args.roots
                report['search_initial_states_verified']=len(rows)
                report['software_initialize']=True
                if not report['software_search']:
                    raise AssertionError('Requested native SoA searches incomplete')
            report['passed'] = ((args.games==0 or report['software_natural_games'])
                                and (report['software_mechanisms'] if args.mechanism_tests else True)
                                and (report['software_search'] if args.models else True))
            save('complete')
    except BaseException:
        report['passed'] = False
        report['error'] = traceback.format_exc()
        save('failed')
        raise
    finally:
        save(report.get('phase', 'failed'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--run', action='store_true')
    p.add_argument('--backend',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--program', type=Path)
    p.add_argument('--games', type=int, default=14)
    p.add_argument('--roots', type=int, default=8)
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
    if (not 0 <= args.games <= 256 or not 1 <= args.roots <= 64 or not 1 <= args.max_decisions <= 2000
            or not 1 <= args.seconds <= 7200 or not 4096 <= args.scratch_nodes <= 4194304
            or not 131072 <= args.scratch_payload <= 134217728 or not 1 <= args.batch_rows <= 32
            or not 1 <= args.rule_steps <= 2**30
            or args.batch_rows * (args.scratch_nodes*32+args.scratch_payload) > 4*1024**3):
        p.error('Bounded games, searches, decisions and seconds required')
    if not args.run:
        print(json.dumps(plan(args), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    if args.output is None:
        p.error('--run requires a new --output directory')
    args.output = args.output.resolve()
    eval_root=(ROOT/'artifacts'/'rl-evals').resolve()
    try:
        args.output.relative_to(eval_root)
    except ValueError:
        p.error('--output must be a new artifacts/rl-evals directory')
    if args.output==eval_root:
        p.error('--output must be a new child directory')
    if args.games==0 and not args.models and not args.mechanism_tests:
        p.error('Request natural games, mechanism tests, or actual-model searches')
    if sys.version_info[:2] != (3, 10) and args.program is None:
        p.error('--program is required outside CPython 3.10; no runtime recapture')
    main(args)
