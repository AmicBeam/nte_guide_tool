"""Freeze and compare inference-only root recommendations; no model updates.

Uses an explicitly supplied prior evaluation's immutable, SHA-listed sources
and numeric serving models. Only this task's trace and recommendation modules
are overlaid into a NEW snapshot. Existing evaluations are never edited.
"""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  VECLIB_MAXIMUM_THREADS='1')
import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

MODES = ('legacy', 'completed_sweep_gumbel', 'completed_sweep_value')
OWNED = ('app/modules/card_game/rl/recovery_search.py',
         'app/modules/card_game/rl/root_recommendation.py')
OUTPUT = None


def write(path, data):
    def default(value):
        if hasattr(value, 'tolist'): return value.tolist()
        raise TypeError(type(value).__name__)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default) + '\n')
    temporary.replace(path)


def state_hash(state):
    return sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_raw(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def initialize(output):
    global OUTPUT
    OUTPUT = Path(output)
    source = OUTPUT / 'sources' / 'experiment'
    os.environ['NTE_DUEL_AI_MODEL_DIR'] = str(source / 'app/modules/card_game/engine/ai/models')
    os.environ['NTE_DATABASE_PATH'] = str(OUTPUT / 'unused-isolated.db')
    sys.path.insert(0, str(source))


def profile_action(state, side, model, mode):
    from app.modules.card_game.engine.ai import advanced_model as ai
    from app.modules.card_game.engine import duel_v2 as engine
    from app.modules.card_game.rl import recovery_search
    from app.modules.card_game.rl.root_recommendation import recommend
    captured = {}
    original = recovery_search.search
    apply = engine.apply_action
    def proof(*args, **kwargs):
        captured['proof_checks'] = captured.get('proof_checks', 0) + 1
        return apply(*args, **kwargs)
    def searched(*args, **kwargs):
        try:
            result = original(*args, **kwargs, root_trace=True)
            recommendations = {m: recommend(result, m)[1] for m in MODES}
        except Exception:
            captured['search_error'] = traceback.format_exc()
            raise
        captured.update(result=result, recommendations=recommendations)
        # Only the chosen recommendation changes. Allocation and pi are intact.
        result = dict(result, search_choice=recommendations[mode]['selected'])
        return result
    profile = state.get('ai_profile')
    state['ai_profile'] = dict(deck_id=model.deck_id, version=model.version,
                              build_sha256=model.serving_build_sha256)
    started = time.monotonic()
    try:
        recovery_search.search = searched
        engine.apply_action = proof
        action = ai.choose_action(state, side)
    finally:
        recovery_search.search = original
        engine.apply_action = apply
        if profile is None: state.pop('ai_profile', None)
        else: state['ai_profile'] = profile
    captured.update(seconds=time.monotonic() - started, selected=action)
    if 'result' in captured:
        assert captured.get('proof_checks', 0) + captured['result']['simulations'] <= 32
    return action, captured


def load_models(keys):
    from app.modules.card_game.engine.ai import advanced_model as ai
    cfg = json.loads((OUTPUT / 'config.json').read_text())
    models = {key: ai.load_model(key) for key in keys}
    for key, model in models.items():
        assert model.version == cfg['models'][key]['sha256']
        assert model.serving_build_sha256 == cfg['models'][key]['build_sha256']
    return models


def diagnose(output):
    if (Path(output) / 'diagnostic.json').exists():
        raise ValueError('Diagnostic already exists; preserve its original records')
    initialize(output)
    from app.modules.card_game.engine import duel_v2 as engine
    from app.modules.card_game.rl.league_rollout import clean
    cfg = json.loads((OUTPUT / 'config.json').read_text())
    previous = Path(cfg['input'])
    selections = json.loads((previous / 'branch-selection.json').read_text())
    models = load_models(('zhenhong', 'starter'))
    rows = []
    for selection in selections:
        if 'fork_action_index' not in selection: continue
        raw = read_raw(previous / 'raw' / (selection['parent'] + '.json.gz'))
        state = deepcopy(raw['opening_state'])
        step = selection['fork_action_index']
        for record in raw['actions'][:step]:
            state = clean(engine.apply_action(state, record['side'], record['action']))
            assert state_hash(state) == record['state_sha256']
        before = state_hash(state)
        action, root = profile_action(state, 'a', models['zhenhong'], 'legacy')
        assert state_hash(state) == before
        root.update(game=selection['parent'], step=step, turn=state['turn'],
                    original=selection['original_action'], state_sha256=before,
                    same_original_action=action == selection['original_action'],
                    archived_root=selection['original_root'])
        # Separate private snapshot and paired, same-information worlds. Do
        # not feed the original opponent hand/ordering to a recommendation.
        from app.modules.card_game.rl import information_search as core, recovery_runtime
        worlds = []
        result = root.get('result')
        if result:
            from app.modules.card_game.rl.root_recommendation import recommend
            for number in range(3):
                world = core.sample_world(state, 'a', cfg['seeds']['probes'] + step * 10 + number,
                                          runtime=recovery_runtime)
                world_rows = {}
                for mode in MODES:
                    candidate = result['actions'][recommend(result, mode)[0]]
                    passive = world['sides']['a']['characters']['zhenhong'].get('surplus_passive_triggers', 0)
                    after = engine.apply_action(deepcopy(world), 'a', candidate)
                    world_rows[mode] = dict(action=candidate, phase=after['phase'], winner=after.get('winner'),
                        passive_delta=after['sides']['a']['characters']['zhenhong'].get('surplus_passive_triggers', 0) - passive)
                worlds.append(world_rows)
        root['information_set_checks'] = worlds
        rows.append(root)
        write(OUTPUT / 'diagnostic.json', rows)
        print(json.dumps(dict(game=root['game'], step=step, same=root['same_original_action'],
            choices={m: result['actions'][r['selected']] for m, r in root.get('recommendations', {}).items()}),
            ensure_ascii=False), flush=True)


def execute(job):
    from app.modules.card_game.engine import duel_v2 as engine
    from app.modules.card_game.rl.league_rollout import clean
    from app.modules.card_game.rl.report_telemetry import GameTelemetry
    from app.modules.card_game.rl.offline_analysis import _public_payload
    policies = load_models((job['learner'], job['foe']))
    models = {'a': policies[job['learner']], 'b': policies[job['foe']]}
    state = engine.new_game(seed=job['seed'], first_side=job['first'],
                            decks={s: m.serving_deck for s, m in models.items()})
    opening = deepcopy(state)
    telemetry = GameTelemetry(state)
    actions = []; events = []; roots = []
    started = time.monotonic(); error = None
    try:
        for step in range(800):
            if state['phase'] == 'finished' or time.time() >= job['deadline']: break
            side = engine.acting_side(state)
            legal = [a for a in engine.legal_actions(state, side) if a['type'] != 'concede']
            action, root = profile_action(state, side, models[side], job['mode'] if side == 'a' else 'legacy')
            assert action in legal
            root.update(step=step, turn=state['turn'], side=side)
            roots.append(root)
            telemetry.before(state, side, legal)
            after = engine.apply_action(state, side, action)
            telemetry.after(state, after, action, side)
            events.extend(deepcopy(after.get('events', [])))
            state = clean(after)
            actions.append(dict(side=side, action=action, state_sha256=state_hash(state)))
            write(OUTPUT / 'progress' / (job['id'] + '.json'),
                  dict(step=step + 1, turn=state['turn'], seconds=time.monotonic() - started))
    except Exception:
        error = traceback.format_exc()
    complete = state['phase'] == 'finished' and error is None
    verified = False
    if complete:
        try:
            replay = deepcopy(opening)
            for record in actions:
                replay = clean(engine.apply_action(replay, record['side'], record['action']))
                assert state_hash(replay) == record['state_sha256']
            assert replay['winner'] == state['winner']
            verified = True
        except Exception:
            error = traceback.format_exc(); complete = False
    summary = dict(**job, complete=complete, winner=state.get('winner'), turn=state['turn'],
                   error=error, replay_verified=verified, seconds=time.monotonic() - started,
                   decisions=len(roots), training_updates=0)
    raw = dict(summary, opening_state=opening, actions=actions, events=events,
               roots=roots, telemetry=telemetry.finish(state))
    temporary = OUTPUT / 'raw' / (job['id'] + '.json')
    write(temporary, raw)
    (temporary.with_suffix('.json.gz')).write_bytes(gzip.compress(temporary.read_bytes()))
    temporary.unlink()
    if complete:
        state['events'] = events
        public = _public_payload(opening, state, game_id=job['id'])
        write(OUTPUT / 'replays' / (job['id'] + '.json'), public)
    write(OUTPUT / 'progress' / (job['id'] + '.json'), summary)
    return summary


def run_games(output):
    root = Path(output); cfg = json.loads((root / 'config.json').read_text())
    if (root / 'launch.json').exists() or (root / 'results.json').exists():
        raise ValueError('Evaluation already launched; do not overwrite frozen records or reuse seeds')
    deadline = time.time() + cfg['seconds']
    results = []
    write(root / 'launch.json', dict(started=time.time(), deadline=deadline, pid=os.getpid()))
    with ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=get_context('spawn'),
                             initializer=initialize, initargs=(str(root),)) as pool:
        futures = {pool.submit(execute, dict(job, deadline=deadline)): job for job in cfg['jobs']}
        for future in as_completed(futures):
            job = futures[future]
            try: result = future.result()
            except Exception: result = dict(job, complete=False, error=traceback.format_exc())
            results.append(result); write(root / 'results.json', results)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    write(root / 'status.json', dict(planned=len(cfg['jobs']), completed=sum(r['complete'] for r in results),
                                    errors=sum(bool(r.get('error')) for r in results), training_updates=0))


def report(output):
    initialize(output)
    import importlib.util
    import math
    import numpy as np
    from app.modules.card_game.engine import duel_v2 as engine
    from app.modules.card_game.engine.duel_v2.replay_log import assemble_replay_game
    from app.modules.card_game.rl.replay_export import public_file_ok
    from app.modules.card_game.rl.root_recommendation import recommend
    from app.modules.card_game.rl.league_rollout import clean
    from app.modules.card_game.rl.cross_lineup import identity
    cfg = json.loads((OUTPUT / 'config.json').read_text())
    assert identity() == cfg['rule_hash']
    final_path = Path(__file__).resolve().parents[1] / OWNED[1]
    final_spec = importlib.util.spec_from_file_location('final_root_recommendation_check', final_path)
    final_module = importlib.util.module_from_spec(final_spec)
    final_spec.loader.exec_module(final_module)
    results = json.loads((OUTPUT / 'results.json').read_text())
    planned = {r['id']: r for r in cfg['jobs']}
    assert len(results) == len({r['id'] for r in results}) == len(planned)
    frozen = OUTPUT / 'sources/experiment'
    manifest = json.loads((OUTPUT / 'source-manifest.json').read_text())
    for name, entry in manifest.items():
        assert sha256((frozen / name).read_bytes()).hexdigest() == entry['sha256'], name
    old_manifest = json.loads((OUTPUT / 'original-manifest.json').read_text())
    previous = Path(cfg['input'])
    evidence_path = OUTPUT / 'input-evidence-manifest.json'
    evidence = json.loads(evidence_path.read_text())['files'] if evidence_path.exists() else {}
    for name, digest in evidence.items():
        assert sha256((previous / name).read_bytes()).hexdigest() == digest, name
    for name, entry in old_manifest.items():
        assert sha256((previous / 'sources/new' / name).read_bytes()).hexdigest() == entry['sha256'], name
        if name not in OWNED:
            assert manifest[name] == entry, name
    for key, entry in cfg['models'].items():
        p = frozen / 'app/modules/card_game/engine/ai/models/recovery-20261007' / (key + '.npz')
        assert sha256(p.read_bytes()).hexdigest() == entry['sha256']
        with np.load(p, allow_pickle=False) as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
    raws = {}; public_views = 0; fallbacks = []; final_root_checks = 0
    stats = {}; paired = []; prefixes = []
    for row in results:
        raw = read_raw(OUTPUT / 'raw' / (row['id'] + '.json.gz'))
        raws[row['id']] = raw
        for key in planned[row['id']]: assert raw[key] == planned[row['id']][key]
        assert raw['training_updates'] == 0
        state = deepcopy(raw['opening_state'])
        for index, (action, root) in enumerate(zip(raw['actions'], raw['roots'])):
            legal = [a for a in engine.legal_actions(state, action['side']) if a['type'] != 'concede']
            assert action['action'] in legal
            result = root.get('result')
            if result:
                # JSON -> arrays for the frozen recommendation implementation.
                for field in ('visits', 'root_prior', 'mean_values', 'root_gumbel'):
                    result[field] = np.asarray(result[field])
                for mode in MODES:
                    assert recommend(result, mode)[0] == root['recommendations'][mode]['selected']
                    assert final_module.recommend(result, mode)[0] == root['recommendations'][mode]['selected']
                final_root_checks += 1
                assert root.get('proof_checks', 0) + result['simulations'] <= 32
                assert action['action'] == result['actions'][root['recommendations'][row['mode'] if action['side'] == 'a' else 'legacy']['selected']]
            after = engine.apply_action(state, action['side'], action['action'])
            if not result and len(legal) > 1:
                proof = (root.get('proof_checks', 0) > 0 and after['phase'] == 'finished'
                         and after.get('winner') == action['side'] and after['rng'] == state['rng']
                         and not any(e['type'] in ('draw', 'gain') for e in after.get('events', [])))
                if not proof:
                    fallbacks.append(dict(game=row['id'], step=index, reason=root.get('search_error', 'no_tree_result')))
            state = clean(after)
            assert state_hash(state) == action['state_sha256']
        if row['complete']:
            assert raw['replay_verified'] and state['winner'] == row['winner']
            public = json.loads((OUTPUT / 'replays' / (row['id'] + '.json')).read_text())
            public_file_ok(public)
            for view in public['views'].values():
                game = assemble_replay_game(view['opening_board'], view['events'])
                assert game['phase'] == 'finished' and game['winner'] == row['winner']
                public_views += 1
    matchups = sorted({(r['learner'], r['foe']) for r in results})
    for learner, foe in matchups:
        group = [r for r in results if (r['learner'], r['foe']) == (learner, foe)]
        for mode in MODES:
            rows = [r for r in group if r['mode'] == mode]
            roots = [root for r in rows for root in raws[r['id']]['roots'] if root['side'] == 'a']
            trees = [r['result'] for r in roots if 'result' in r]
            normal = [r for r in rows if r['complete']]
            sample = [r['seconds'] for r in roots]
            def pos(first):
                x = [r for r in normal if r['first'] == first]
                return dict(wins=sum(r['winner'] == 'a' for r in x), normal=len(x))
            stats[f'{learner}/{foe}/{mode}'] = dict(learner=learner, foe=foe, mode=mode,
                planned=len(rows), normal=len(normal), wins=sum(r['winner'] == 'a' for r in normal),
                attempted=sum(r.get('decisions', 0) > 0 for r in rows),
                losses=sum(r['winner'] == 'b' for r in normal), draws=sum(r['winner'] not in ('a', 'b') for r in normal),
                errors=sum(bool(r.get('error')) for r in rows), truncated=sum(not r['complete'] and not r.get('error') for r in rows),
                first=pos('a'), second=pos('b'), decisions=len(roots), tree_decisions=len(trees),
                partial_trees=sum(not r['complete'] for r in trees), tree_simulations=sum(r['simulations'] for r in trees),
                proof_checks=sum(r.get('proof_checks', 0) for r in roots),
                changed_at_same_root=sum(r['recommendations'][mode]['selected'] != r['recommendations']['legacy']['selected']
                                          for r in roots if 'result' in r),
                p50_seconds=float(np.median(sample)) if sample else None,
                p95_seconds=float(np.percentile(sample, 95)) if sample else None,
                max_seconds=max(sample, default=None), game_seconds=sum(r.get('seconds', 0) for r in rows))
        for mode in MODES[1:]:
            clusters = []
            for seed in sorted({r['seed'] for r in group}):
                cells = {(r['mode'], r['first']): r for r in group if r['seed'] == seed}
                needed = [(m, f) for m in ('legacy', mode) for f in ('a', 'b')]
                if not all(k in cells and cells[k]['complete'] for k in needed): continue
                difference = sum(int(cells[mode, f]['winner'] == 'a') - int(cells['legacy', f]['winner'] == 'a')
                                 for f in ('a', 'b')) / 2
                clusters.append(dict(seed=seed, difference=difference))
                for first in ('a', 'b'):
                    base = raws[cells['legacy', first]['id']]['actions']
                    candidate = raws[cells[mode, first]['id']]['actions']
                    divergence = next((i for i, (a, b) in enumerate(zip(base, candidate)) if
                                       (a['side'], a['action']) != (b['side'], b['action'])), None)
                    prefixes.append(dict(learner=learner, foe=foe, seed=seed, first=first, mode=mode,
                                         first_different_action=divergence))
            mean = sum(c['difference'] for c in clusters) / len(clusters) if clusters else None
            radius = math.sqrt(2 * math.log(2 * 6 / .05) / len(clusters)) if clusters else None
            paired.append(dict(learner=learner, foe=foe, mode=mode, clusters=clusters, mean_difference=mean,
                               interval95_family6=[max(-1., mean - radius), min(1., mean + radius)] if clusters else None))
    verification = dict(frozen_members=len(manifest),
                        source_members=sum('__pycache__' not in name for name in manifest),
                        initial_bytecode_members=sum('__pycache__' in name for name in manifest),
                        original_members_unchanged=len(old_manifest), rule_hash=identity(),
                        input_evidence_members_unchanged=len(evidence),
                        final_recommender_sha256=sha256(final_path.read_bytes()).hexdigest(),
                        frozen_recommender_sha256=manifest[OWNED[1]]['sha256'],
                        final_recommender_root_checks=final_root_checks,
                        model_shas=list(cfg['models']), normal=sum(r['complete'] for r in results),
                        public_views_verified=public_views, unexpected_fallbacks=fallbacks,
                        action_sha_replays=len(results), torch_imported='torch' in sys.modules,
                        training_updates=0, quality_approved=False)
    write(OUTPUT / 'analysis.json', dict(stats=stats, paired=paired, prefixes=prefixes, verification=verification))
    write(OUTPUT / 'verification.json', verification)
    attempts = sum(r.get('decisions', 0) > 0 for r in results)
    normal_count = sum(r['complete'] for r in results)
    lines = ['# 根最终选点：冻结有界对照', '',
             '本轮仅改变A方搜索末端推荐，双方模型、构筑和搜索预算冻结；B方保持原搜索。已知六个根只作诊断，不进入下表。', '',
             f'预定{len(planned)}局，实际开始操作{attempts}局、正常终局{normal_count}局；'
             f'异常{sum(bool(r.get("error")) for r in results)}局，未完成{len(results)-normal_count}局'
             f'（其中未开始操作{len(results)-attempts}局）。不以缺局缩小计划数或补造胜负。', '',
             '| 对阵（A 对 B） | 模式 | 计划/尝试/正常 | A胜/正常 | A先手 | A后手 | 异常/未完成 | 决策p50/p95秒 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    def seconds(value):
        return f'{value:.3f}' if value is not None else '未评估'
    for s in stats.values():
        lines.append(f"| {s['learner']} / {s['foe']} | {s['mode']} | {s['planned']}/{s['attempted']}/{s['normal']} | {s['wins']}/{s['normal']} | "
                     f"{s['first']['wins']}/{s['first']['normal']} | {s['second']['wins']}/{s['second']['normal']} | "
                     f"{s['errors']}/{s['truncated']} | {seconds(s['p50_seconds'])}/{seconds(s['p95_seconds'])} |")
    lines += ['', '预定独立样本单位是8个新根种子簇；每簇安排交换先后手及三个变体，共48局。'
              '各对阵仅2或4簇。配对差异的95%区间采用[-1,1]簇均值的Hoeffding界，六项比较作Bonferroni校正；'
              '如此小的样本区间很宽，不提供强度批准。未完成配对不进入差异分母。', '',
              '原始逐决策评分、完整轮次和实际访问在 `raw/`；公开录像在 `replays/`。'
              '每局均按实际动作重放状态SHA，公开录像核验隐藏信息与双方视角赢家；详见 `verification.json`。', '',
              '完整轮次模式只改变候选资格；价值模式另以一份根价值伪观测收缩均值。后者是启发式，'
              '并不保证Q准确或跨隐藏世界胜利。默认网站和训练算法、模型权重与卡效均未修改。']
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(verification, ensure_ascii=False), flush=True)


def prepare(args):
    repo = Path(__file__).resolve().parents[1]
    previous = args.input.resolve(); output = args.output.resolve()
    if output.exists(): raise ValueError('Output must be new')
    old_config = json.loads((previous / 'config.json').read_text())
    old_manifest = json.loads((previous / 'new-source-manifest.json').read_text())
    source = previous / 'sources' / 'new'
    for name, item in old_manifest.items():
        p = source / name
        if p.is_symlink() or not p.resolve().is_relative_to(source.resolve()) or '..' in Path(name).parts:
            raise ValueError('Unsafe snapshot member')
        assert sha256(p.read_bytes()).hexdigest() == item['sha256'], name
    output.mkdir(parents=True)
    frozen = output / 'sources' / 'experiment'
    for name in old_manifest:
        p = frozen / name; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, p)
    for name in OWNED:
        target = frozen / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, target)
    shutil.copyfile(__file__, output / 'run.py')
    sys.path.insert(0, str(repo))
    from app.modules.card_game.rl.build_acceptance import claim_seeds
    seeds = claim_seeds(repo / 'artifacts/rl-seed-ledger.json', str(output))
    # Model manifests are copied verbatim; load them in the isolated child,
    # never rewrite source training identity or service authorization.
    model_script = """import sys,json;sys.path.insert(0,sys.argv[1]);from app.modules.card_game.engine.ai import advanced_model as ai
out={}
for key in ('zhenhong','starter','weave-rush','quick-rush'):
 m=ai.load_model(key);out[key]=dict(sha256=m.version,build=m.serving_deck,build_sha256=m.serving_build_sha256)
print(json.dumps(out,ensure_ascii=False))
"""
    env = dict(os.environ, NTE_DUEL_AI_MODEL_DIR=str(frozen / 'app/modules/card_game/engine/ai/models'))
    models = json.loads(subprocess.check_output([sys.executable, '-c', model_script, str(frozen)],
                       cwd=output, env=env, text=True))
    for key in old_config['models']:
        assert models[key] == old_config['models'][key]
    jobs = []
    matchups = [('zhenhong', 'starter', args.pairs), ('starter', 'starter', 2),
                ('weave-rush', 'quick-rush', 2)]
    number = 0
    for learner, foe, pairs in matchups:
        for pair in range(pairs):
            for first in ('a', 'b'):
                for mode in MODES:
                    jobs.append(dict(id=f'{learner}-{pair:02}-{first}-{mode}', learner=learner, foe=foe,
                                     mode=mode, first=first, seed=seeds['audit'] + number, cluster=number))
            number += 1
    manifest = {str(p.relative_to(frozen)): dict(sha256=sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
                for p in frozen.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write(output / 'source-manifest.json', manifest)
    write(output / 'original-manifest.json', old_manifest)
    config = dict(input=str(previous), models=models, seeds=seeds, jobs=jobs,
                  seconds=args.seconds, workers=args.workers, modes=MODES,
                  rule_hash=old_config['rule_variants']['new']['rule_hash'],
                  search=old_config['search'], training_updates=0, selection='exploratory_no_promotion',
                  inference_only=True, prior_pseudo_observations=1, minimum_finalist_visits=2,
                  opponent_recommendation='legacy', noisy_roots=False,
                  diagnostic='Six posthoc loss roots; excluded from independent game results',
                  statistics='Paired seed clusters; report all three predeclared variants, no sample extension',
                  scope='Bounded search inference comparison, not full model/5x5 acceptance')
    write(output / 'config.json', config)
    for folder in ('raw', 'replays', 'progress'): (output / folder).mkdir()
    print(json.dumps(dict(output=str(output), games=len(jobs), clusters=number)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--seconds', type=int, default=1200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--phase', choices=('prepare', 'diagnose', 'games', 'report'), default='prepare')
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.input is None or not 1 <= args.pairs <= 16 or not 60 <= args.seconds <= 1800 or not 1 <= args.workers <= 4:
            parser.error('Explicit input, pairs 1..16, seconds 60..1800 and workers 1..4 required')
        prepare(args)
    elif args.phase == 'diagnose': diagnose(args.output.resolve())
    elif args.phase == 'report': report(args.output.resolve())
    else: run_games(args.output.resolve())


if __name__ == '__main__':
    main()
