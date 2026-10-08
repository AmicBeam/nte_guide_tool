"""Five-team recovery orchestration. No training or Torch import on import.

All samples come from complete official games. Selection and value checks
are newly reserved, never replayed into policy training. Readiness is separate
from strength and from the complete-report specification.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import math
import time

from .cross_schedule import ROSTER, learning_schedule, matrix_schedule, coverage


def learning_keys(config):
    keys = tuple(config.get('learning_keys', ROSTER))
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= set(ROSTER):
        raise ValueError('Distinct known learning keys required')
    return keys


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)); tmp.replace(path)


def fingerprint():
    from .recovery_checkpoint import algorithm_identity
    root=Path(__file__).resolve().parents[4]
    content=algorithm_identity()+Path(__file__).read_text()
    for relative in ('scripts/train_duel_v2_current_round.py','app/modules/card_game/rl/full_cycle_experiment.py',
                     'app/modules/card_game/rl/recovery_policy.py','app/modules/card_game/rl/recovery_checkpoint.py','app/modules/card_game/rl/episode_replay.py'):
        content+=(root/relative).read_text()
    return sha256(content.encode()).hexdigest()


def play(job):
    from .recovery_runtime import search_game
    return search_game(job)


def initialize(folder, hidden, device, seed, source=None):
    import torch
    from . import recovery_runtime as rt, preserved_policy, residual_runtime
    from scripts.train_duel_v2_current_round import preset_decks
    torch.manual_seed(seed); decks = preset_decks()
    nets = {}; origins = {}
    for key in ROSTER:
        if source:
            net, build, origin = rt.restore(source, key, device)
            if build != decks[key]:
                raise ValueError('Recovery source must match this fixed build')
            origin = dict(origin, recovery_source=str(source), recovery_source_sha=rt.model(source, key).version)
        elif key in ROSTER[:3]:
            net, _ = preserved_policy.create_network(key, hidden=hidden, device=device, partial=True)
            origin = dict(initialization='frozen_legacy_plus_zero_residual',
                          unsupported_root='residual_only', optimizer_reset=True)
        else:
            net = residual_runtime.create_network(hidden, device)
            origin = dict(initialization='fresh_residual_no_teacher', optimizer_reset=True)
        if not hasattr(net,'value_net'):
            from .recovery_policy import attach
            net = attach(net)
            origin=dict(origin,value_tower='independent_clone',value_policy_gradients_separated=True)
        rt.export(net, folder, key, decks[key], origin)
        nets[key] = net; origins[key] = origin
    return nets, decks, origins


def versions(folder):
    from .recovery_runtime import model
    return {key: model(folder, key).version for key in ROSTER}


def jobs_for(models, decks, base, deadline, *, pure=True, cycle=0, mirrors=True, report=None,
             simulations=32, historical=None,teacher_mode='network',covered_value_source=None,
             keys=None):
    selected = learning_keys({'learning_keys': keys}) if keys is not None else ROSTER
    plan = learning_schedule()
    if mirrors:
        plan += [dict(left=k, right=k, first=f) for k in ROSTER for f in ('a', 'b')]
    jobs = []
    for i, item in enumerate(plan):
        sides = [s for s, key in (('a', item['left']), ('b', item['right'])) if key in selected]
        if not sides:
            continue
        j = dict(item, id=f'cycle-{cycle}-{i}', seed=base+i//2, pure=pure,
                 policies={'a': (str(models), item['left']), 'b': (str(models), item['right'])},
                 decks={'a': decks[item['left']], 'b': decks[item['right']]},
                 deadline=deadline, training=not pure, train_sides=sides,
                 q_scale='natural_wdl', value_precheck=report, simulations=simulations,teacher_mode=teacher_mode,
                 covered_value_source=covered_value_source,
                 gumbel_candidates=16, terminal_horizon=0 if teacher_mode in ('covered_terminal','covered_expert') else 3, max_actions=800)
        if historical:
            for learner in sides:
                clone = deepcopy(j); enemy = 'b' if learner == 'a' else 'a'
                clone['id'] += learner; clone['train_sides'] = [learner]
                clone['policies'][enemy] = (str(historical), clone['policies'][enemy][1])
                jobs.append(clone)
        else:
            jobs.append(j)
    return jobs


def collect(pool, jobs, folder, *, learning=False, require_search=False):
    from .episode_replay import dump_episode
    folder = Path(folder); records = []; games = []
    for job, result in zip(jobs, pool.map(play, jobs)):
        record = {k: job[k] for k in ('id', 'seed', 'first', 'left', 'right') if k in job}
        record.update(complete=result['complete'], winner=result.get('winner'), reason=result.get('reason'))
        records.append(record)
        if result['complete']:
            if (learning and not result['learning_eligible']) or ((learning or require_search) and
                    (result['decisions'] != result['searched'] or result['searched'] != len(result['rows']))):
                raise ValueError('Complete search labels required')
            dump_episode(folder / 'episodes' / f"{job['id']}.json.gz", job, result)
        games.append(result)
    write(folder / 'games.json', records)
    if not games or not all(g['complete'] for g in games):
        raise ValueError('Incomplete phase; no partial result grants readiness')
    return games


def value_metrics(games, policies, *, backend='numpy', device=None):
    if backend == 'cuda_batched':
        if device != 'cuda':
            raise ValueError('CUDA metrics cannot silently use a CPU device')
        from .batched_search.value_metrics import compute
        return compute(games, policies, device=device)
    if backend != 'numpy':
        raise ValueError('Unknown WDL metric backend')
    from scripts.check_duel_v2_value_calibration import probability_metrics
    result = {}
    for key, policy in policies.items():
        probabilities = []; labels = []; weights = []; count = 0
        for game in games:
            rows = [r for r in game['value_rows'] if r['key'] == key]
            if not rows: continue
            count += 1
            for row in rows:
                probabilities.append(policy.wdl(row['x'], row['value_actor']))
                labels.append(int(row['z'])+1); weights.append(1/len(rows))
        result[key] = dict(**probability_metrics(probabilities, labels, weights), normal_games=count)
    return result


def value_calibration_protocol(config):
    """A readiness receipt binds the exact calibration method and sample budget."""
    full = config.get('value_full_tower', False)
    return dict(full_tower=full, train_replicas=config.get('value_replicas', 2),
                check_replicas=config.get('value_check_replicas', 1),
                epoch_candidates=[1, 2, 4, 8] if full else [config.get('value_passes', 16)],
                temperature_candidates=[1., 2., 4.])


def warm_value(pool, nets, decks, origins, models, folder, config, cycle, deadline,*,seed_offset=0):
    """Separate value training, selection, and heldout; policy arrays are frozen."""
    import numpy as np
    import torch
    from . import recovery_runtime as rt
    if seed_offset not in (0,6000):raise ValueError('Declared disjoint value seed slot required')
    protocol = value_calibration_protocol(config)
    metric_fn = value_metrics
    if config.get('value_metric_backend', 'numpy') != 'numpy':
        from functools import partial
        metric_fn = partial(value_metrics, backend=config['value_metric_backend'], device=config['device'])
    full = protocol['full_tower']
    folder = Path(folder); before = {k: {n: t.detach().clone() for n, t in net.state_dict().items()
                                       if not n.startswith(('wdl.', 'value_net.'))} for k, net in nets.items()}
    if full and any(not hasattr(n, 'value_net') for n in nets.values()):
        raise ValueError('Full calibration requires a separate value tower')
    # Disjoint per-cycle 10k blocks; each phase uses fewer than 1k roots.
    train_jobs = []
    for replica in range(config['value_replicas']):
        train_jobs += jobs_for(models, decks, config['seeds']['adaptation']+cycle*10000+replica*100+seed_offset,
                              deadline, cycle=replica)
    games = collect(pool, train_jobs, folder / 'train')
    def check_jobs(source, base):
        return [j for replica in range(protocol['check_replicas'])
                for j in jobs_for(source, decks, base+replica*100, deadline, cycle=replica)]
    # Argmax trajectories depend only on unchanged policy arrays. Collect once;
    # neither the epoch grid nor temperature grid can change these opponents.
    selection = collect(pool, check_jobs(models,
        config['seeds']['selection']+cycle*10000+seed_offset), folder / 'selection') if full else None
    chosen = {}; chosen_epochs = {}; update_counts = {}
    for key, net in nets.items():
        parameters = list(net.wdl.parameters()) + (list(net.value_net.parameters()) if full else [])
        opt = torch.optim.Adam(parameters, lr=.0003 if full else .003)
        rows = [r for g in games for r in g['value_rows'] if r['key'] == key]
        if not rows: raise ValueError('Missing complete calibration training rows')
        if full:
            # Give complete games equal total weight, as in heldout metrics.
            represented = [g for g in games if any(r['key'] == key for r in g['value_rows'])]
            rows = [dict(r, sample_weight=len(rows)/len(represented)/len(own))
                    for g in represented for own in [[r for r in g['value_rows'] if r['key']==key]] for r in own]
        rng = np.random.default_rng(config['seeds']['adaptation']+cycle*10000)
        best = None; updates = 0
        for epoch in range(1, max(protocol['epoch_candidates'])+1):
            indexes = rng.permutation(len(rows))
            for start in range(0, len(rows), 128):
                if time.time() >= deadline: raise TimeoutError('Value warmup deadline')
                # The dedicated optimizer does not own policy tensors. Clear
                # their previous joint-step gradients before global clipping.
                net.zero_grad(set_to_none=True)
                rt.update(net, opt, [rows[int(i)] for i in indexes[start:start+128]],
                          warmup=not full, value_only=full)
                updates += 1
            if full and epoch in protocol['epoch_candidates']:
                original_w = net.wdl.weight.detach().clone(); original_b = net.wdl.bias.detach().clone()
                for temperature in protocol['temperature_candidates']:
                    with torch.no_grad():
                        net.wdl.weight.copy_(original_w/temperature); net.wdl.bias.copy_(original_b/temperature)
                    target=folder/'candidates'/f'epoch-{epoch}-temperature-{temperature}'
                    rt.export(net,target,key,decks[key],origins[key])
                    metric=metric_fn(selection,{key:rt.model(target,key)})[key]
                    score=(metric['log_loss'],epoch,temperature)
                    if best is None or score < best[0]:
                        best=(score,{n:t.detach().clone() for n,t in net.state_dict().items()
                                     if n.startswith(('wdl.','value_net.'))},updates)
                with torch.no_grad():
                    net.wdl.weight.copy_(original_w); net.wdl.bias.copy_(original_b)
        update_counts[key]=dict(total_attempted=updates,selected=updates)
        if full:
            for n,t in best[1].items():net.state_dict()[n].copy_(t)
            chosen_epochs[key]=best[0][1];chosen[key]=best[0][2]
            update_counts[key]['selected']=best[2]
            rt.export(net,folder/'models',key,decks[key],dict(origins[key],value_temperature=chosen[key],
                      value_epoch=chosen_epochs[key],value_calibration=protocol))
        if any(not torch.equal(t, net.state_dict()[n]) for n, t in before[key].items()):
            raise ValueError('Value calibration changed policy')
        if not full: rt.export(net, folder / 'uncalibrated', key, decks[key], origins[key])
    if not full:
        selection = collect(pool, check_jobs(folder / 'uncalibrated',
            config['seeds']['selection']+cycle*10000+seed_offset), folder / 'selection')
    # Temperature selection is confined to selection, never the heldout check.
    for key, net in ([] if full else nets.items()):
        scores = []
        original_w = net.wdl.weight.detach().clone(); original_b = net.wdl.bias.detach().clone()
        for temperature in (1., 2., 4.):
            with torch.no_grad():
                net.wdl.weight.copy_(original_w/temperature); net.wdl.bias.copy_(original_b/temperature)
            rt.export(net, folder / f'temperature-{temperature}', key, decks[key], origins[key])
            metric = metric_fn(selection, {key: rt.model(folder / f'temperature-{temperature}', key)})[key]
            scores.append((metric['log_loss'], temperature))
        chosen[key] = min(scores)[1]
        with torch.no_grad():
            net.wdl.weight.copy_(original_w/chosen[key]); net.wdl.bias.copy_(original_b/chosen[key])
        rt.export(net, folder / 'models', key, decks[key], dict(origins[key], value_temperature=chosen[key]))
    held = collect(pool, check_jobs(folder / 'models',
                   config['seeds']['selection']+cycle*10000+1000+seed_offset), folder / 'heldout')
    policies = {k: rt.model(folder / 'models', k) for k in ROSTER}
    report = dict(schema='recovery_value_precheck_v1', complete=True, heldout=True,
                  passed=True, rule_hash=config['rule_hash'], models=versions(folder / 'models'),
                  metrics=metric_fn(held, policies), temperatures=chosen,
                  independent_current_rule_seeds=True, policy_preserved=True,
                  calibration=protocol, selected_epochs=chosen_epochs, updates=update_counts,
                  phase_games=dict(training=len(games),selection=len(selection),heldout=len(held)))
    report['passed'] = rt.value_precheck_ok(report, policies)
    write(folder / 'value-precheck.json', report)
    return folder / 'models', report


def compare_jobs(candidate, reference, decks, seed, deadline, roots, *, search=False, reports=None,
                 simulations=32,teacher_mode='network',covered_value_source=None,keys=None):
    """Compare both versions against identical frozen opponents in both seats/initiatives."""
    jobs = []
    selected = learning_keys({'learning_keys': keys}) if keys is not None else ROSTER
    for k, key in enumerate(ROSTER):
        if key not in selected:
            continue
        for replica in range(roots):
            foe = ROSTER[(k+replica) % len(ROSTER)]
            for learner in ('a', 'b'):
                for first in ('a', 'b'):
                    for label, folder in (('candidate', candidate), ('reference', reference)):
                        enemy = 'b' if learner == 'a' else 'a'
                        jobs.append(dict(id=f'{key}-{replica}-{learner}-{first}-{label}', key=key,
                            left=key if learner == 'a' else foe, right=foe if learner == 'a' else key,
                            replica=replica, label=label, learner=learner, first=first,
                            seed=seed+k*100+replica, pure=not search, training=False,
                            policies={learner:(str(folder),key), enemy:(str(reference),foe)},
                            references={learner:(str(reference),key)}, train_sides=['a','b'],
                            decks={learner:decks[key],enemy:decks[foe]}, deadline=deadline,
                            value_precheck=reports, q_scale='natural_wdl', simulations=simulations,teacher_mode=teacher_mode,
                            covered_value_source=covered_value_source))
    for job in jobs:job['terminal_horizon']=0 if teacher_mode in ('covered_terminal','covered_expert') else 3
    return jobs


def comparison(pool, jobs, folder):
    games = collect(pool, jobs, folder)
    result = {k: dict(candidate=0., reference=0., games=0, changed=0, decisions=0,
                      plays=0, reference_plays=0, legal_play=0, unsupported_legacy=0, entropy_sum=0.,
                      root_differences=[]) for k in ROSTER}
    grouped = {}
    for job, game in zip(jobs, games):
        r = result[job['key']]; winner = game['winner']
        score = 1. if winner == job['learner'] else 0.
        r[job['label']] += score
        cluster = grouped.setdefault((job['key'],job['replica']), {'candidate': [], 'reference': []})
        cluster[job['label']].append(score)
        if job['label'] == 'candidate':
            r['games'] += 1
            for n, v in game.get('behavior', {}).get(job['learner'], {}).items():
                if n in r: r[n] += v
    for (key, _), row in grouped.items():
        if len(row['candidate']) != 4 or len(row['reference']) != 4:
            raise ValueError('Incomplete paired root')
        result[key]['root_differences'].append(sum(row['candidate'])/4-sum(row['reference'])/4)
    for row in result.values():
        row['play_top1']=row['plays']/row['legal_play'] if row['legal_play'] else None
        row['reference_play_top1']=row['reference_plays']/row['legal_play'] if row['legal_play'] else None
        row['mean_entropy']=row['entropy_sum']/row['decisions'] if row['decisions'] else None
    write(Path(folder) / 'comparison.json', result)
    return result


def gate_decision(pure, confirmed, health, *, no_gain, search=None, search_confirm=None,
                  minimum_gain=.02,keys=None):
    """Fail before promotion; unchanged policies never become new best."""
    reasons = []
    selected = learning_keys({'learning_keys': keys}) if keys is not None else ROSTER
    for key in selected:
        h = health[key]; p = pure[key]; c = confirmed[key]
        if not h['finite'] or not h['gradient'] or h['coverage'] != 1 or not h['policy_rows']:
            reasons.append(dict(key=key, reason='learning_health'))
        if (p['legal_play'] >= 8 and p['plays'] == 0) or (
                c['legal_play'] >= 8 and c['plays'] == 0):
            reasons.append(dict(key=key, reason='no_play'))
        if (p['candidate']+p['games']*.25 < p['reference'] and
                c['candidate']+c['games']*.25 < c['reference']):
            reasons.append(dict(key=key, reason='confirmed_regression'))
    if reasons: return dict(decision='p0', reasons=reasons)
    # Each promoted team needs its own confirmed gain; another unchanged team
    # must not discard that gain. Unqualified teams retain their reference.
    improved = [k for k in selected if pure[k]['changed'] > 0 and confirmed[k]['changed'] > 0 and
                   pure[k]['candidate']-pure[k]['reference'] >= minimum_gain*pure[k]['games'] and
                   confirmed[k]['candidate']-confirmed[k]['reference'] >= minimum_gain*confirmed[k]['games']]
    qualified = [k for k in improved if search is not None and search_confirm is not None
                 and search[k]['candidate']>=search[k]['reference']
                 and search_confirm[k]['candidate']>=search_confirm[k]['reference']]
    if qualified:return dict(decision='promote',reasons=[],promoted_keys=qualified,
                              retained_keys=[k for k in ROSTER if k not in qualified])
    return dict(decision='plateau' if no_gain else 'keep', reasons=['no_confirmed_improvement'])


def select_models(candidate,reference,keys,folder):
    """Copy approved team arrays; do not average models or alter other teams."""
    from . import recovery_runtime as rt
    import shutil
    if not keys or not set(keys)<=set(ROSTER) or len(set(keys))!=len(keys):
        raise ValueError('Distinct qualified team keys required')
    folder=Path(folder)
    if folder.exists():raise ValueError('New selective-best output required')
    folder.mkdir(parents=True)
    chosen={}
    for key in ROSTER:
        source=Path(candidate if key in keys else reference)
        policy=rt.model(source,key)
        if policy.serving_deck!=rt.model(reference,key).serving_deck:
            raise ValueError('Selective candidate changed frozen build')
        for extension in ('.json','.npz'):shutil.copyfile(source/(key+extension),folder/(key+extension))
        chosen[key]=dict(source=str(source),sha256=policy.version,promoted=key in keys)
        if rt.model(folder,key).version!=policy.version:raise ValueError('Selective arrays changed')
    write(folder.parent/'selection.json',chosen)
    return folder


def numeric_check(nets, folder, rows, device):
    import numpy as np
    import torch
    from . import recovery_runtime as rt
    from .league_learning import tensors
    result = {}
    for key, net in nets.items():
        sample = rows[key][:32]
        if not sample: raise ValueError('No real policy roots for numeric check')
        numeric = rt.model(folder, key)
        with torch.no_grad():
            x,c,mask = tensors([(r['x'],r['c']) for r in sample],device)
            logits,_ = net(x,c,mask)
            h = getattr(net,'value_net',net.state_net)(x)
            actor = torch.tensor([[r.get('value_actor',1)] for r in sample], device=device,dtype=h.dtype)
            wdl = net.wdl(torch.cat((h,actor),-1)).softmax(-1).cpu().numpy()
        for i,r in enumerate(sample):
            a = logits[i,:len(r['c'])].cpu().numpy(); b = numeric.scores(r['x'],r['c'])
            np.testing.assert_allclose(a,b,atol=2e-4,rtol=2e-4)
            np.testing.assert_allclose(wdl[i],numeric.wdl(r['x'],r.get('value_actor',1)),atol=2e-4,rtol=2e-4)
            if int(a.argmax()) != int(b.argmax()): raise ValueError('Export changed action')
        result[key] = dict(rows=len(sample), policy_wdl_equal=True, actions_equal=True)
    return result


def update_batch(nets, opts, pools, jobs, games, rng, config, deadline, out):
    from . import recovery_runtime as rt
    from .episode_replay import dump_episode
    health = {}; rows_by_key = {}
    selected = learning_keys(config)
    if any(job['policies'][s][1] not in selected for job in jobs for s in job['train_sides']):
        raise ValueError('Frozen opponent cannot contribute learning rows')
    for pool in pools.values(): pool.begin_fresh_batch()
    for job, game in zip(jobs,games):
        data = dump_episode(Path(out) / 'episodes' / f"{job['id']}.json.gz",job,game)
        for key in set(job['policies'][s][1] for s in job['train_sides']):
            rows = [r for r in data['rows'] if r['key'] == key]
            pools[key].add(job['id'],dict(rows=rows))
    for key in selected:
        stats = []; pool = pools[key]
        rows_by_key[key] = [r for g in games for r in g['rows'] if r['key']==key]
        for rows,refs,sampling in pool.policy_pass_batches(rng,passes=config['policy_passes'],deadline=deadline):
            metric = rt.update(nets[key],opts[key],rows)
            stats.append(dict(metric, refs=refs, sampling=sampling))
        values = [r for job,g in zip(jobs,games) for r in g['value_rows']
                  if r['key']==key and r['side'] in job['train_sides'] and r['value_actor']==0]
        for start in range(0,len(values),128):
            if time.time() >= deadline: raise TimeoutError('Update deadline')
            rt.update(nets[key],opts[key],values[start:start+128],value_only=True)
        cov = pool.coverage_report()
        health[key] = dict(finite=bool(stats) and all(math.isfinite(m[n]) for m in stats for n in ('policy_loss','value_loss')),
                           gradient=all(any(m['policy_gradient_norms'].get(prefix,0)>0 for m in stats)
                                        for prefix in ('state_net','cand_net','score')),
                           coverage=cov['first_use_coverage'],policy_rows=cov['fresh_policy_rows'],
                           updates=len(stats),value_rows=len(values))
        write(Path(out) / f'{key}-updates.json',stats)
    return health, rows_by_key


def save_round(out, nets, opts, pools, models, rng, config, cycle, best, no_gain):
    from .recovery_checkpoint import save
    folder = Path(out) / 'recovery' / str(cycle%2)
    for key in ROSTER:
        save(folder / f'{key}.pt',nets[key],opts[key],directory=models,key=key,
             rng_state=rng.bit_generator.state,metadata=dict(cycle=cycle,config=config,best=str(best),
             no_gain=no_gain,model_folder=str(models)))
    # Pools are numeric JSON, separate from safe weights_only checkpoints.
    from .episode_replay import dump_episode
    import numpy as np
    state = {k:p.export_state() for k,p in pools.items()}
    path = folder / 'pools.json'
    path.write_text(json.dumps(state,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()))
    files = {p.name:sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    write(Path(out)/'recovery/pointer.json',dict(cycle=cycle,models=str(models),best=str(best),
          folder=str(folder),files=files,original_train_until=config['train_until'],
          hard_deadline=config['hard_deadline'],algorithm_sha256=fingerprint()))


def immutable_receipt_files(out):
    root=Path(out)
    # The parent guard changes after worker exit. A receipt must not hash
    # itself or mutable status; numerical and raw evidence remain mandatory.
    excluded={'status.json','guard.json','readiness.json','delivery-manifest.json'}
    return {p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix() not in excluded}


def validate_launch(config, *, resume=False):
    """Formal launches require actual current-source CUDA preflight, not flags."""
    from .cross_lineup import identity
    if config.get('runtime') != 'recovery' or config.get('rule_hash') != identity():
        raise ValueError('Recovery rule/runtime mismatch')
    if config.get('teacher_mode','network') not in ('network','covered_terminal','covered_expert','covered_value'):
        raise ValueError('Unknown frozen teacher mode')
    if (type(config.get('value_full_tower',False)) is not bool
            or type(config.get('value_replicas',2)) is not int or config.get('value_replicas',2) not in (2,6,12)
            or type(config.get('value_check_replicas',1)) is not int or config.get('value_check_replicas',1) not in (1,2,3)):
        raise ValueError('Bounded value calibration protocol required')
    scope_sha=None
    if config.get('teacher_mode')=='covered_value':
        from .covered_value import load_scope
        if not config.get('covered_value_source'):raise ValueError('Qualified covered-value-source required')
        load_scope(config['covered_value_source'])
        scope_sha=sha256((Path(config['covered_value_source'])/'scope.json').read_bytes()).hexdigest()
    if config.get('smoke') is True:
        replicas=config.get('preflight_replicas',1)
        if type(replicas) is not int or replicas not in (1,2,3):
            raise ValueError('Bounded preflight replica count required')
        max_seconds=7200.1 if config.get('preflight') and replicas>1 else 1800.1
        if (config.get('planned_batches',0) not in (1,2,3) or config.get('workers',0) not in range(1,17)
                or config.get('simulations') not in (2,32)
                or config.get('hard_deadline',0)-config.get('started_at',0)>max_seconds
                or config.get('train_until',0)>config.get('hard_deadline',0)
                or config.get('max_row_reuse',0)<config.get('policy_passes',1)):
            raise ValueError('Bounded recovery configuration required')
        return
    if not config.get('recovery_source'):
        raise ValueError('CUDA recovery preflight receipt required with matching recovery-source')
    if (config.get('simulations')!=32 or config.get('gumbel_candidates')!=16
            or config.get('terminal_horizon')!=(0 if config.get('teacher_mode') in ('covered_terminal','covered_expert') else 3) or config.get('rewards')!=0
            or config.get('planned_batches') not in (4,5,6)
            or not 5<=config.get('gate_roots',0)<=64
            or not 1<=config.get('search_gate_roots',0)<=8):
        raise ValueError('Formal recovery protocol mismatch')
    from .five_cross_gates import schedule_gates
    if config.get('gates')!=schedule_gates(config['planned_batches']):
        raise ValueError('Formal gate schedule must match frozen planned batches')
    path = Path(config.get('readiness_report') or 'missing-readiness-report')
    if not path.is_file(): raise ValueError('CUDA recovery preflight receipt required')
    receipt = json.loads(path.read_text())
    if (receipt.get('schema') != 'five_recovery_readiness_v1' or receipt.get('device') != 'cuda'
            or receipt.get('algorithm_sha256') != fingerprint() or receipt.get('rule_hash') != identity()
            or receipt.get('simulations') != 32 or receipt.get('workers') != config['workers']
            or receipt.get('teacher_mode','network') != config.get('teacher_mode','network')
            or receipt.get('covered_value_sha')!=scope_sha
            or not 0 <= time.time()-receipt.get('measured_at',0) < 86400
            or receipt.get('ready') is not True or set(receipt.get('health',{})) != set(ROSTER)
            or receipt.get('initial_models') != versions(config['recovery_source'])):
        raise ValueError('Recovery CUDA readiness identity/age/config mismatch')
    if config.get('teacher_mode','network')=='network':
        raise ValueError('Fixed-build network teacher lacks sampled-build rollout coverage qualification')
    required=('initial-value/value-precheck.json','cycles/1/numeric.json','cycles/1/restore-check.json',
              'cycles/1/gate.json','cycles/1/sampling/games.json','cycles/1/capacity.json')
    if not all(name in receipt.get('files',{}) for name in required):
        raise ValueError('Readiness raw evidence is incomplete')
    for relative,digest in receipt['files'].items():
        file = (path.parent/relative).resolve()
        if not file.is_relative_to(path.parent.resolve()) or sha256(file.read_bytes()).hexdigest()!=digest:
            raise ValueError('Readiness raw evidence changed')
    for h in receipt['health'].values():
        if not (h['finite'] and h['gradient'] and h['coverage']==1 and h['policy_rows']>0):
            raise ValueError('All-five CUDA gradient and coverage evidence required')
    from .recovery_runtime import value_precheck_ok,model
    initial_check=json.loads((path.parent/'initial-value/value-precheck.json').read_text())
    if not value_precheck_ok(initial_check,{k:model(config['recovery_source'],k) for k in ROSTER}):
        raise ValueError('Readiness initial value arrays mismatch')
    if (receipt.get('policy_passes')!=config['policy_passes']
            or receipt.get('learning_rate')!=config['learning_rate']
            or receipt.get('value_calibration',value_calibration_protocol({}))!=value_calibration_protocol(config)):
        raise ValueError('Readiness optimizer protocol mismatch')
    capacity=json.loads((path.parent/'cycles/1/capacity.json').read_text())
    if (capacity.get('schema')!='five_recovery_isolation_fit_v1' or capacity.get('passed') is not True
            or capacity.get('complete') is not True or capacity.get('device')!='cuda'
            or capacity.get('source_models')!=receipt['initial_models']):
        raise ValueError('CUDA whole-network isolation fit and heldout required')
    numeric=json.loads((path.parent/'cycles/1/numeric.json').read_text())
    restored=json.loads((path.parent/'cycles/1/restore-check.json').read_text())
    gate=json.loads((path.parent/'cycles/1/gate.json').read_text())
    records=json.loads((path.parent/'cycles/1/sampling/games.json').read_text())
    if (len(records)!=30*receipt.get('preflight_replicas',1) or not all(r['complete'] for r in records)
            or set(numeric)!=set(ROSTER) or set(restored)!=set(ROSTER)
            or not all(numeric[k]['actions_equal'] and numeric[k]['policy_wdl_equal']
                       and restored[k]['adam_next_step_equal'] and restored[k]['rng_equal'] for k in ROSTER)
            or gate['decision']=='p0'):
        raise ValueError('All-five numeric/restore/independent behavior evidence required')
    # Price the actual measured full batch, repeated value/gates and both
    # matrices before launching; retain 25% capacity slack and 15min delivery.
    # The short receipt has small gates; explicitly expand the actual
    # formal pure/search gate counts and the historical 60-game round.
    batches=config['planned_batches']
    if resume:
        batches=0  # restore_round enforces original remaining cycles/deadline
    # Isolation fitting and its separate heldout corpus are performed once.
    # Their cost is not silently charged as a recurring learning batch.
    # Include both candidate and possible mixed-slate value/gate checks.
    per_cycle=(30*receipt['search_seconds_per_game']
        +receipt['update_and_numeric_seconds']
        +config['gate_roots']/receipt['gate_roots']*receipt['pure_gate_seconds']
        +2*receipt['selected_value_seconds']
        +config['search_gate_roots']*160*receipt['search_seconds_per_game'])
    needed = (receipt['selected_value_seconds']+per_cycle*batches
              +30*receipt['search_seconds_per_game']*(batches//4))*1.25
    if needed > config['train_until']-time.time(): raise ValueError('Insufficient staged learning capacity')
    needed_eval = receipt['search_seconds_per_game']*25*config['matrix_games']*2*1.25
    if needed_eval > config['evaluate_until']-config['train_until']:
        raise ValueError('Insufficient frozen matrix capacity')
    if config['hard_deadline']-config['evaluate_until']<900 or config['device']!='cuda':
        raise ValueError('CUDA and 15min delivery reserve required')


def merged_reports(*reports):
    from .cross_lineup import identity
    result = dict(schema='recovery_value_precheck_v1', complete=True, heldout=True, passed=True,
                  rule_hash=identity(), by_version={})
    for report in reports:
        if not report['passed']: raise ValueError('Cannot merge a failed value check')
        for key, version in report['models'].items():
            result['by_version'][version] = dict(version=version,metrics=report['metrics'][key])
    return result


def restore_round(out, device, config):
    """Exact in-budget resume; no new deadline, seed reservation, or hidden continuation."""
    import numpy as np
    import torch
    from .episode_replay import EpisodePool
    from .recovery_checkpoint import restore
    out = Path(out); pointer=json.loads((out/'recovery/pointer.json').read_text())
    if (pointer['algorithm_sha256']!=fingerprint() or pointer['hard_deadline']!=config['hard_deadline']
            or pointer['original_train_until']!=config['train_until'] or time.time()>=config['train_until']):
        raise ValueError('Resume source/deadline mismatch or learning expired')
    terminal=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
    if (pointer['cycle']>=config.get('planned_batches',float('inf'))
            or terminal.get('stop_reason') in ('p0','plateau','candidate_value_failed','isolation_fit_failed')
            or terminal.get('phase') in ('complete','evaluation_incomplete','preflight_complete','preflight_failed')):
        raise ValueError('Learning already stopped; refuse to reuse confirmation seeds or restart optimization')
    if (terminal.get('cycle',pointer['cycle'])>pointer['cycle']
            and terminal.get('phase') in ('candidate_value_check','independent_gate',
                                          'selected_value_check','selected_independent_gate')):
        raise ValueError('Independent seeds consumed in an uncommitted cycle; optimization resume is forbidden')
    folder=Path(pointer['folder'])
    if not folder.resolve().is_relative_to(out.resolve()):
        raise ValueError('Recovery folder must belong to this output')
    for name,digest in pointer['files'].items():
        if sha256((folder/name).read_bytes()).hexdigest()!=digest: raise ValueError('Recovery file changed')
    nets={};opts={}; saved=None
    for key in ROSTER:
        nets[key],opts[key],data=restore(folder/f'{key}.pt',directory=pointer['models'],key=key,device=device)
        if data['metadata']['config']!=config: raise ValueError('Resume config changed')
        saved=data
    pools={k:EpisodePool.from_state(v) for k,v in json.loads((folder/'pools.json').read_text()).items()}
    # JSON replay arrays must regain numeric type for tensors/update.
    for pool in pools.values():
        for episode in pool.episodes:
            for row in episode['rows']:
                for name in ('x','c','pi'):
                    if name in row: row[name]=np.asarray(row[name],dtype=np.float32)
    rng=np.random.default_rng();rng.bit_generator.state=saved['rng_state']
    torch.set_rng_state(saved['torch_rng'])
    if device=='cuda' and saved.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return nets,opts,pools,rng,pointer,saved['metadata']['no_gain']


def evaluate_frozen(pool, out, selected, initial, decks, report, config):
    from .full_cycle_experiment import evaluate_job
    from .new_model_report import write_matrix
    signatures=versions(selected); plan=matrix_schedule(config['matrix_games']); all_complete=True
    for mode in ('argmax','search'):
        folder=Path(out)/mode
        for name in ('raw','replays'): (folder/'evaluation'/name).mkdir(parents=True,exist_ok=True)
        jobs=[]
        for i,item in enumerate(plan):
            jobs.append(dict(item,id=f'{mode}-{i}',runtime='recovery',
                cell=('mirror/' if item['left']==item['right'] else 'cross/')+item['row']+'/'+item['column'],
                seed=config['seeds']['confirmation']+i,deadline=config['evaluate_until'],
                decks={'a':decks[item['left']],'b':decks[item['right']]},
                policies={'a':(str(selected),item['left']),'b':(str(selected),item['right'])},
                output=str(folder/'evaluation'),replay=item['replica']==0,
                search_sides=['a','b'] if mode=='search' else [],simulations=config['simulations'],
                value_precheck=report,q_scale='natural_wdl',gumbel_candidates=16,
                terminal_horizon=0 if config.get('teacher_mode') in ('covered_terminal','covered_expert') else 3,
                teacher_mode=config.get('teacher_mode','network'),covered_value_source=config.get('covered_value_source')))
        write(folder/'planned-jobs.json',jobs)
        results=[]
        for offset in range(0,len(jobs),config['workers']):
            if time.time()>=config['evaluate_until']: break
            results.extend(pool.map(evaluate_job,jobs[offset:offset+config['workers']]))
        write(folder/'evaluation/index.json',results)
        write_matrix(folder,results,ROSTER,selected,dict(config,smoke=False,initial_models=str(initial),inference=mode))
        cells=coverage(results,plan);write(folder/'coverage.json',cells)
        all_complete &= all(c['complete'] for c in cells)
    if versions(selected)!=signatures: raise ValueError('Weights changed during frozen evaluation')
    return bool(all_complete)


def run(out, config, *, resume=False):
    """Complete fixed-build five-team learning path; supplemental release eval stays explicit."""
    validate_launch(config,resume=resume)
    import numpy as np
    import torch
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from . import recovery_runtime as rt
    from .episode_replay import EpisodePool
    from .recovery_checkpoint import save,restore
    from .offline_sources import snapshot
    from scripts.prepare_duel_v2_offline_readiness import EXTRA
    out=Path(out);started=time.time();torch.set_num_threads(1)
    if config['device']=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable; no fallback')
    cycle=0;no_gain=False;stop_reason=None;health={};search_wall=0.;search_games=0;pure_gate_wall=0.;value_wall=0.;update_wall=0.
    frozen_algorithm=fingerprint()
    if resume:
        nets,opts,pools,rng,pointer,no_gain=restore_round(out,config['device'],config)
        cycle=pointer['cycle'];current=Path(pointer['models']);best=Path(pointer['best'])
        decks={k:rt.model(current,k).serving_deck for k in ROSTER}
        origins={k:rt.model(current,k).manifest['origin'] for k in ROSTER}
        report=json.loads((current.parent/'value-precheck.json').read_text())
        best_report=json.loads((best.parent/'value-precheck.json').read_text())
        initial=Path(json.loads((out/'initial-pointer.json').read_text())['models'])
    else:
        nets,decks,origins=initialize(out/'uncalibrated-initial',config['hidden'],config['device'],
                                    config['seeds']['foundation'],config.get('recovery_source'))
        current=out/'uncalibrated-initial';best=None
        rng=np.random.default_rng(config['seeds']['foundation'])
        opts={k:torch.optim.Adam(net.parameters(),lr=config['learning_rate']) for k,net in nets.items()}
        pools={k:EpisodePool(limit=64,max_row_reuse=config['max_row_reuse']) for k in ROSTER}
        snapshot(Path(__file__).resolve().parents[4],out/'source',EXTRA)
    def status(phase,**extra):
        write(out/'status.json',dict(phase=phase,cycle=cycle,latest=str(current),best=str(best),
            hard_deadline=config['hard_deadline'],runtime='recovery',roster=list(ROSTER),
            **dict(dict(full_matrix=False,complete_report=False,quality_approved=False,website_approval=False),**extra)))
        print(phase,cycle,extra,flush=True)
    with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as pool:
        if not resume:
            status('value_preflight')
            value_started=time.time()
            current,report=warm_value(pool,nets,decks,origins,current,out/'initial-value',config,0,config['train_until'])
            value_wall=max(value_wall,time.time()-value_started)
            if not report['passed']:
                status('blocked_value_precheck',metrics=report['metrics']);return
            best=current;initial=current;best_report=report
            write(out/'initial-pointer.json',dict(models=str(initial),sha256=versions(initial)))
        while cycle<config['planned_batches'] and time.time()<config['train_until']-30:
            if fingerprint()!=frozen_algorithm: raise ValueError('Frozen training source changed')
            cycle+=1;folder=out/'cycles'/str(cycle);status('searching')
            historical=cycle%4==0
            combined=merged_reports(report,best_report) if historical else report
            jobs=jobs_for(current,decks,config['seeds']['foundation']+cycle*10000,
                          config['train_until']-20,pure=False,cycle=cycle,report=combined,
                          simulations=config['simulations'],historical=best if historical else None,
                          teacher_mode=config.get('teacher_mode','network'),covered_value_source=config.get('covered_value_source'))
            if config.get('preflight'):
                for replica in range(1,config.get('preflight_replicas',1)):
                    jobs+=jobs_for(current,decks,config['seeds']['foundation']+cycle*10000+replica*100,
                        config['train_until']-20,pure=False,cycle=replica+1000,report=report,
                        simulations=config['simulations'],teacher_mode=config.get('teacher_mode','network'),
                        covered_value_source=config.get('covered_value_source'))
            start=time.time();games=collect(pool,jobs,folder/'sampling',learning=True)
            search_wall+=time.time()-start;search_games+=len(games)
            if config.get('preflight'):
                from .recovery_capacity import fit
                status('isolation_fit')
                # Same opponents and initiatives with genuinely new paired
                # seeds. Capacity heldout never enters update_batch/pools.
                held_jobs=jobs_for(current,decks,config['seeds']['probes']+10000,config['train_until']-20,
                                   pure=False,cycle=100,report=report,simulations=config['simulations'],
                                   teacher_mode=config.get('teacher_mode','network'),covered_value_source=config.get('covered_value_source'))
                for replica in range(1,config.get('preflight_replicas',1)):
                    held_jobs+=jobs_for(current,decks,config['seeds']['probes']+10000+replica*100,
                        config['train_until']-20,pure=False,cycle=replica+2000,report=report,
                        simulations=config['simulations'],teacher_mode=config.get('teacher_mode','network'),
                        covered_value_source=config.get('covered_value_source'))
                for job in held_jobs:
                    job.update(training=False,exploration=True,data_partition='capacity_heldout')
                held_games=collect(pool,held_jobs,folder/'capacity-heldout',learning=False,require_search=True)
                capacity=fit(current,jobs,games,config,min(config['train_until'],time.time()+1200),
                             held_jobs=held_jobs,held_games=held_games)
                write(folder/'capacity.json',capacity)
                if not capacity['passed']:
                    stop_reason='isolation_fit_failed'
                    save_round(out,nets,opts,pools,current,rng,config,cycle,best,no_gain)
                    break
            status('updating')
            update_started=time.time()
            health,rows=update_batch(nets,opts,pools,jobs,games,rng,config,config['train_until'],folder)
            for key in ROSTER: rt.export(nets[key],folder/'updated',key,decks[key],dict(origins[key],cycle=cycle))
            numeric=numeric_check(nets,folder/'updated',rows,config['device']);write(folder/'numeric.json',numeric)
            # Restore tests on copies also verify the next actual Adam step;
            # candidates remain unchanged. CUDA RNG is saved by checkpoint.
            checks={}
            for key in ROSTER:
                path=folder/'restore'/f'{key}.pt'
                save(path,nets[key],opts[key],directory=folder/'updated',key=key,rng_state=rng.bit_generator.state,metadata={'cycle':cycle})
                copy_net,copy_opt,data=restore(path,directory=folder/'updated',key=key,device=config['device'])
                control=deepcopy(nets[key]);control_opt=torch.optim.Adam(control.parameters())
                control_opt.load_state_dict(deepcopy(opts[key].state_dict()))
                sample=rows[key][:32];rt.update(control,control_opt,sample);rt.update(copy_net,copy_opt,sample)
                if any(not torch.equal(t,copy_net.state_dict()[n]) for n,t in control.state_dict().items()):
                    raise ValueError('Next Adam step did not restore')
                checks[key]=dict(adam_next_step_equal=True,rng_equal=data['rng_state']==rng.bit_generator.state)
            write(folder/'restore-check.json',checks)
            update_wall=max(update_wall,time.time()-update_started)
            if any(not h['finite'] or not h['gradient'] or h['coverage']!=1 or not h['policy_rows'] for h in health.values()):
                write(folder/'gate.json',dict(decision='p0',health=health,reason='learning_health'))
                stop_reason='p0';current=best;report=best_report
                nets={k:rt.restore(best,k,config['device'])[0] for k in ROSTER}
                opts={k:torch.optim.Adam(n.parameters(),lr=config['learning_rate']*.1) for k,n in nets.items()}
                pools={k:EpisodePool(limit=64,max_row_reuse=config['max_row_reuse']) for k in ROSTER}
                save_round(out,nets,opts,pools,current,rng,config,cycle,best,no_gain)
                break
            # No candidate can search again with its starting-value receipt.
            status('candidate_value_check')
            value_started=time.time()
            candidate,new_report=warm_value(pool,nets,decks,origins,folder/'updated',folder/'value',
                                            config,cycle,config['train_until'])
            value_wall=max(value_wall,time.time()-value_started)
            current=candidate
            for key in ROSTER:
                for name,parameter in nets[key].named_parameters():
                    if name.startswith('wdl.') or (config.get('value_full_tower') and name.startswith('value_net.')):
                        opts[key].state.pop(parameter,None)
            if not new_report['passed']:
                stop_reason='candidate_value_failed';current=best;report=best_report
                nets={k:rt.restore(best,k,config['device'])[0] for k in ROSTER}
                opts={k:torch.optim.Adam(n.parameters(),lr=config['learning_rate']) for k,n in nets.items()}
                pools={k:EpisodePool(limit=64,max_row_reuse=config['max_row_reuse']) for k in ROSTER}
                save_round(out,nets,opts,pools,current,rng,config,cycle,best,no_gain)
                break
            report=new_report
            status('independent_gate')
            deadline=min(config['train_until'],time.time()+config['gate_seconds'])
            pure_gate_started=time.time()
            pure=comparison(pool,compare_jobs(candidate,best,decks,config['seeds']['selection']+cycle*10000+2000,
                            deadline,config['gate_roots']),folder/'pure-screen')
            confirmed=comparison(pool,compare_jobs(candidate,best,decks,config['seeds']['selection']+cycle*10000+3000,
                            deadline,config['gate_roots']),folder/'pure-confirm')
            pure_gate_wall+=time.time()-pure_gate_started
            designated=config['smoke'] or cycle in (config['gates']['mid'],config['gates']['late'])
            decision=gate_decision(pure,confirmed,health,no_gain=no_gain and designated)
            search_screen=search_confirm=None
            if not config['smoke'] and decision['decision']!='p0':
                combined=merged_reports(report,best_report)
                search_screen=comparison(pool,compare_jobs(candidate,best,decks,
                    config['seeds']['selection']+cycle*10000+4000,deadline,config['search_gate_roots'],search=True,
                    reports=combined,simulations=32,teacher_mode=config.get('teacher_mode','network'),covered_value_source=config.get('covered_value_source')),folder/'search-screen')
                search_confirm=comparison(pool,compare_jobs(candidate,best,decks,
                    config['seeds']['selection']+cycle*10000+5000,deadline,config['search_gate_roots'],search=True,
                    reports=combined,simulations=32,teacher_mode=config.get('teacher_mode','network'),covered_value_source=config.get('covered_value_source')),folder/'search-confirm')
                decision=gate_decision(pure,confirmed,health,no_gain=no_gain and designated,search=search_screen,search_confirm=search_confirm)
            proposal=dict(decision)
            if proposal['decision']=='promote':proposal['decision']='pending_mixed_validation'
            write(folder/'gate.json',dict(**proposal,pure=pure,confirmed=confirmed,health=health,
                  independent_selection=True,strength_approved=False))
            if decision['decision']=='promote':
                status('selected_value_check')
                selected=select_models(candidate,best,decision['promoted_keys'],folder/'selected-uncalibrated'/'models')
                selected_nets={k:rt.restore(selected,k,config['device'])[0] for k in ROSTER}
                selected_origins={k:rt.model(selected,k).manifest['origin'] for k in ROSTER}
                selected,selected_report=warm_value(pool,selected_nets,decks,selected_origins,selected,
                    folder/'selected',config,cycle,config['train_until'],seed_offset=6000)
                if selected_report['passed']:
                    status('selected_independent_gate')
                    selected_reports=merged_reports(selected_report,best_report)
                    selected_checks=[]
                    for offset,label in ((8000,'screen'),(9000,'confirm')):
                        selected_checks.append(comparison(pool,compare_jobs(selected,best,decks,
                            config['seeds']['selection']+cycle*10000+offset,config['train_until'],
                            config['search_gate_roots'],search=True,reports=selected_reports,
                            simulations=config['simulations'],teacher_mode=config.get('teacher_mode','network'),covered_value_source=config.get('covered_value_source')),
                            folder/('selected-search-'+label)))
                    safe=all(check[k]['candidate']>=check[k]['reference'] for check in selected_checks for k in ROSTER)
                    if safe:best=selected;best_report=selected_report;no_gain=False
                    else:decision.update(decision='keep',reasons=['selected_search_regression'],promoted_keys=[],retained_keys=list(ROSTER))
                else:
                    decision.update(decision='keep',reasons=['selected_value_failed'],promoted_keys=[],retained_keys=list(ROSTER))
                if decision['decision']!='promote' and designated:no_gain=True
                write(folder/'gate.json',dict(**decision,pure=pure,confirmed=confirmed,health=health,
                    independent_selection=True,strength_approved=False))
            elif decision['decision'] in ('p0','plateau'):
                stop_reason=decision['decision']
                # Actual in-memory rollback as well as exported pointer rollback.
                nets={k:rt.restore(best,k,config['device'])[0] for k in ROSTER}
                opts={k:torch.optim.Adam(n.parameters(),lr=config['learning_rate']*.1) for k,n in nets.items()}
                pools={k:EpisodePool(limit=64,max_row_reuse=config['max_row_reuse']) for k in ROSTER}
                current=best;report=best_report
            elif designated: no_gain=True
            save_round(out,nets,opts,pools,current,rng,config,cycle,best,no_gain)
            if stop_reason: break
        if not cycle: raise ValueError('No complete learning batch before deadline')
        if stop_reason is None:
            stop_reason='planned_batches' if cycle==config['planned_batches'] else 'learning_deadline'
        # A preflight is not formal quality approval. Only complete CUDA32
        # records can create readiness for a separate newly seeded trial.
        if fingerprint()!=frozen_algorithm: raise ValueError('Frozen training source changed')
        if config.get('preflight'):
            ready=stop_reason=='planned_batches' and config.get('teacher_mode','network')!='network' and bool(health) and all(h['finite'] and h['gradient'] and h['coverage']==1 for h in health.values())
            receipt=dict(schema='five_recovery_readiness_v1',device=config['device'],ready=ready,
                rule_hash=config['rule_hash'],algorithm_sha256=fingerprint(),simulations=config['simulations'],
                teacher_mode=config.get('teacher_mode','network'),
                covered_value_sha=sha256((Path(config['covered_value_source'])/'scope.json').read_bytes()).hexdigest() if config.get('covered_value_source') else None,
                selected_value_seconds=value_wall,
                preflight_replicas=config.get('preflight_replicas',1),
                update_and_numeric_seconds=update_wall,
                workers=config['workers'],health=health,measured_at=time.time(),initial_models=versions(initial),
                initial_folder=str(initial),cycle_seconds=time.time()-started,
                search_seconds_per_game=search_wall/max(1,search_games),
                gate_roots=config['gate_roots'],pure_gate_seconds=pure_gate_wall,
                policy_passes=config['policy_passes'],learning_rate=config['learning_rate'],quality_approved=False,
                value_calibration=value_calibration_protocol(config),
                files=immutable_receipt_files(out))
            write(out/'readiness.json',receipt)
            status('preflight_complete' if ready else 'preflight_failed',stop_reason=stop_reason)
            return
        status('evaluating',stop_reason=stop_reason or 'planned_batches')
        complete=evaluate_frozen(pool,out,best,initial,decks,best_report,config)
        status('complete' if complete else 'evaluation_incomplete',full_matrix=complete,
               stop_reason=stop_reason or 'planned_batches',selected=str(best),
               retained_baseline=versions(best)==versions(initial))
    if fingerprint()!=frozen_algorithm: raise ValueError('Frozen training source changed during evaluation')
    write(out/'delivery-manifest.json',{p.relative_to(out).as_posix():sha256(p.read_bytes()).hexdigest()
        for p in out.rglob('*') if p.is_file() and p.name!='delivery-manifest.json'})
