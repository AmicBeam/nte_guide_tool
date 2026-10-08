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

from .seven_schedule import ROSTER, learning_schedule, matrix_schedule, coverage


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
    from .seven_checkpoint import algorithm_identity
    return sha256((algorithm_identity()+Path(__file__).read_text()).encode()).hexdigest()


def play(job):
    from .seven_runtime import search_game
    return search_game(job)


def versions(folder):
    from .seven_runtime import model
    return {key: model(folder, key).version for key in ROSTER}


def jobs_for(models, decks, base, deadline, *, pure=True, cycle=0, mirrors=True, report=None,
             simulations=32, historical=None,teacher_mode='network',covered_value_source=None,
             keys=None):
    from .build_acceptance import canonical_build
    decks={key:canonical_build(build) for key,build in decks.items()}
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
        from .seven_value_metrics import compute
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
    from . import seven_runtime as rt
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
    report = dict(schema='seven_value_precheck_v1', complete=True, heldout=True,
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
    from . import seven_runtime as rt
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
    from . import seven_runtime as rt
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
    from . import seven_runtime as rt
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
    from .seven_checkpoint import save
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




def merged_reports(*reports):
    from .seven_lineup import identity
    result=dict(schema='seven_value_precheck_v1',complete=True,heldout=True,passed=True,rule_hash=identity(),by_version={})
    for report in reports:
        if not report['passed']:raise ValueError('Cannot merge failed value check')
        for key,version in report['models'].items():result['by_version'][version]=dict(version=version,metrics=report['metrics'][key])
    return result
