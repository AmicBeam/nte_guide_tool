"""Bounded frozen-weight SoA label capture; no optimizer or formal launch gate.

The existing device training tape supplies both viewers and full policy targets.
Only naturally completed, replay-verified games may export numerical labels.
This module does not assert driver qualification or authorize model updates.
"""


def training_tape_capacity(compact, *, roots, decisions, observation_dim, max_bytes):
    from .device_training_tape import DeviceTrainingTape
    for value in (roots,decisions,observation_dim,max_bytes):
        if type(value) is not int or value<1:
            raise ValueError('Positive bounded training archive dimensions required')
    records=roots*decisions
    estimate=DeviceTrainingTape._estimate_bytes
    lower,upper=compact.max_candidates,records*compact.max_candidates
    # The prototype is owned by the collector too, not a free layout object.
    tape_budget=max_bytes-compact.allocated_bytes
    budget=tape_budget
    if estimate(compact,roots,records+1,lower+1,observation_dim)>budget:
        raise MemoryError('Full observer records and one complete legal prefix exceed the training byte budget')
    while lower<upper:
        middle=(lower+upper+1)//2
        if estimate(compact,roots,records+1,middle+1,observation_dim)<=budget:
            lower=middle
        else:
            upper=middle-1
    return dict(max_records=records,max_legal_entries=lower,max_bytes=tape_budget,
                estimated_bytes=estimate(compact,roots,records+1,lower+1,observation_dim))


def create_training_tape(rules, capacity, *, decisions, max_bytes):
    from .device_training_tape import DeviceTrainingTape
    from .device_soa_rules import DeviceSoARules
    from .device_tiled_soa_rules import _empty_tokens
    from .compact_device_candidates import CompactDeviceCandidates
    from ..batched_duel.gpu_seven_observation import OBS_DIM
    if not isinstance(rules,DeviceSoARules) or rules.device.type!='cuda':
        raise TypeError('Actual native CUDA SoA rules required for frozen collection')
    compact=CompactDeviceCandidates(_empty_tokens(rules.t,rules.rows,rules.max_actions,rules.device,
        integer_dtype=getattr(rules,'candidate_integer_dtype',rules.t.int64)),
        scratch_rows=capacity.inference_tile,device=rules.device,template_table=rules.templates)
    dimensions=training_tape_capacity(compact,roots=rules.rows,decisions=decisions,
        observation_dim=OBS_DIM,max_bytes=max_bytes)
    dimensions.pop('estimated_bytes')
    tape=DeviceTrainingTape(compact,rules,rows=rules.rows,observation_dim=OBS_DIM,
                            device=rules.device,**dimensions)
    tape.collection_allocated_bytes=tape.allocated_bytes+compact.allocated_bytes
    if tape.collection_allocated_bytes>max_bytes:
        raise MemoryError('Training tape and owned prototype exceed the collection byte budget')
    return tape


def completed_rows_after_batch(tape, *, verified, natural_games, roots):
    if type(roots) is not int or roots<1 or type(natural_games) is not int or not 0<=natural_games<=roots:
        raise ValueError('Exact natural-game and root counts required')
    if verified is not True:
        raise ValueError('Independent physical replay proof required before label export')
    policy=tape.numerical_rows_after_batch()
    values=tape.value_rows_after_batch()
    # Truncated/failed trajectories have no terminal label. Reject the entire
    # acceptance batch rather than training on whichever game happened to finish.
    if natural_games!=roots:
        if policy or values:
            raise ValueError('Partial natural cohort cannot grant frozen collection acceptance')
        return dict(complete=False,policy=[],values=[],optimizer_steps=0)
    if not policy or len(values)!=2*len(policy):
        raise ValueError('Every physical decision requires a full policy and both terminal-WDL viewers')
    coordinates=set()
    for row in policy:
        key=(row['owner'],row['step'])
        if key in coordinates or not row['learning_eligible'] or row['encode_error']:
            raise ValueError('Duplicate, ineligible or unencoded GPU policy row')
        coordinates.add(key)
    value_coordinates={(r['owner'],r['step'],r['actor']) for r in values}
    if len(value_coordinates)!=len(values) or value_coordinates!={(*key,side) for key in coordinates for side in (0,1)}:
        raise ValueError('Missing or duplicate terminal-WDL viewer')
    return dict(complete=True,policy=policy,values=values,optimizer_steps=0)


def write_verified_episodes(dataset, *, initial, final, records, jobs, versions, policy_reference, output,deadline=None):
    """Postbatch CPU oracle checks GPU labels, then writes the existing trainer format."""
    import hashlib
    import json
    import time
    from pathlib import Path
    import numpy as np
    from ...engine.duel_v2 import observe,apply_action
    from ..league_rollout import clean
    from ..seven_lineup import encode
    from ..episode_replay import dump_episode
    if not dataset['complete']:
        return dict(complete=False,policy_rows=0,value_rows=0,optimizer_steps=0)
    policies={(r['owner'],r['step']):r for r in dataset['policy']}
    values={(r['owner'],r['step'],r['actor']):r for r in dataset['values']}
    raw={(r['game'],r['step']):r for r in records}
    if len(raw)!=len(records) or set(policies)!=set(raw) or set(raw)!=set(policy_reference):
        raise ValueError('GPU numerical, policy-audit and raw-action coordinates differ')
    states=list(initial);games=[dict(rows=[],value_rows=[],episode_actions=[]) for _ in jobs]
    for record in records:
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError('Frozen collection proof budget exhausted')
        owner,step=record['game'],record['step'];row=policies[owner,step]
        state=states[owner];actor=record['actor'];actor_id=int(actor=='b')
        view=observe(state,actor,include_previews=False)
        legal=[v['action'] for v in view['legal_actions'] if v['action']['type']!='concede']
        x,c=encode(view,legal)
        np.testing.assert_array_equal(row['x'],x);np.testing.assert_array_equal(row['c'],c)
        reference=policy_reference[owner,step]
        np.testing.assert_array_equal(row['pi'],reference['pi'])
        np.testing.assert_array_equal(row['visits'],reference['visits'])
        if (row['actor']!=actor_id or row['selected']!=record['chosen']
                or row['version']!=state['version'] or row['job_seed']!=jobs[owner]['seed']
                or row['requested']!=record['requested'] or row['completed']!=record['completed']):
            raise ValueError('GPU policy identity, action or simulation budget changed')
        winner=final[owner].get('winner')
        expected_z=0 if winner not in ('a','b') else (1 if winner==actor else -1)
        if row['z']!=expected_z:
            raise ValueError('GPU acting-viewer WDL target changed')
        game=games[owner]
        game['rows'].append(dict(x=row['x'],c=row['c'],pi=row['pi'],visit_pi=row['visit_pi'],
            selected=row['selected'],z=row['z'],side=actor,key=jobs[owner]['keys'][actor],step=step,
            phase=view['phase'],value_actor=1,search_budget=row['completed'],
            search_algorithm='gumbel',policy_source='gumbel_q_natural_wdl',teacher_mode='network',q_scale='natural_wdl'))
        for side in ('a','b'):
            value=values[owner,step,int(side=='b')]
            vx,_=encode(view if side==actor else observe(state,side,include_previews=False),[])
            np.testing.assert_array_equal(value['x'],vx)
            vz=0 if winner not in ('a','b') else (1 if winner==side else -1)
            if value['value_actor']!=int(side==actor) or value['z']!=vz:
                raise ValueError('GPU fixed-viewer WDL context or target changed')
            game['value_rows'].append(dict(x=value['x'],z=value['z'],side=side,
                key=jobs[owner]['keys'][side],step=step,value_actor=value['value_actor']))
        game['episode_actions'].append(dict(side=actor,action=record['raw_action']))
        states[owner]=clean(apply_action(state,actor,record['raw_action']))
    artifacts=[];output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for owner,(job,game) in enumerate(zip(jobs,games)):
        if states[owner]['phase']!='finished' or states[owner].get('winner')!=final[owner].get('winner'):
            raise ValueError('Training export requires the same natural terminal')
        count=len(game['rows']);keys=job['keys']
        game.update(complete=True,winner=final[owner].get('winner'),decisions=count,searched=count,
            model_versions={side:versions[keys[side]] for side in ('a','b')})
        bound=dict(id='soa-frozen-'+str(owner),seed=job['seed'],first=job['first_side'],
                   left=keys['a'],right=keys['b'],train_sides=['a','b'])
        path=output/(bound['id']+'.json.gz');dump_episode(path,bound,game)
        artifacts.append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              policy_rows=count,value_rows=2*count))
    receipt=dict(complete=True,policy_rows=len(policies),value_rows=len(values),episodes=artifacts,
                 optimizer_steps=0,formal_training_qualified=False)
    (output/'manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    return receipt
