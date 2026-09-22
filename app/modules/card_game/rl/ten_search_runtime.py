"""Explicit ten-character adapter for the shared information-set search engine."""
from functools import lru_cache
from pathlib import Path
from .fixed_lineup import FixedModel, encode, SEATS, CARD_IDS
from .league_rollout import determinize as sample
from ..engine.duel_v2 import observe

@lru_cache(maxsize=24)
def _load(directory, key, modified):
    return FixedModel(directory, key)

def model(directory, key):
    return _load(str(directory), key, (Path(directory)/(key+'.json')).stat().st_mtime_ns)

def decision(state, side):
    view = observe(state, side, include_previews=False)
    actions = [e['action'] for e in view['legal_actions'] if e['action']['type'] != 'concede']
    if not actions:
        raise ValueError('No legal non-concede action')
    return actions, encode(view, actions)

def determinize(state, side, seed):
    return sample(state, side, seed, encoder=encode, seats=SEATS, cards=CARD_IDS)


def passive_count(state, side):
    return int(state['sides'][side]['characters'].get('zhenhong',{}).get('surplus_passive_triggers',0))


def load_numeric(directory, key, device):
    import torch
    from .gpu_duel.ppo import CompactScorer
    from .fixed_lineup import all_features, CAND_DIM
    policy = FixedModel(directory,key)
    net=CompactScorer(len(all_features()),CAND_DIM,policy.hidden)
    net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in policy.weights.items()})
    return net.to(device),dict(source_sha256=policy.version,source_deck=key,source_kind='validated_ten_numeric',setup_reward_updates=(policy.manifest.get('origin') or {}).get('setup_reward_updates',0))


def export_model(net, output, key, build, *, origin=None):
    from .fixed_lineup import save_weights
    save_weights({k:v.detach().cpu().numpy() for k,v in net.state_dict().items()},output,key,build,origin)


def probe_results(directory,key,probes):
    policy=FixedModel(directory,key)
    return [int(policy.scores(*decision(p['state'],p['side'])[1]).argmax()) in p['targets'] for p in probes[key]]


def probes(builds, seed_base):
    from ..engine.duel_v2 import new_game, apply_action
    from copy import deepcopy
    result={}
    for ki,(key,build) in enumerate(builds.items()):
        result[key]=[]
        for first in ('a','b'):
            for i in range(6):
                s=new_game(seed=seed_base+ki*100+i,first_side=first,skip_mulligan=True,decks={'a':build,'b':build})
                foe='b' if first=='a' else 'a'
                # Synthetic visible lethal probes: defender has no hidden responses.
                team=s['sides'][foe];team['deck']+=team['hand'];team['hand']=[];team['hp']=1;team['shield']=0
                actions=decision(s,first)[0]
                targets=[j for j,a in enumerate(actions) if apply_action(deepcopy(s),first,a).get('winner')==first]
                if not targets:raise ValueError('No lethal in probe')
                result[key].append(dict(state=s,side=first,targets=targets))
    return result


def migrate_source(source, output, builds):
    """Named-column research migration; never inherits serving or optimizer approval."""
    import json, hashlib
    import numpy as np
    from .fixed_lineup import all_features, candidate_names, tensor_shapes, save_weights
    from .build_acceptance import check_source
    check_source(source)
    for key,build in builds.items():
        source_key=key if (Path(source)/(key+'.json')).exists() else 'starter'
        m=json.loads((Path(source)/(source_key+'.json')).read_text(encoding='utf-8'))
        p=Path(source)/(source_key+'.npz')
        from .league_schema import build_hash
        if m.get('schema')!='fixed_ten_v1' or m.get('deck')!=source_key or build_hash(m['build'])!=m.get('build_sha256') or hashlib.sha256(p.read_bytes()).hexdigest()!=m['sha256']:
            raise ValueError('Invalid source identity')
        before=m['features'];after=all_features()
        if len(set(before))!=len(before) or not set(before)<=set(after) or m['candidates']!=candidate_names():
            raise ValueError('Unsupported feature migration')
        with np.load(p,allow_pickle=False) as z:w={n:z[n].copy() for n in tensor_shapes(m['hidden'])}
        for n,shape in tensor_shapes(m['hidden']).items():
            expected=(m['hidden'],len(before)) if n=='state_net.0.weight' else shape
            if w[n].shape!=expected or not np.isfinite(w[n]).all():raise ValueError('Invalid source tensor')
        old=w['state_net.0.weight'];w['state_net.0.weight']=np.zeros((m['hidden'],len(after)),np.float32)
        columns={name:i for i,name in enumerate(after)}
        for i,name in enumerate(before):w['state_net.0.weight'][:,columns[name]]=old[:,i]
        save_weights(w,output,key,build,dict(kind='current_rule_research_warm_start',source_sha256=m['sha256'],source_rule_hash=m['rule_hash'],source_key=source_key,optimizer_reset=True,setup_reward_updates=(m.get('origin') or {}).get('setup_reward_updates',0) if key==source_key=='zhenhong' else 0,new_features_zero_initialized=sorted(set(after)-set(before))))
        FixedModel(output,key)
