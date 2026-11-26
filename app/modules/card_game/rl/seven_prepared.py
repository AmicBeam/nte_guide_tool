"""Explicit migration of locked preparation arrays; never inherit readiness/quality."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from . import seven_runtime as rt, seven_lineup as layout, seven_grounded as grounded
from .build_acceptance import build_identity


def verify_preparation_domain(previous_source,current_source):
    old=Path(previous_source);new=Path(current_source)
    before=json.loads((old/'source-manifest.json').read_text(encoding='utf8'))
    after=json.loads((new/'source-manifest.json').read_text(encoding='utf8'))
    def relevant(name):
        return (name.startswith(('app/modules/card_game/engine/duel_v2/','app/modules/card_game/content/duel_v2/'))
                or name=='app/modules/card_game/engine/events.py'
                or name in ('app/modules/card_game/rl/league_observation.py','app/modules/card_game/rl/league_schema.py',
                            'app/modules/card_game/rl/seven_lineup.py','app/modules/card_game/rl/seven_grounded.py',
                            'app/modules/card_game/rl/seven_sampler.py','app/modules/card_game/rl/policies.py'))
    selected={n:e for n,e in before.items() if relevant(n)}
    if not selected or selected!={n:e for n,e in after.items() if relevant(n)}:
        raise ValueError('Prepared rule/encoding/sampling domain changed; fresh preparation required')
    for name,entry in selected.items():
        if any(sha256((root/name).read_bytes()).hexdigest()!=entry['sha256'] for root in (old,new)):
            raise ValueError('Prepared domain source changed: '+name)
    return dict(files=len(selected),sha256=sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest())


def migrate_prepared(source,key,output,*,previous_source,current_source,expected_sha256,expected_manifest_sha256,build,device='cuda'):
    import torch
    root=Path(source);manifest=root/(key+'.json');path=root/(key+'.npz')
    domain=verify_preparation_domain(previous_source,current_source)
    m=json.loads(manifest.read_text(encoding='utf8'))
    if (sha256(manifest.read_bytes()).hexdigest()!=expected_manifest_sha256
            or sha256(path.read_bytes()).hexdigest()!=expected_sha256 or m.get('sha256')!=expected_sha256
            or m.get('schema')!='cross_seven_decoupled_residual_wdl_v1' or m.get('deck')!=key
            or m.get('architecture_sha256')!=sha256((Path(previous_source)/'app/modules/card_game/rl/seven_policy.py').read_bytes()).hexdigest()
            or m.get('features')!=layout.all_features() or m.get('candidates')!=layout.candidate_names()
            or m.get('grounded_features')!=grounded.transformed_names()
            or m.get('automatic_serving_approval') is not False or m.get('reward_mode')!='terminal_wdl_only'
            or build_identity(m['build'])!=build_identity(build)):
        raise ValueError('Locked preparation numeric identity mismatch')
    net=rt.create_network(key,m['hidden'],device,legacy=bool(m.get('source_identity')))
    expected=net.state_dict()
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!=set(expected):raise ValueError('Prepared tensor set mismatch')
        arrays={n:data[n].copy() for n in data.files}
    if any(w.shape!=tuple(expected[n].shape) or not np.isfinite(w).all() for n,w in arrays.items()):
        raise ValueError('Prepared tensor shape/nonfinite mismatch')
    net.load_state_dict({n:torch.as_tensor(w,device=device) for n,w in arrays.items()})
    origin=dict(m['origin'],preparation_migration=dict(source=str(root),source_sha256=expected_sha256,
        source_manifest_sha256=expected_manifest_sha256,source_rule_hash=m['rule_hash'],domain=domain,
        all_numeric_arrays_preserved=True,prior_ready_adopted=False,prior_quality_adopted=False,
        formal_updates_inherited=0,optimizer_reset=True,experience_reset=True))
    rt.export(net,output,key,build,origin)
    exported=rt.model(output,key)
    if any(not np.array_equal(w,exported.weights[n]) for n,w in arrays.items()):raise ValueError('Preparation arrays changed')
    return net,origin
