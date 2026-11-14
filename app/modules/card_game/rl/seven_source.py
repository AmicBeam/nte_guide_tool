"""Named, normalization-preserving research migration; source arrays stay immutable."""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from . import cross_lineup as old, cross_grounded as old_grounded
from . import seven_lineup as current, seven_grounded as grounded, seven_runtime as rt


def migrate_seven(directory, key, build, output, *, source_snapshot,
                  expected_sha256, expected_manifest_sha256, device='cuda',
                  expand_named_observation=False):
    """Fresh research migration from an authenticated, same-layout seven model.

    Rule changes require fresh preparation/preflight; no previous readiness,
    optimizer, experience, value qualification or serving approval is adopted.
    """
    import ast
    import torch
    from .seven_policy import SCHEMA
    from .build_acceptance import build_identity
    source = Path(source_snapshot).resolve()
    declared = json.loads((source/'source-manifest.json').read_text(encoding='utf8'))
    module = source/'app/modules/card_game'
    paths = sorted({*(module/'engine/duel_v2').rglob('*.py'),
                    *(module/'content/duel_v2').rglob('*.py'),
                    module/'content/duel_v2/catalog.json', module/'rl/league_schema.py',
                    module/'rl/league_observation.py'})
    checked = {}
    def data(path):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink() or source not in path.resolve().parents or relative not in declared:
            raise ValueError('Unverified seven-model source member: '+relative)
        payload = path.read_bytes(); entry = declared[relative]
        if sha256(payload).hexdigest()!=entry['sha256'] or len(payload)!=entry['bytes']:
            raise ValueError('Seven-model source member changed: '+relative)
        checked[relative] = entry['sha256']
        return payload
    tree = ast.parse(data(module/'rl/league_schema.py'))
    schema = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCHEMA' for t in node.targets))
    historical = sha256(schema.encode())
    for path in paths:
        historical.update(path.relative_to(module).as_posix().encode()); historical.update(data(path))
    historical_seven = sha256(historical.hexdigest().encode())
    for name in ('seven_lineup.py','seven_grounded.py','seven_runtime.py','seven_policy.py',
                 'murk_lineup.py','information_search.py','search_policy.py'):
        historical_seven.update(data(module/'rl'/name))
    root = Path(directory); manifest_path=root/(key+'.json'); path=root/(key+'.npz')
    m=json.loads(manifest_path.read_text(encoding='utf8'))
    features = m.get('features')
    target_features = current.all_features()
    expanded = bool(expand_named_observation and isinstance(features, list)
                    and len(features) < len(target_features)
                    and target_features[:len(features)] == features)
    if (sha256(manifest_path.read_bytes()).hexdigest()!=expected_manifest_sha256
            or sha256(path.read_bytes()).hexdigest()!=expected_sha256 or m.get('sha256')!=expected_sha256
            or m.get('rule_hash')!=historical_seven.hexdigest() or m.get('schema')!=SCHEMA
            or m.get('deck')!=key or m.get('reward_mode')!='terminal_wdl_only'
            or m.get('automatic_serving_approval') is not False
            or (features != target_features and not expanded) or m.get('candidates')!=current.candidate_names()
            or m.get('grounded_features')!=grounded.transformed_names()
            or build_identity(m['build'])!=build_identity(build)
            or m.get('architecture_sha256')!=sha256(data(module/'rl/seven_policy.py')).hexdigest()
            or data(module/'rl/seven_policy.py')!=Path(rt.architecture.__file__).read_bytes()
            or (data(module/'rl/seven_grounded.py')!=Path(grounded.__file__).read_bytes() and not expanded)):
        raise ValueError('Authenticated same-layout seven-model migration required')
    net=rt.create_network(key,m['hidden'],device,legacy=bool(m.get('source_identity')))
    expected=net.state_dict()
    with np.load(path,allow_pickle=False) as saved:
        if set(saved.files)!=set(expected):raise ValueError('Seven source tensor set mismatch')
        arrays={name:saved[name].copy() for name in saved.files}
    original_arrays = {name: value.copy() for name, value in arrays.items()}
    if expanded:
        # Append-only public columns are inactive until trained. InputNorm must
        # exclude them as well as zeroing their projection, otherwise merely
        # adding zeros changes the old mean/variance and every existing score.
        for prefix in ('state_net', 'value_net'):
            for suffix in ('active', 'weight', 'bias'):
                name = prefix + '.0.' + suffix
                if arrays[name].shape != (len(features),):
                    raise ValueError('Historical named normalization shape mismatch')
                value = np.zeros(tuple(expected[name].shape), dtype=arrays[name].dtype)
                if suffix == 'weight':
                    value.fill(1)
                value[:len(features)] = arrays[name]
                arrays[name] = value
            name = prefix + '.1.weight'
            if arrays[name].shape != (m['hidden'], len(features)):
                raise ValueError('Historical named input projection shape mismatch')
            value = np.zeros(tuple(expected[name].shape), dtype=arrays[name].dtype)
            value[:, :len(features)] = arrays[name]
            arrays[name] = value
    if any(w.shape!=tuple(expected[n].shape) or w.dtype!=expected[n].detach().cpu().numpy().dtype
           or not np.isfinite(w).all() for n,w in arrays.items()):
        raise ValueError('Seven source tensor shape/nonfinite mismatch')
    net.load_state_dict({n:torch.as_tensor(w,device=device) for n,w in arrays.items()})
    origin=dict(m['origin'],research_migration=dict(source=str(root),source_sha256=expected_sha256,
        source_manifest_sha256=expected_manifest_sha256,source_rule_hash=m['rule_hash'],
        authenticated_source_files=checked,all_numeric_arrays_preserved=not expanded,
        named_observation_expansion=target_features[len(features):] if expanded else [],
        prior_ready_adopted=False,prior_quality_adopted=False,optimizer_reset=True,experience_reset=True))
    rt.export(net,output,key,build,origin)
    policy=rt.model(output,key)
    if any(w.dtype!=policy.weights[n].dtype or w.tobytes()!=policy.weights[n].tobytes()
           for n,w in arrays.items()):
        raise ValueError('Research migration changed seven source arrays')
    for name, old_array in original_arrays.items():
        new_array = policy.weights[name]
        prefix = new_array[tuple(slice(0, size) for size in old_array.shape)]
        if not np.array_equal(prefix, old_array):
            raise ValueError('Named migration changed an existing numerical column')
    return net,origin


def old_transformed_names():
    categories=(tuple(map(str,range(6))),('none',*old.SEATS),('none',*old.CARD_IDS),
                ('none','own','enemy'),('none',*old.SEATS))
    names=[]
    for col,labels in enumerate(categories):
        if col==0:labels=current.ACTION_TYPES[:6]
        names += [f'category:{col}:{label}' for label in labels]
    names += ['numeric:'+name for name in old.candidate_names()[5:]]
    names += [f'{role}:{field}' for role in ('actor','target') for field in old_grounded.FIELDS]
    assert len(names)==old_grounded.DIM
    return names


def migrate(directory,key,build,output,*,source_snapshot,expected_sha256,device='cpu'):
    import torch
    from .residual_policy import tensor_shapes
    from .recovery_policy import extra_shapes, SCHEMAS, PRESERVED_SCHEMA
    root=Path(directory);path=root/(key+'.npz');manifest_path=root/(key+'.json')
    m=json.loads(manifest_path.read_text(encoding='utf8'));digest=sha256(path.read_bytes()).hexdigest()
    snapshot=Path(source_snapshot)/'app/modules/card_game/rl'
    if (digest!=expected_sha256 or m['sha256']!=digest or m['deck']!=key or m['schema'] not in SCHEMAS
            or m['features']!=old.all_features() or m['candidates']!=old.candidate_names()):
        raise ValueError('Historical source SHA/schema/named layout mismatch')
    for field,file in (('architecture_sha256','residual_policy.py'),('candidate_transform_sha256','cross_grounded.py'),
                       ('value_architecture_sha256','recovery_policy.py')):
        if m.get(field)!=sha256((snapshot/file).read_bytes()).hexdigest():raise ValueError('Source architecture snapshot mismatch: '+field)
    legacy=m['schema']==PRESERVED_SCHEMA
    net=rt.create_network(key,m['hidden'],device,legacy=legacy)
    expected=dict(tensor_shapes(m['hidden']),**extra_shapes(m['hidden']))
    if legacy:
        expected.update({n:tuple(t.shape) for n,t in net.state_dict().items() if n.startswith('legacy.')})
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!=set(expected):raise ValueError('Historical tensor set mismatch')
        source={n:data[n].copy() for n in data.files}
    if any(w.shape!=expected[n] or not np.isfinite(w).all() for n,w in source.items()):raise ValueError('Historical arrays invalid')
    mappings={}
    for prefix,names,target in (('state_net',old.all_features(),current.all_features()),
                               ('value_net',old.all_features(),current.all_features()),
                               ('cand_net',old_transformed_names(),grounded.transformed_names())):
        if len(names)!=len(set(names)) or len(target)!=len(set(target)) or not set(names)<=set(target):
            raise ValueError('Named migration contains missing/duplicate columns')
        indexes=[target.index(n) for n in names];mappings[prefix]=indexes
    weights=net.state_dict()
    with torch.no_grad():
        for prefix,indexes in mappings.items():
            weights[prefix+'.0.active'].zero_();weights[prefix+'.0.active'][indexes]=1
            weights[prefix+'.0.weight'].fill_(1);weights[prefix+'.0.weight'][indexes]=torch.as_tensor(source[prefix+'.0.weight'],device=device)
            weights[prefix+'.0.bias'].zero_();weights[prefix+'.0.bias'][indexes]=torch.as_tensor(source[prefix+'.0.bias'],device=device)
            weights[prefix+'.1.weight'].zero_();weights[prefix+'.1.weight'][:,indexes]=torch.as_tensor(source[prefix+'.1.weight'],device=device)
        mapped={prefix+suffix for prefix in mappings for suffix in ('.0.weight','.0.bias','.1.weight')}
        for n,w in source.items():
            if n not in mapped:weights[n].copy_(torch.as_tensor(w,device=device))
    origin=dict(initialization='named_normalization_preserving_migration',source_schema=m['schema'],
                source_rule_hash=m['rule_hash'],source_sha256=digest,source_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
                source_build=m['build'],source_features=m['features'],source_grounded_features=old_transformed_names(),
                normalization='historical_columns_define_mean_and_variance',new_projection_columns_zero=True,
                optimizer_reset=True,experience_reset=True,quality_approved=False)
    rt.export(net,output,key,build,origin)
    return net,origin
