"""Named, normalization-preserving research migration; source arrays stay immutable."""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from . import cross_lineup as old, cross_grounded as old_grounded
from . import seven_lineup as current, seven_grounded as grounded, seven_runtime as rt


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
