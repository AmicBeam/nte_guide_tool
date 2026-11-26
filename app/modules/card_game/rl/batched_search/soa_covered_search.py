"""Explicit scoped auxiliary-value binding for formal covered SoA search.

This factory binds components; it does not drive games, train models, or grant
GPU/full-search qualification. Rules and policy inference remain caller-owned.
"""
from dataclasses import dataclass,field
import math
from typing import Any


@dataclass
class SoACoveredSearchBundle:
    search:Any
    auxiliary:Any
    scope_receipt:dict
    auxiliary_keys:tuple
    auxiliary_versions:tuple
    closed:bool=field(default=False,init=False)

    def close(self):
        if self.closed:return
        self.closed=True
        try:self.search.close()
        finally:self.auxiliary.close()


_PERFORMANCE_OPTIONS={'queued_row_work','row_work_seconds','group_ready_leaves',
                      'reuse_root_observation','reuse_interior_observation'}


def create_soa_covered_search_bundle(rules,inference,statistics,keys,*,auxiliary_scope,
                                     rows,random,max_actions=800,inference_rows=2,noise=False,
                                     q_scale='natural_wdl',**options):
    """Bind verified seven-team auxiliary models to fixed h3 Gumbel32 search.

    Policy and auxiliary model indices are separate inputs to initialize().
    Callers must route auxiliary IDs using the returned auxiliary_keys order.
    """
    from .device_soa_rules import DeviceSoARules
    if not isinstance(rules,DeviceSoARules):
        raise TypeError('Actual DeviceSoARules owner required')
    if q_scale not in ('natural_wdl', 'mctx_default'):
        raise ValueError('Unknown covered search Q profile')
    if getattr(rules,'require_teacher',None) is not True or getattr(rules,'staged_teacher',None) is not True:
        raise ValueError('Covered search requires staged three-step SoA teacher rules')
    unknown=set(options)-_PERFORMANCE_OPTIONS
    if unknown:
        raise ValueError('Fixed covered search contract; unsupported overrides: '+', '.join(sorted(unknown)))
    for name,value,high in (('rows',rows,1600),('max_actions',max_actions,800),('inference_rows',inference_rows,32)):
        if type(value) is not int or not 1<=value<=high:
            raise ValueError('Bounded '+name+' required')
    if rules.rows!=rows or rules.max_actions!=max_actions:
        raise ValueError('Covered search must match the exact rule cohort/action capacity')
    if type(noise) is not bool:
        raise TypeError('Explicit covered search noise selection required')
    if getattr(rules.device,'type',None)!='cuda':
        raise ValueError('Covered SoA binding requires the actual CUDA rule device')
    if getattr(inference,'device',None)!=rules.device:
        raise ValueError('Policy inference and SoA rules must share a device')
    config=dict(queued_row_work=True,row_work_seconds=120,group_ready_leaves=True,
                reuse_root_observation=False,reuse_interior_observation=False)
    config.update(options)
    for name in _PERFORMANCE_OPTIONS-{'row_work_seconds'}:
        if type(config[name]) is not bool:
            raise TypeError('Explicit boolean '+name+' required')
    seconds=config['row_work_seconds']
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=86400:
        raise ValueError('Bounded finite row-work seconds required')
    # The real loader checks rule/algorithm identity, raw evidence completeness,
    # disjoint partitions and every actual model's heldout calibration/version.
    from .. import seven_covered_value
    from ..seven_schedule import ROSTER
    models,scope=seven_covered_value.load_scope(auxiliary_scope)
    if scope.get('passed') is not True:
        raise ValueError('Verified covered auxiliary scope required')
    if set(models)!=set(ROSTER) or set(scope.get('models',{}))!=set(ROSTER):
        raise ValueError('Exactly seven covered auxiliary models required')
    versions={key:models[key].version for key in ROSTER}
    if versions!=scope['models']:
        raise ValueError('Covered auxiliary model versions differ from verified scope')
    from .device_value_only_inference import DeviceValueOnlyInference
    from .device_search import DeviceGumbelSearch
    auxiliary=None;search=None
    try:
        tile=min(rows,inference_rows)
        auxiliary=DeviceValueOnlyInference(models,device=str(rules.device),max_rows=tile,
            max_candidate_elements=tile*max_actions*1576)
        if auxiliary is inference:
            raise ValueError('Covered auxiliary must be independent from the policy bank')
        auxiliary_keys=tuple(auxiliary.keys);auxiliary_versions=tuple(auxiliary.versions)
        if (len(auxiliary_keys)!=7 or set(auxiliary_keys)!=set(ROSTER)
                or auxiliary_versions!=tuple(versions[key] for key in auxiliary_keys)):
            raise ValueError('Device auxiliary bank differs from its verified seven-model scope')
        if auxiliary.device!=rules.device:
            raise ValueError('Auxiliary bank and SoA rules must share a device')
        search=DeviceGumbelSearch(rules,inference,auxiliary,statistics,keys,rows=rows,
            max_actions=max_actions,simulations=32,gumbel_candidates=16,max_depth=10,
            teacher_mode='covered_value',terminal_horizon=3,opponent_mode='rule',
            prior_guide=None,random=random,noise=noise,device=str(rules.device),
            value_batch_rows=tile,reuse_sampled_world=False,reuse_root_decision=False,
            reuse_root_policy_value=False,verify_root_value_reuse=False,q_scale=q_scale,**config)
        receipt={name:scope.get(name) for name in ('schema','objective','rule_hash','algorithm_sha256')}
        receipt.update(scope_validated=True,models=dict(versions),world_sampling='per_simulation',q_scale=q_scale,
                       gpu_qualified=False,full_search_qualified=False,training_qualified=False)
        return SoACoveredSearchBundle(search,auxiliary,receipt,auxiliary_keys,auxiliary_versions)
    except BaseException:
        try:
            if search is not None:search.close()
        finally:
            if auxiliary is not None and auxiliary is not inference:auxiliary.close()
        raise
