"""Visible rule opponent and real-terminal leaves for an isolated teacher trial.

This specifies a broad rollout opponent, not a learned opponent capability
claim. Root policy and learner remain numerical models. No true opponent build
is used for conditioning, and no heuristic score becomes an environment reward.
"""
from copy import deepcopy
import time
import numpy as np
from ..engine.duel_v2 import acting_side,observe
from ..engine.duel_v2.simulation import simulate_action, _without_patches
from .league_rollout import clean
from .policies import VisibleEngineRulePolicy
from .information_search import softmax,terminal_value


def opponent_action(state,actor,legal,policies,runtime,rng):
    view=observe(state,actor,include_previews=True)
    return int(VisibleEngineRulePolicy()(view,tuple(legal)))


def rule_decision(state,actor):
    # No neural encoding is needed by this policy. Obtain the public legal
    # mask and previews once, rather than rebuilding them through runtime.
    view=observe(state,actor,include_previews=False)
    legal=[e['action'] for e in view['legal_actions'] if e['action']['type']!='concede']
    if any(action.get('type') in ('mulligan','choose') for action in legal):
        return legal,int(VisibleEngineRulePolicy()(view,tuple(legal)))
    from ..engine.duel_v2.projection import rule_battle_preview, prepared_preview_sandbox
    cards={c['instance_id']:c for c in view['sides'][actor]['hand']}
    combat_entries=[entry for entry in view['legal_actions']
                    if entry['action']['type']=='attack' or
                    (entry['action']['type']=='play_card' and
                     cards.get(entry['action'].get('card_id'),{}).get('type')=='battle')]
    if not combat_entries:
        return legal,int(VisibleEngineRulePolicy()(view,tuple(legal)))
    # Anonymize hidden cards once, then clone only a private sandbox per combat
    # action. Numerical results still use the exact rules and score fields.
    with _without_patches(), prepared_preview_sandbox(state,actor):
        for entry in combat_entries:
            entry['preview']=rule_battle_preview(state,actor,entry['action'])
    return legal,int(VisibleEngineRulePolicy()(view,tuple(legal)))


def terminal_leaf(state,viewer,policies,runtime,rng,deadline,*,max_actions=512,own_rule=False):
    world=deepcopy(state)
    for _ in range(max_actions):
        if world['phase']=='finished':return terminal_value(world,viewer),True
        if time.time()>=deadline:return None,False
        actor=acting_side(world)
        if actor==viewer and not own_rule:
            legal,(x,c)=runtime.decision(world,actor)
            scores=runtime.action_scores(policies[actor],x,c)
            index=int(rng.choice(len(legal),p=softmax(scores)))
        else:legal,index=rule_decision(world,actor)
        world=clean(simulate_action(world,actor,legal[index]))
    if world['phase']=='finished':return terminal_value(world,viewer),True
    return None,False
