"""Frozen teacher composition parts for GPU combat-preview fanout.

These callables split covered_rollout.rule_decision. They are not a CUDA
backend, not a production CPU fallback, and not a performance claim.
Canonical observe, sandbox, battle preview, and VisibleEngineRulePolicy
remain the rule source.
"""
from app.modules.card_game.engine.duel_v2.projection import (
    observe as canonical_observe,
    prepared_preview_sandbox,
    rule_battle_preview,
)
from app.modules.card_game.engine.duel_v2.simulation import _without_patches
from app.modules.card_game.rl.policies import VisibleEngineRulePolicy


def teacher_prepare(state, actor):
    view = canonical_observe(state, actor, include_previews=False)
    legal = [entry['action'] for entry in view['legal_actions'] if entry['action']['type'] != 'concede']
    if any(action.get('type') in ('mulligan', 'choose') for action in legal):
        return view, legal, []
    cards = {card['instance_id']: card for card in view['sides'][actor]['hand']}
    combat_view_indices = [
        index for index, entry in enumerate(view['legal_actions'])
        if entry['action']['type'] == 'attack' or (
            entry['action']['type'] == 'play_card' and
            cards.get(entry['action'].get('card_id'), {}).get('type') == 'battle')
    ]
    return view, legal, combat_view_indices


def teacher_preview(state, actor, action):
    with _without_patches(), prepared_preview_sandbox(state, actor):
        return rule_battle_preview(state, actor, action)


def teacher_finish(view, legal, indices, previews):
    entries = view['legal_actions']
    expected_legal = [entry['action'] for entry in entries if entry['action']['type'] != 'concede']
    if legal != expected_legal:
        raise ValueError('Teacher legal order mismatch')
    if len(indices) != len(previews):
        raise ValueError('Teacher preview count mismatch')
    bound = len(entries)
    if any(type(index) is not int or index < 0 or index >= bound for index in indices):
        raise ValueError('Teacher preview indices mismatch')
    if len(set(indices)) != len(indices):
        raise ValueError('Teacher preview indices mismatch')
    if any(action.get('type') in ('mulligan', 'choose') for action in legal) and indices:
        raise ValueError('Unsupported teacher preview phase')
    cards = {card['instance_id']: card for card in view['sides'][view['viewer_side']]['hand']}
    early = any(action.get('type') in ('mulligan', 'choose') for action in legal)
    expected_indices = [] if early else [i for i, entry in enumerate(entries)
        if entry['action']['type'] == 'attack' or
        (entry['action']['type'] == 'play_card' and
         cards.get(entry['action'].get('card_id'), {}).get('type') == 'battle')]
    if indices != expected_indices:
        raise ValueError('Teacher full preview order mismatch')
    for index in indices:
        action = entries[index]['action']
        kind = action.get('type')
        if kind != 'attack' and not (
                kind == 'play_card' and cards.get(action.get('card_id'), {}).get('type') == 'battle'):
            raise ValueError('Unsupported candidate alignment')
    for job in range(len(indices)):
        entries[indices[job]]['preview'] = previews[job]
    return legal, int(VisibleEngineRulePolicy()(view, tuple(legal)))


def compose_teacher_rule_decision(state, actor):
    """Unit oracle only: prepare, independent previews, finish. Not a fallback."""
    view, legal, indices = teacher_prepare(state, actor)
    previews = [teacher_preview(state, actor, view['legal_actions'][index]['action'])
                for index in indices]
    return teacher_finish(view, legal, indices, previews)
