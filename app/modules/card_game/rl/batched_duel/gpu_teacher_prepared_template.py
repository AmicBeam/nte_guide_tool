"""Exact teacher fanout with one immutable anonymized template per parent.

The canonical covered teacher prepares this same template once and gives every
battle candidate a private copy. These entries preserve that lifecycle across
independent VM tasks; only rule_battle_preview computes the score fields.
"""
from .gpu_teacher_preview_parts import teacher_prepare
from app.modules.card_game.engine.duel_v2.projection import (
    _prepare_preview_template, _PREPARED_PREVIEW, rule_battle_preview,
)
from app.modules.card_game.engine.duel_v2.simulation import _without_patches


def teacher_prepare_with_template(state,actor):
    view,legal,indices=teacher_prepare(state,actor)
    template=_prepare_preview_template(state,actor) if indices else None
    return view,legal,indices,template


def teacher_preview_with_template(state,actor,packet):
    template=packet['template']
    if (not isinstance(template,dict) or template.get('version')!=state.get('version')
            or template.get('event_seq')!=state.get('event_seq')):
        raise ValueError('Prepared teacher template state identity mismatch')
    context=dict(side=actor,source_id=id(state),version=state.get('version'),
        event_seq=state.get('event_seq'),template=template)
    with _without_patches():
        token=_PREPARED_PREVIEW.set(context)
        try:
            return rule_battle_preview(state,actor,packet['action'])
        finally:
            _PREPARED_PREVIEW.reset(token)
