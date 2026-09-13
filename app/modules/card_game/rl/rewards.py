"""Sparse terminal reward plus small public-event shaping. Does not change engine rules."""
from __future__ import annotations

from typing import Any

# Intermediate events are breadcrumbs. A win is ~100x the largest mid event.
TERMINAL_WIN = 10.0
TERMINAL_LOSS = -10.0
SHAPE_HARMONY = 0.05
SHAPE_AWAKEN = 0.08
SHAPE_DAMAGE = 0.02
SHAPE_DOWN = 0.10
SHAPE_COLLAPSE = 0.02
SHAPE_STEP_CAP = 0.5
DAMAGE_EVENT_TYPES = frozenset({
    'combat', 'flower', 'delay', 'penetration', 'followup', 'damage',
    'burn', 'star', 'nightmare',
})


def _hp_lost(event: dict[str, Any]) -> int:
    before = event.get('before') if isinstance(event.get('before'), dict) else {}
    after = event.get('after') if isinstance(event.get('after'), dict) else {}
    try:
        return max(0, int(before.get('hp') or 0) - int(after.get('hp') or 0))
    except (TypeError, ValueError):
        return 0


def shape_reward(events: list | None, learner: str, after_seq: int) -> tuple[float, dict[str, float | int]]:
    """Score new public events from the learner's perspective only."""
    parts: dict[str, float | int] = {
        'harmony': 0, 'awaken': 0, 'damage_hp': 0, 'down': 0, 'collapse': 0, 'shaping': 0.0,
    }
    if learner not in ('a', 'b') or not events:
        return 0.0, parts
    total = 0.0
    prefix = learner + ':'
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            seq = int(event.get('seq') or 0)
        except (TypeError, ValueError):
            continue
        if seq <= after_seq:
            continue
        kind = event.get('type')
        if kind == 'harmony' and event.get('side') == learner:
            total += SHAPE_HARMONY
            parts['harmony'] = int(parts['harmony']) + 1
        elif kind in ('awaken', 'ultimate') and event.get('side') == learner:
            total += SHAPE_AWAKEN
            parts['awaken'] = int(parts['awaken']) + 1
        elif kind == 'collapse' and event.get('side') != learner:
            after = event.get('after') if isinstance(event.get('after'), dict) else {}
            if after.get('collapse') is True:
                total += SHAPE_COLLAPSE
                parts['collapse'] = int(parts['collapse']) + 1
        elif kind == 'down' and event.get('side') != learner:
            total += SHAPE_DOWN
            parts['down'] = int(parts['down']) + 1
        elif kind in DAMAGE_EVENT_TYPES and event.get('side') != learner:
            source = str(event.get('source') or event.get('actor') or '')
            if not source.startswith(prefix):
                continue
            lost = _hp_lost(event)
            if lost:
                total += SHAPE_DAMAGE * lost
                parts['damage_hp'] = int(parts['damage_hp']) + lost
    shaped = min(total, SHAPE_STEP_CAP)
    parts['shaping'] = round(shaped, 6)
    return shaped, parts
