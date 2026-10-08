"""Declarative next-round request; never changes frozen five-team model schemas."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from app.modules.card_game.content.duel_v2 import STARTER_DECKS, PUBLIC_CHARACTER_IDS, validate_deck
from .cross_schedule import learning_schedule

PROFILE_PATH = Path(__file__).resolve().parents[1] / 'content' / 'duel_v2' / 'next_training.json'


def plan(now=None):
    request = json.loads(PROFILE_PATH.read_text(encoding='utf-8'))
    roster = tuple(request['roster'])
    found = {deck['id']: deck for deck in STARTER_DECKS}
    decks = {key: validate_deck(deepcopy(found[key])) for key in roster}
    for key, deck in decks.items():
        if deck['character_ids'] != request['character_orders'][key]:
            raise ValueError('Next-round character order changed: ' + key)
        if not set(deck['character_ids']) <= PUBLIC_CHARACTER_IDS:
            raise ValueError('Next-round preset includes restricted characters: ' + key)
    zone = timezone(timedelta(hours=8))
    now = datetime.now(zone) if now is None else now.astimezone(zone)
    hard = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if hard <= now:
        hard += timedelta(days=1)
    pairs = learning_schedule(roster)
    mirrors = [dict(left=key, right=key, first=side) for key in roster for side in ('a', 'b')]
    return dict(request, status='planned_not_started', builds=decks,
                hard_deadline=hard.isoformat(), hard_deadline_epoch=hard.timestamp(),
                cross_learning_schedule=pairs, mirror_learning_schedule=mirrors,
                new_games_per_complete_round=len(pairs) + len(mirrors),
                evaluation_cells=[dict(row=a, column=b) for a in roster for b in roster],
                matrix_games_per_cell=None, preflight_passed=False, quality_approved=False,
                runtime_readiness='six_team_encoding_and_preflight_required')
