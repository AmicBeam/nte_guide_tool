"""Public replay assembly from engine states. No Flask, DAO, or database."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
import re

from app.modules.card_game.engine.duel_v2.presentation import (
    PRIVATE_EVENT_KEYS,
    diff_public,
    localize_text,
    merge_patch,
    project_event,
    replay_public_board,
)
from app.modules.card_game.engine.duel_v2.projection import observe
from app.modules.card_game.engine.duel_v2.state import SIDES

HIDDEN_KEYS = (
    'deck', 'rng', 'seed', 'private_card', 'private_side', 'private_hand', '_deferred_card_marks', '_public_board',
    'next_instance', 'mulligan_done', 'current_action_id', 'current_action_version',
    'combat_group', 'removed', 'requests', 'fingerprint', 'legal_actions',
    'operation', 'turn_snapshots', 'time_removed', 'visible_deck',
)
HAND_EVENT_TYPES = ('play', 'discard', 'remove', 'draw', 'gain', 'mulligan')


def _hidden_card():
    return {'hidden': True}


def _is_hidden_card(card):
    return (not card) or card.get('hidden') or (
        not card.get('copy') and not card.get('instance_id') and not card.get('name')
    )


def _public_removal(value, path):
    return (isinstance(value, dict) and set(value) == {'id', 'removed'}
            and isinstance(value['id'], str) and value['removed'] is True
            and re.search(r'(?:^|\.)patch\.sides\.[ab]\.characters\[\d+\]$', path) is not None)


def strip_hidden(value, path=''):
    if isinstance(value, dict):
        if _public_removal(value, path):
            return deepcopy(value)
        return {key: strip_hidden(item, f'{path}.{key}' if path else key)
                for key, item in value.items() if key not in HIDDEN_KEYS}
    if isinstance(value, list):
        return [strip_hidden(item, f'{path}[{index}]') for index, item in enumerate(value)]
    return deepcopy(value)


def scrub_opponent_hand(view: dict, viewer: str) -> dict:
    sides = view.get('sides') or {}
    for side, team in sides.items():
        if side == viewer:
            for card in team.get('hand') or []:
                if isinstance(card, dict):
                    card.pop('unavailable_reason', None)
            continue
        cleaned = []
        for card in team.get('hand') or []:
            if not isinstance(card, dict):
                cleaned.append(_hidden_card())
                continue
            if card.get('copy') and (card.get('instance_id') or card.get('name')):
                cleaned.append(card)
            else:
                cleaned.append(_hidden_card())
        team['hand'] = cleaned
    pending = view.get('pending_choice')
    if pending and (pending.get('side') != viewer or pending.get('kind') in ('inspect_top', 'hand_redraw')):
        pending['choices'] = []
        pending['prompt'] = '对手正在选择' if pending.get('side') != viewer else '正在进行私有选择'
    return view


def public_replay_view(state: dict, viewer: str) -> dict:
    view = observe(state, viewer)
    view['legal_actions'] = []
    view['is_my_turn'] = False
    view.pop('events', None)
    view.pop('logs', None)
    view.pop('presentation', None)
    scrub_opponent_hand(view, viewer)
    return strip_hidden(view)


def public_replay_event(event: dict, viewer: str) -> dict:
    item = project_event(event, viewer)
    item['text'] = localize_text(event, viewer)
    for key in PRIVATE_EVENT_KEYS:
        item.pop(key, None)
    return strip_hidden(item)


def apply_event_to_hand(board: dict, event: dict) -> dict:
    if not board or not event or event.get('side') not in (board.get('sides') or {}):
        return board
    kind = str(event.get('type') or '')
    if kind not in HAND_EVENT_TYPES:
        return board
    result = board
    team = dict(result['sides'][event['side']] or {})
    viewer = result.get('viewer_side')
    is_self = event['side'] == viewer
    # Engine patches already contain the post-event count. Only reconstruct
    # identities/placeholders here; never account for the same movement twice.
    patched = ((event.get('patch') or {}).get('sides') or {}).get(event['side'], {})
    if 'hand' in patched:
        return board
    hand = list(team.get('hand') or [])
    card = event.get('card') if isinstance(event.get('card'), dict) else None
    if kind == 'mulligan' and is_self:
        outgoing = set(map(str, event.get('card_ids') or []))
        hand = [row for row in hand if str((row or {}).get('instance_id')) not in outgoing]
    elif kind in ('play', 'discard', 'remove'):
        removed = False
        if is_self and card and card.get('instance_id'):
            for index, row in enumerate(hand):
                if row and str(row.get('instance_id')) == str(card.get('instance_id')):
                    hand.pop(index)
                    removed = True
                    break
        if not removed and not is_self and card and card.get('copy') and card.get('instance_id'):
            for index, row in enumerate(hand):
                if row and str(row.get('instance_id')) == str(card.get('instance_id')):
                    hand.pop(index)
                    removed = True
                    break
        if not removed and not is_self and kind != 'discard':
            for index in range(len(hand) - 1, -1, -1):
                if _is_hidden_card(hand[index]) and not (hand[index] or {}).get('copy'):
                    hand.pop(index)
                    break
    elif kind in ('draw', 'gain'):
        if is_self and card and not card.get('hidden'):
            exists = any(
                row and card.get('instance_id') and str(row.get('instance_id')) == str(card.get('instance_id'))
                for row in hand
            )
            if not exists:
                hand.append(deepcopy(card))
        elif not is_self:
            if card and card.get('copy'):
                hand.append(deepcopy(card))
            else:
                hand.append(_hidden_card())
    team['hand'] = hand
    result['sides'][event['side']] = team
    return normalize_replay_hands(result)


def normalize_replay_hands(board: dict) -> dict:
    """Keep placeholders in sync, including private mulligan removals."""
    for team in (board.get('sides') or {}).values():
        if 'hand_count' not in team:
            continue
        hand = list(team.get('hand') or [])
        count = max(0, int(team['hand_count']))
        while len(hand) > count:
            hidden = next((i for i in range(len(hand)-1, -1, -1) if _is_hidden_card(hand[i])), None)
            if hidden is None:
                # A private removal does not identify which old card survived.
                # Hide the uncertain identities instead of displaying stale cards.
                hand = [_hidden_card() for _ in range(count)]
                break
            hand.pop(hidden)
        hand.extend(_hidden_card() for _ in range(max(0, count-len(hand))))
        team['hand'] = hand
    return board


def assemble_replay_game(opening: dict, events: list, *, upto: int | None = None) -> dict:
    applied = list(events or [])
    if upto is not None:
        applied = applied[:max(0, int(upto))]
    board = replay_public_board(opening, applied)
    board['legal_actions'] = []
    board['is_my_turn'] = False
    board['events'] = deepcopy(applied)
    board['logs'] = [event.get('text') or '' for event in applied]
    cursor = applied[-1]['seq'] if applied else int((opening or {}).get('version') or 0)
    oldest = applied[0]['seq'] if applied else 0
    board['presentation'] = {
        'schema_version': 1,
        'cursor': cursor,
        'oldest_seq': oldest,
        'events': deepcopy(applied),
    }
    return strip_hidden(board)


def leak_markers(payload) -> list[str]:
    found = []

    def walk(value, path):
        if isinstance(value, dict):
            if _public_removal(value, path):
                return
            for key, item in value.items():
                here = f'{path}.{key}' if path else key
                if key in HIDDEN_KEYS:
                    found.append(here)
                else:
                    walk(item, here)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f'{path}[{index}]')

    walk(payload, '')
    return found


def label_sides(view: dict, name_a: str, name_b: str) -> dict:
    sides = view.setdefault('sides', {})
    if 'a' in sides:
        sides['a']['name'] = name_a
    if 'b' in sides:
        sides['b']['name'] = name_b
    return view


def append_projected_events(
    view: dict, game: dict, viewer: str, raw_events: list, name_a: str, name_b: str,
) -> None:
    projected = [public_replay_event(event, viewer) for event in raw_events]
    if not projected:
        return
    running = replay_public_board(view['opening_board'], view['events'])
    final = public_replay_view(game, viewer)
    label_sides(final, name_a, name_b)
    for index, item in enumerate(projected):
        engine_patch = item.get('patch') if isinstance(item.get('patch'), dict) else {}
        if index < len(projected) - 1:
            nxt = merge_patch(running, engine_patch)
            nxt = normalize_replay_hands(apply_event_to_hand(nxt, item))
            item['patch'] = diff_public(running, nxt)
            running = nxt
        else:
            item['patch'] = diff_public(running, final)
            running = final
    view['events'].extend(projected)


def build_viewer_log(
    opening_state: dict, final_state: dict, viewer: str, *, name_a: str, name_b: str,
) -> dict:
    opening = public_replay_view(opening_state, viewer)
    label_sides(opening, name_a, name_b)
    view = {'opening_board': opening, 'events': []}
    last_seq = int(opening_state.get('event_seq') or 0)
    raw = [event for event in (final_state.get('events') or []) if int(event.get('seq') or 0) > last_seq]
    append_projected_events(view, final_state, viewer, raw, name_a, name_b)
    return view


def build_match_payload(
    opening_state: dict,
    final_state: dict,
    *,
    room_code: str,
    name_a: str = '训练策略',
    name_b: str = '规则AI',
    mode: str = 'solo',
    source: str = 'rl-eval',
    learning_side: str = 'a',
) -> dict[str, Any]:
    views = {
        side: build_viewer_log(opening_state, final_state, side, name_a=name_a, name_b=name_b)
        for side in SIDES
    }
    status = 'finished' if final_state.get('phase') == 'finished' else 'playing'
    payload = {
        'schema_version': 1,
        'room_code': room_code,
        'mode': mode,
        'status': status,
        'winner': final_state.get('winner'),
        'player_a_id': None,
        'player_b_id': None,
        'name_a': name_a,
        'name_b': name_b,
        'learning_side': learning_side,
        'source': source,
        'last_seq': int(final_state.get('event_seq') or 0),
        'views': views,
        'updated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    leaks = leak_markers(payload)
    if leaks:
        raise ValueError('Replay payload would leak hidden keys: ' + ', '.join(leaks[:8]))
    return payload
