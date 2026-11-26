"""Migrate retired actions and remove steps now handled by automatic settlement."""
from math import floor, isfinite
from typing import Any


def _migrate_form_action_steps(steps: list[dict[str, Any]], actions: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve retired shared actions from the latest setup in their own slot."""
    selectors: dict[str, list[tuple[str, str]]] = {}
    for action_id, action in actions.items():
        for form in (action.get('legacy_form_action_migration') or {}).get('forms', []):
            for setup_id in form.get('setup_action_ids', []):
                selectors.setdefault(setup_id, []).append((action_id, form['action_id']))

    def number(value: Any) -> float:
        try:
            result = float(value or 0)
            return result if isfinite(result) else 0
        except (TypeError, ValueError):
            return 0

    ordered = sorted(enumerate(steps), key=lambda item: (
        number(item[1].get('start_tick')) if isinstance(item[1], dict) else 0, item[0]))
    modes: dict[tuple[int, str], str] = {}
    converted = list(steps)
    for index, step in ordered:
        if not isinstance(step, dict):
            continue
        slot = floor(number(step.get('slot')) + 0.5)
        action_id = str(step.get('action_id') or '')
        for old_id, target_id in selectors.get(action_id, []):
            modes[(slot, old_id)] = target_id
        policy = actions.get(action_id, {}).get('legacy_form_action_migration')
        if not policy:
            continue
        target_id = modes.get((slot, action_id), policy.get('default_action_id'))
        target = actions.get(target_id)
        if target and not target.get('legacy_only'):
            converted[index] = {**step, 'action_id': target_id, 'action_name': target['name']}
    return converted


def migrate_lingke_joint_steps(steps: list[dict[str, Any]], team: list[dict[str, Any]],
                               catalog: dict[str, Any]) -> list[dict[str, Any]]:
    actions = {a['id']: a for a in catalog['actions']}
    steps = _migrate_form_action_steps(steps, actions)
    characters = {c['id']: c for c in catalog['characters']}
    members = sorted(team, key=lambda m: int(m.get('slot', 0)))
    migrated = []
    for step in steps:
        if not isinstance(step, dict):
            migrated.append(step)
            continue
        old = actions.get(str(step.get('action_id') or ''), {})
        if old.get('automatic_settlement'):
            continue
        if not old.get('legacy_only') or old.get('trigger_character_selector') != 'same_element_non_lingke':
            migrated.append(step)
            continue
        candidates = [m for m in members if m.get('character_id') != old.get('character_id')
                      and characters.get(m.get('character_id'), {}).get('element') == old.get('damage_element')]
        member = next((m for m in candidates if m.get('character_id') == step.get('trigger_character_id')),
                      candidates[0] if candidates else None)
        # User 2026-09-18: remove the old step when there is no matching teammate.
        if member is None:
            continue
        support = next((a for a in catalog['actions'] if a.get('character_id') == member['character_id']
                        and a.get('action_type') == '援护' and not a.get('legacy_only')
                        and not a.get('required_buff_key')), None)
        if support is None:
            continue
        converted = {**step, 'slot': member['slot'], 'action_id': support['id'], 'action_name': support['name']}
        converted.pop('trigger_character_id', None)
        migrated.append(converted)
    return migrated
