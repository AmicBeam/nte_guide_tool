"""Migrate retired joints and remove steps now handled by automatic settlement."""
from typing import Any


def migrate_lingke_joint_steps(steps: list[dict[str, Any]], team: list[dict[str, Any]],
                               catalog: dict[str, Any]) -> list[dict[str, Any]]:
    actions = {a['id']: a for a in catalog['actions']}
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
