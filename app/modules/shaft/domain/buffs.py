from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_TRIGGER_EVENTS = {
    'passive',
    'action_start',
    'action_hit',
    'action_end',
    'loop_start',
    'reaction_trigger',
    'periodic_damage',
    'full_stack',
}

PERMANENT_BUFF_END_TICK = 1_000_000_000


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    return int(round(_num(value, default)))


def _stack_num(value: Any, default: float = 1.0) -> float:
    return max(0.0, _num(value, default))


def _duration_ticks(rule: dict[str, Any]) -> int:
    duration = rule.get('duration') if isinstance(rule.get('duration'), dict) else {}
    ticks = max(0, _int(duration.get('ticks')))
    raw_nodes = rule.get('owner_awakening_nodes')
    active_nodes = (
        {_int(value) for value in raw_nodes}
        if isinstance(raw_nodes, list)
        else set(range(1, max(0, min(6, _int(rule.get('owner_awakening')))) + 1))
    )
    for raw_level, value in (
        duration.get('ticks_by_awakening_node')
        if isinstance(duration.get('ticks_by_awakening_node'), dict)
        else {}
    ).items():
        if _int(raw_level) in active_nodes:
            ticks = max(0, _int(value))
    return ticks


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _awakening_nodes(source: dict[str, Any]) -> set[int]:
    raw_nodes = source.get('awakening_nodes')
    if isinstance(raw_nodes, list):
        return {level for value in raw_nodes if 1 <= (level := _int(value)) <= 6}
    return set(range(1, max(0, min(6, _int(source.get('awakening')))) + 1))


def _str_set(value: Any) -> set[str]:
    return {str(item) for item in _as_list(value) if str(item)}


def _record_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get('id') or ''): record for record in records}


def registered_buff_rules(team: list[dict[str, Any]], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve catalog buff definitions into owner-slot runtime rules."""
    buffs = catalog.get('buffs') if isinstance(catalog.get('buffs'), list) else []
    characters = _record_by_id(catalog.get('characters') if isinstance(catalog.get('characters'), list) else [])
    arcs = _record_by_id(catalog.get('arcs') if isinstance(catalog.get('arcs'), list) else [])
    cartridges = _record_by_id(catalog.get('cartridges') if isinstance(catalog.get('cartridges'), list) else [])
    rules: list[dict[str, Any]] = []
    for buff in buffs:
        if not isinstance(buff, dict):
            continue
        providers = _as_list(buff.get('providers'))
        if not providers:
            continue
        for member in team:
            for provider in providers:
                if not isinstance(provider, dict):
                    continue
                kind = str(provider.get('kind') or '')
                provider_id = str(provider.get('id') or '')
                if kind == 'arc':
                    selected_id = str(member.get('arc_id') or '')
                elif kind == 'cartridge':
                    selected_id = str(member.get('cartridge_id') or '')
                elif kind == 'character':
                    selected_id = str(member.get('character_id') or '')
                else:
                    selected_id = ''
                if not provider_id or provider_id != selected_id:
                    continue
                rule = deepcopy(buff)
                rule['owner_slot'] = _int(member.get('slot'))
                rule['owner_character_id'] = str(member.get('character_id') or '')
                rule['owner_character_name'] = str(member.get('character_name') or '')
                rule['owner_awakening'] = len(_awakening_nodes(member))
                rule['owner_awakening_nodes'] = sorted(_awakening_nodes(member))
                arc_refinement_record = (
                    (((catalog.get('arc_refinements') or {}).get('arcs') or {}).get(provider_id) or {})
                    if kind == 'arc'
                    else {}
                )
                requested_arc_refinement = _int(member.get('arc_refinement'))
                rule['owner_arc_refinement'] = (
                    requested_arc_refinement
                    if 1 <= requested_arc_refinement <= 5
                    else max(1, min(5, _int(arc_refinement_record.get('default_level'), 1)))
                )
                if kind == 'arc':
                    refinement = (
                        arc_refinement_record.get('levels', {})
                        .get(str(rule['owner_arc_refinement']))
                    )
                    refinement_effects = (
                        (refinement.get('buff_effects') or {}).get(str(rule.get('id') or ''))
                        if isinstance(refinement, dict)
                        else None
                    )
                    if isinstance(refinement_effects, dict):
                        rule['effects'] = deepcopy(refinement_effects)
                    rule['provider_name'] = (arcs.get(provider_id) or {}).get('name') or ''
                elif kind == 'cartridge':
                    rule['provider_name'] = (cartridges.get(provider_id) or {}).get('name') or ''
                elif kind == 'character':
                    rule['provider_name'] = (characters.get(provider_id) or {}).get('name') or rule['owner_character_name']
                rule['provider_kind'] = kind
                rules.append(rule)
    rules.sort(key=lambda item: (_int(item.get('priority'), 100), str(item.get('id') or ''), _int(item.get('owner_slot'))))
    return rules


def _action_tags(action: dict[str, Any]) -> set[str]:
    tags = _str_set(action.get('tags'))
    extra_tag = str(action.get('extra_tag') or '')
    if extra_tag:
        tags.add(extra_tag)
    return tags


def _placement(is_background: bool) -> str:
    return 'background' if is_background else 'foreground'


def _source_matches(
    source: dict[str, Any],
    rule: dict[str, Any],
    step: dict[str, Any],
    action: dict[str, Any],
    snapshot: dict[str, Any],
    is_background: bool,
) -> bool:
    scope = str(source.get('scope') or 'registrar')
    if scope == 'registrar' and _int(step.get('slot')) != _int(rule.get('owner_slot')):
        return False
    if scope == 'non_registrar' and _int(step.get('slot')) == _int(rule.get('owner_slot')):
        return False
    action_types = _str_set(source.get('action_types'))
    if action_types and str(action.get('action_type') or '') not in action_types:
        return False
    damage_types = _str_set(source.get('damage_types'))
    if damage_types and str(action.get('damage_type') or '') not in damage_types:
        return False
    action_names = _str_set(source.get('action_names'))
    if action_names and str(action.get('name') or '') not in action_names:
        return False
    action_ids = _str_set(source.get('action_ids'))
    if action_ids and str(action.get('id') or '') not in action_ids:
        return False
    tags = _str_set(source.get('tags'))
    if tags and not (_action_tags(action) & tags):
        return False
    placements = _str_set(source.get('placements'))
    if placements and _placement(is_background) not in placements:
        return False
    elements = _str_set(source.get('elements'))
    character = snapshot.get('character') if isinstance(snapshot.get('character'), dict) else {}
    if elements and str(character.get('element') or '') not in elements:
        return False
    return True


def _conditions_match(conditions: Any, context: dict[str, Any]) -> bool:
    for condition in _as_list(conditions):
        if not isinstance(condition, dict):
            return False
        condition_type = str(condition.get('type') or '')
        if condition_type in {'', 'always'}:
            continue
        if condition_type == 'unsupported':
            return False
        if condition_type == 'action_tag':
            tags = {str(tag) for tag in _as_list(context.get('action_tags')) if str(tag)}
            expected = _str_set(condition.get('tags'))
            if expected and not (tags & expected):
                return False
            continue
        if condition_type == 'self_hp_loss':
            tags = {str(tag) for tag in _as_list(context.get('action_tags')) if str(tag)}
            if not (tags & {'self_hp_loss', 'hp_loss', '扣血', '降低生命'}):
                return False
            continue
        if condition_type == 'heal':
            tags = {str(tag) for tag in _as_list(context.get('action_tags')) if str(tag)}
            if not (tags & {'heal', '治疗'}):
                return False
            continue
        if condition_type == 'fons_full':
            if context.get('fons_full') is False:
                return False
            continue
        if condition_type == 'awakening_min':
            level = _int(condition.get('min'), _int(condition.get('value')))
            raw_nodes = context.get('owner_awakening_nodes')
            if (
                level not in {_int(value) for value in raw_nodes}
                if isinstance(raw_nodes, list)
                else _int(context.get('owner_awakening')) < level
            ):
                return False
            continue
        if condition_type == 'awakening_max':
            next_level = _int(condition.get('max'), _int(condition.get('value'))) + 1
            raw_nodes = context.get('owner_awakening_nodes')
            if (
                next_level in {_int(value) for value in raw_nodes}
                if isinstance(raw_nodes, list)
                else _int(context.get('owner_awakening')) >= next_level
            ):
                return False
            continue
        if condition_type == 'awakening_count_min':
            required_count = _int(condition.get('min'), _int(condition.get('value')))
            raw_nodes = context.get('owner_awakening_nodes')
            active_count = (
                len({_int(value) for value in raw_nodes if 1 <= _int(value) <= 6})
                if isinstance(raw_nodes, list)
                else _int(context.get('owner_awakening'))
            )
            if active_count < required_count:
                return False
            continue
        if condition_type == 'awakening_count_max':
            max_count = _int(condition.get('max'), _int(condition.get('value')))
            raw_nodes = context.get('owner_awakening_nodes')
            active_count = (
                len({_int(value) for value in raw_nodes if 1 <= _int(value) <= 6})
                if isinstance(raw_nodes, list)
                else _int(context.get('owner_awakening'))
            )
            if active_count > max_count:
                return False
            continue
        if condition_type == 'expected_critical_hit':
            if _num(context.get('expected_critical_hits')) <= 0:
                return False
            continue
        if condition_type == 'hit_count_positive':
            if _num(context.get('hit_count')) <= 0:
                return False
            continue
        if condition_type == 'enemy_debuff_active':
            active = {
                str(key)
                for key, value in (context.get('enemy_debuffs') if isinstance(context.get('enemy_debuffs'), dict) else {}).items()
                if _num(value) > _num(context.get('tick'))
            }
            expected = _str_set(condition.get('debuffs'))
            if expected and not (active & expected):
                return False
            continue
        if condition_type == 'enemy_debuff_applied':
            applied = {str(item) for item in _as_list(context.get('applied_enemy_debuffs')) if str(item)}
            expected = _str_set(condition.get('debuffs'))
            if expected and not (applied & expected):
                return False
            continue
        if condition_type == 'shield_state':
            if not bool(context.get('shield_active')):
                return False
            continue
        if condition_type == 'owner_character_id':
            if str(context.get('owner_character_id') or '') not in _str_set(condition.get('ids')):
                return False
            continue
        if condition_type == 'take_damage':
            if not bool(context.get('take_damage')):
                return False
            continue
        if condition_type == 'enemy_hp_below':
            enemy = context.get('enemy') if isinstance(context.get('enemy'), dict) else {}
            hp_ratio = _num(enemy.get('hp_ratio'), _num(enemy.get('hp_percent'), 100) / 100)
            if hp_ratio >= _num(condition.get('threshold'), 0.5):
                return False
            continue
        if condition_type == 'enemy_weak_to_owner_element':
            snapshot = context.get('snapshot') if isinstance(context.get('snapshot'), dict) else {}
            character = snapshot.get('character') if isinstance(snapshot.get('character'), dict) else {}
            element = str(character.get('element') or '')
            enemy = context.get('enemy') if isinstance(context.get('enemy'), dict) else {}
            if element not in set(enemy.get('weakness_elements') or []):
                return False
            continue
        if condition_type == 'active_buff_key':
            if str(condition.get('key') or '') not in {str(item) for item in _as_list(context.get('active_buff_keys'))}:
                return False
            continue
        if condition_type == 'active_buff_any':
            active_keys = {str(item) for item in _as_list(context.get('active_buff_keys'))}
            if not any(str(key) in active_keys for key in _as_list(condition.get('keys'))):
                return False
            continue
        if condition_type == 'active_buff_none':
            active_keys = {str(item) for item in _as_list(context.get('active_buff_keys'))}
            if any(str(key) in active_keys for key in _as_list(condition.get('keys'))):
                return False
            continue
        if condition_type == 'existing_buff_none':
            existing_keys = {str(item) for item in _as_list(context.get('existing_buff_keys'))}
            if any(str(key) in existing_keys for key in _as_list(condition.get('keys'))):
                return False
            continue
        if condition_type == 'reaction_owner_involved':
            reaction = context.get('reaction') if isinstance(context.get('reaction'), dict) else {}
            owner_slot = _int(context.get('owner_slot'), -1)
            if owner_slot not in {_int(reaction.get('previous_slot'), -2), _int(reaction.get('support_slot'), -2)}:
                return False
            continue
        if condition_type == 'reaction_type':
            reaction = context.get('reaction') if isinstance(context.get('reaction'), dict) else {}
            if str(reaction.get('reaction') or '') not in _str_set(condition.get('reactions')):
                return False
            continue
        return False
    return True


def stack_gain_for_rule(rule: dict[str, Any], context: dict[str, Any] | None = None) -> float:
    stacking = rule.get('stacking') if isinstance(rule.get('stacking'), dict) else {}
    if stacking.get('stack_gain') not in (None, ''):
        return _stack_num(stacking.get('stack_gain'), 1.0)
    if str(stacking.get('stack_gain_from') or '') == 'hit_count':
        return _stack_num((context or {}).get('hit_count'), 0.0)
    trigger = rule.get('trigger') if isinstance(rule.get('trigger'), dict) else {}
    for condition in _as_list(trigger.get('conditions')):
        if isinstance(condition, dict) and str(condition.get('type') or '') == 'expected_critical_hit':
            return _stack_num((context or {}).get('expected_critical_hits'), 0.0)
    return 1.0


def event_matches_rule(
    rule: dict[str, Any],
    event: str,
    step: dict[str, Any],
    action: dict[str, Any],
    snapshot: dict[str, Any],
    is_background: bool,
    context: dict[str, Any] | None = None,
) -> bool:
    trigger = rule.get('trigger') if isinstance(rule.get('trigger'), dict) else {}
    matches_periodic_damage = event == 'periodic_damage' and trigger.get('periodic_damage') is True
    if str(trigger.get('event') or '') != event and not matches_periodic_damage:
        return False
    if event not in SUPPORTED_TRIGGER_EVENTS:
        return False
    source = trigger.get('source') if isinstance(trigger.get('source'), dict) else {}
    if not _source_matches(source, rule, step, action, snapshot, is_background):
        return False
    event_context = dict(context or {})
    event_context.setdefault('snapshot', snapshot)
    event_context.setdefault('owner_awakening', rule.get('owner_awakening'))
    event_context.setdefault('owner_awakening_nodes', rule.get('owner_awakening_nodes'))
    return _conditions_match(trigger.get('conditions'), event_context)


def _target_matches(
    target: dict[str, Any],
    instance: dict[str, Any],
    step: dict[str, Any],
    action: dict[str, Any],
    snapshot: dict[str, Any],
    is_background: bool,
    context: dict[str, Any] | None = None,
) -> bool:
    scope = str(target.get('scope') or 'registrar')
    owner_slot = _int(instance.get('owner_slot'))
    step_slot = _int(step.get('slot'))
    if scope == 'registrar' and step_slot != owner_slot:
        return False
    if scope == 'other_team' and step_slot == owner_slot:
        return False
    if scope == 'front' and is_background:
        return False
    if scope == 'front_non_registrar' and (is_background or step_slot == owner_slot):
        return False
    if scope == 'front_registrar' and (is_background or step_slot != owner_slot):
        return False
    if scope == 'slots':
        slots = {_int(item) for item in _as_list(target.get('slots'))}
        if slots and step_slot not in slots:
            return False
    action_types = _str_set(target.get('action_types'))
    if action_types and str(action.get('action_type') or '') not in action_types:
        return False
    action_ids = _str_set(target.get('action_ids'))
    if action_ids and str(action.get('id') or '') not in action_ids:
        return False
    damage_types = _str_set(target.get('damage_types'))
    if damage_types and str(action.get('damage_type') or '') not in damage_types:
        return False
    action_names = _str_set(target.get('action_names'))
    if action_names and str(action.get('name') or '') not in action_names:
        return False
    tags = _str_set(target.get('tags'))
    if tags and not (_action_tags(action) & tags):
        return False
    placements = _str_set(target.get('placements'))
    if placements and _placement(is_background) not in placements:
        return False
    elements = _str_set(target.get('elements'))
    character = snapshot.get('character') if isinstance(snapshot.get('character'), dict) else {}
    if elements and str(character.get('element') or '') not in elements:
        return False
    target_context = dict(context or {})
    target_context.setdefault('snapshot', snapshot)
    if not _conditions_match(target.get('conditions'), target_context):
        return False
    return True


def active_buff_resets_on_action_start(
    instance: dict[str, Any],
    step: dict[str, Any],
    action: dict[str, Any],
    is_background: bool,
    tick: int = 0,
    previous_foreground_slot: int | None = None,
) -> bool:
    rule = instance.get('rule') if isinstance(instance.get('rule'), dict) else {}
    reset = rule.get('reset') if isinstance(rule.get('reset'), dict) else {}
    if not reset or is_background:
        return False
    owner_slot = _int(instance.get('owner_slot'))
    step_slot = _int(step.get('slot'))
    if bool(reset.get('owner_foreground')) and step_slot == owner_slot:
        return True
    if bool(reset.get('owner_leaves_foreground')) and step_slot != owner_slot:
        return True
    if bool(reset.get('owner_leaves_foreground_after_start')) and step_slot != owner_slot:
        if tick < _int(instance.get('start_tick')):
            return False
        return _int(previous_foreground_slot, -1) == owner_slot
    action_ids = _str_set(reset.get('action_ids'))
    if action_ids and str(action.get('id') or '') in action_ids:
        return True
    action_names = _str_set(reset.get('action_names'))
    if action_names and str(action.get('name') or '') in action_names:
        return True
    action_types = _str_set(reset.get('action_types'))
    if action_types and str(action.get('action_type') or '') in action_types:
        return True
    return False


def trigger_cooldown_ticks(rule: dict[str, Any]) -> int:
    trigger = rule.get('trigger') if isinstance(rule.get('trigger'), dict) else {}
    return max(0, _int(trigger.get('cooldown_ticks')))


def active_buff_applies(
    instance: dict[str, Any],
    step: dict[str, Any],
    action: dict[str, Any],
    snapshot: dict[str, Any],
    is_background: bool,
    context: dict[str, Any] | None = None,
) -> bool:
    rule = instance.get('rule') if isinstance(instance.get('rule'), dict) else {}
    target = rule.get('target') if isinstance(rule.get('target'), dict) else {}
    return _target_matches(target, instance, step, action, snapshot, is_background, context)


def activate_buff(
    active_buffs: list[dict[str, Any]],
    rule: dict[str, Any],
    trigger_tick: int,
    stack_gain: float = 1.0,
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    duration = rule.get('duration') if isinstance(rule.get('duration'), dict) else {}
    duration_ticks = PERMANENT_BUFF_END_TICK if str(duration.get('type') or '') == 'permanent' else _duration_ticks(rule)
    if duration_ticks <= 0:
        return None
    start_tick = trigger_tick + max(0, _int(duration.get('delay_ticks')))
    end_tick = start_tick + duration_ticks
    pause_while_buff_key = str(duration.get('pause_while_buff_key') or '')
    if pause_while_buff_key:
        pause_until_tick = max(
            (
                _int(instance.get('end_tick'))
                for instance in active_buffs
                if _int(instance.get('owner_slot')) == _int(rule.get('owner_slot'))
                and str(instance.get('definition_id') or '') == pause_while_buff_key
                and start_tick >= _int(instance.get('start_tick'))
                and start_tick < _int(instance.get('end_tick'))
            ),
            default=start_tick,
        )
        end_tick += max(0, pause_until_tick - start_tick)
    stacking = rule.get('stacking') if isinstance(rule.get('stacking'), dict) else {}
    stacking_mode = str(stacking.get('mode') or 'refresh')
    max_stacks = max(1.0, _num(stacking.get('max_stacks'), 1))
    stack_gain = max(0.0, stack_gain)
    if stack_gain <= 0:
        return None
    definition_id = str(stacking.get('key') or rule.get('id') or '')
    owner_slot = _int(rule.get('owner_slot'))
    for instance in active_buffs:
        existing_rule = instance.get('rule') if isinstance(instance.get('rule'), dict) else {}
        existing_stacking = existing_rule.get('stacking') if isinstance(existing_rule.get('stacking'), dict) else {}
        existing_definition_id = str(instance.get('definition_id') or existing_stacking.get('key') or existing_rule.get('id') or '')
        if existing_definition_id != definition_id or _int(instance.get('owner_slot')) != owner_slot:
            continue
        if stacking_mode == 'independent':
            continue
        if stacking_mode == 'add_stack':
            if stacking.get('unique_source_slots'):
                source_slot = _int((context or {}).get('source_slot'), -1)
                source_slots = {_int(item) for item in _as_list(instance.get('source_slots'))}
                if source_slot in source_slots:
                    return None
                source_slots.add(source_slot)
                instance['source_slots'] = sorted(source_slots)
            instance['stack_count'] = min(max_stacks, max(0.0, _num(instance.get('stack_count'), 1)) + stack_gain)
            instance['start_tick'] = min(_int(instance.get('start_tick'), start_tick), start_tick)
            instance['end_tick'] = end_tick
            instance['rule'] = rule
            instance['name'] = rule.get('name') or instance.get('name') or ''
            return instance
        if stacking_mode == 'extend':
            instance['end_tick'] = max(_int(instance.get('end_tick')), end_tick)
            return instance
        if stacking_mode == 'extend_duration':
            instance['end_tick'] = max(_int(instance.get('end_tick')), trigger_tick) + duration_ticks
            return instance
        instance['start_tick'] = start_tick
        instance['end_tick'] = end_tick
        instance['stack_count'] = min(max_stacks, max(1.0, stack_gain))
        return instance
    if stacking_mode == 'independent':
        latest_instance = None
        for _ in range(max(1, int(stack_gain))):
            siblings = sorted(
                (
                    instance
                    for instance in active_buffs
                    if str(instance.get('definition_id') or '') == definition_id
                    and _int(instance.get('owner_slot')) == owner_slot
                ),
                key=lambda instance: (_int(instance.get('end_tick')), _int(instance.get('start_tick'))),
            )
            while len(siblings) >= int(max_stacks):
                active_buffs.remove(siblings.pop(0))
            latest_instance = {
                'rule': rule,
                'definition_id': definition_id,
                'name': rule.get('name') or '',
                'owner_slot': owner_slot,
                'start_tick': start_tick,
                'end_tick': end_tick,
                'stack_count': 1,
            }
            if stacking.get('unique_source_slots'):
                latest_instance['source_slots'] = [_int((context or {}).get('source_slot'), -1)]
            active_buffs.append(latest_instance)
        return latest_instance
    instance = {
        'rule': rule,
        'definition_id': definition_id,
        'name': rule.get('name') or '',
        'owner_slot': owner_slot,
        'start_tick': start_tick,
        'end_tick': end_tick,
        'stack_count': min(max_stacks, max(1.0, stack_gain) if stacking_mode != 'add_stack' else stack_gain),
    }
    if stacking.get('unique_source_slots'):
        instance['source_slots'] = [_int((context or {}).get('source_slot'), -1)]
    active_buffs.append(instance)
    return instance


def buff_effects(instance: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, float]:
    rule = instance.get('rule') if isinstance(instance.get('rule'), dict) else {}
    effects = rule.get('effects') if isinstance(rule.get('effects'), dict) else {}
    factor = max(0.0, _num(instance.get('stack_count'), 1))
    resolved = {str(key): _num(value) * factor for key, value in effects.items() if _num(value) != 0}
    dynamic = rule.get('dynamic_effects') if isinstance(rule.get('dynamic_effects'), dict) else {}
    runtime = context or {}
    negative = dynamic.get('negative_effect_count') if isinstance(dynamic.get('negative_effect_count'), dict) else {}
    if negative.get('effect_key'):
        enemy_debuffs = set((runtime.get('enemy_debuffs') or {}).keys())
        active_keys = {str(key) for key in _as_list(runtime.get('active_buff_keys'))}
        damage_tags = {str(tag) for tag in _as_list(negative.get('damage_tags'))}
        tagged_damage_types = {
            str(source.get('type_id') or '')
            for source in _as_list(runtime.get('active_damage_sources'))
            if isinstance(source, dict)
            and str(source.get('type_id') or '')
            and damage_tags.intersection(str(tag) for tag in _as_list(source.get('tags')))
        }
        count = min(
            max(0, _int(negative.get('max_count'), 1)),
            sum(str(key) in enemy_debuffs for key in _as_list(negative.get('enemy_debuffs')))
            + sum(str(key) in active_keys for key in _as_list(negative.get('buff_keys')))
            + len(tagged_damage_types),
        )
        effect_key = str(negative.get('effect_key'))
        resolved[effect_key] = _num(resolved.get(effect_key)) + count * _num(negative.get('per_count'))
    active_stack = dynamic.get('active_stack_count') if isinstance(dynamic.get('active_stack_count'), dict) else {}
    if active_stack.get('effect_key') and active_stack.get('key'):
        count = min(
            max(0, _int(active_stack.get('max_count'), 1)),
            sum(
                max(0.0, _num(buff.get('stack_count'), 1))
                for buff in _as_list(runtime.get('active_buffs'))
                if isinstance(buff, dict) and str(buff.get('definition_id') or '') == str(active_stack.get('key'))
            ),
        )
        effect_key = str(active_stack.get('effect_key'))
        resolved[effect_key] = _num(resolved.get(effect_key)) + count * _num(active_stack.get('per_count'))
    elapsed = dynamic.get('elapsed_ticks') if isinstance(dynamic.get('elapsed_ticks'), dict) else {}
    if elapsed.get('effect_key'):
        intervals = max(
            0,
            (_int(runtime.get('tick')) - _int(instance.get('start_tick')))
            // max(1, _int(elapsed.get('interval_ticks'), 10)),
        )
        effect_key = str(elapsed.get('effect_key'))
        resolved[effect_key] = _num(resolved.get(effect_key)) + min(
            _num(elapsed.get('max_value'), float('inf')),
            intervals * _num(elapsed.get('per_interval')),
        )
    return {key: value for key, value in resolved.items() if value != 0}


def buff_displays_as_line(rule: dict[str, Any]) -> bool:
    display = rule.get('display') if isinstance(rule.get('display'), dict) else {}
    if display.get('line') is False:
        return False
    if display.get('line') is True:
        return True
    duration = rule.get('duration') if isinstance(rule.get('duration'), dict) else {}
    target = rule.get('target') if isinstance(rule.get('target'), dict) else {}
    scope = str(target.get('scope') or 'registrar')
    duration_ticks = max(0, _int(duration.get('ticks')))
    delay_ticks = max(0, _int(duration.get('delay_ticks')))
    if scope in {'registrar', 'front_registrar'} and duration_ticks <= 1 and delay_ticks <= 0:
        return False
    return True


def buff_summary(instance: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    rule = instance.get('rule') if isinstance(instance.get('rule'), dict) else {}
    stacking = rule.get('stacking') if isinstance(rule.get('stacking'), dict) else {}
    stack_count = max(0.0, _num(instance.get('stack_count'), 1))
    display_as_line = buff_displays_as_line(rule)
    return {
        'rule_id': rule.get('id') or '',
        'definition_id': str(instance.get('definition_id') or stacking.get('key') or rule.get('id') or ''),
        'name': rule.get('name') or '',
        'provider_name': rule.get('provider_name') or '',
        'owner_slot': _int(instance.get('owner_slot')),
        'start_tick': _int(instance.get('start_tick')),
        'end_tick': _int(instance.get('end_tick')),
        'stacking_mode': str(stacking.get('mode') or 'refresh'),
        'max_stacks': max(1, _int(stacking.get('max_stacks'), 1)),
        'stack_count': int(stack_count) if stack_count.is_integer() else stack_count,
        'effects': buff_effects(instance, context),
        'display_as_line': display_as_line,
        'line_hidden_reason': '' if display_as_line else ('passive' if str(rule.get('duration', {}).get('type') or '') == 'permanent' else 'self_action'),
    }


def legacy_buff_rules_from_axis(raw_rules: Any) -> list[dict[str, Any]]:
    """Convert pre-registry hand-authored buff rules to the new runtime shape."""
    rules: list[dict[str, Any]] = []
    for index, raw_rule in enumerate(_as_list(raw_rules)):
        if not isinstance(raw_rule, dict):
            continue
        trigger = raw_rule.get('trigger') if isinstance(raw_rule.get('trigger'), dict) else {}
        targets = raw_rule.get('targets') if isinstance(raw_rule.get('targets'), dict) else {}
        modifiers = raw_rule.get('modifiers') if isinstance(raw_rule.get('modifiers'), dict) else {}
        duration_ticks = max(0, _int(raw_rule.get('duration_ticks')))
        if not modifiers or duration_ticks <= 0:
            continue
        trigger_slot = trigger.get('slot')
        has_trigger_slot = trigger_slot not in (None, '')
        target_slots = [_int(item) for item in _as_list(targets.get('slots'))]
        rule = {
            'id': str(raw_rule.get('id') or f'legacy_buff_{index + 1:03d}'),
            'name': str(raw_rule.get('name') or f'增益 {index + 1}'),
            'provider_kind': 'legacy',
            'provider_name': '手动增益',
            'owner_slot': _int(trigger_slot) if has_trigger_slot else -1,
            'priority': 10000 + index,
            'trigger': {
                'event': 'action_hit',
                'source': {
                    'scope': 'registrar' if has_trigger_slot else 'team',
                    'action_ids': [str(trigger.get('action_id'))] if trigger.get('action_id') else [],
                    'action_types': [str(trigger.get('action_type'))] if trigger.get('action_type') else [],
                },
            },
            'target': {
                'scope': 'slots' if target_slots else 'team',
                'slots': target_slots,
                'action_ids': [str(item) for item in _as_list(targets.get('action_ids'))],
                'action_types': [str(item) for item in _as_list(targets.get('action_types'))],
            },
            'duration': {
                'type': 'time',
                'ticks': duration_ticks,
                'delay_ticks': max(0, _int(raw_rule.get('delay_ticks'))),
                'loop_carry': False,
            },
            'stacking': {
                'mode': 'refresh',
                'max_stacks': 1,
            },
            'effects': {
                str(key): _num(value)
                for key, value in modifiers.items()
                if _num(value) != 0
            },
        }
        if rule['effects']:
            rules.append(rule)
    return rules
