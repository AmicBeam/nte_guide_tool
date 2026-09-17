"""Startup completeness for the compiled starter engine.

A card or hook is supported only when content declares it, IR names it, tables
lower it, and the generated engine actually implements the handler. A name on a
list is not enough.
"""
from __future__ import annotations

from typing import Any

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, STARTER_DECK, STARTER_DECKS
from app.modules.card_game.content.duel_v2.characters import EFFECTS, TARGET_POLICIES
from app.modules.card_game.content.duel_v2.registry import KITS
from app.modules.card_game.rl.gpu_duel.catalog import CARD_IDS, CARD_INDEX, SEATS, public_card_ids
from app.modules.card_game.rl.rule_ir.lower_effects import TABLE_KEYS, compile_effect_tables
from app.modules.card_game.rl.rule_ir.starter import ALLOWED_OPS, STARTER_CARDS, STARTER_HOOKS, validate_starter_ir

COMPILED_DECK_ID = 'starter'
COMPILED_DECK_NAME = '创生预组'
COMPILED_CHARACTERS = tuple(STARTER_DECK['character_ids'])
COMPILED_CARDS = public_card_ids()

# Ops the generated engine has a real handler for. Extra IR ops used only by
# other GPU presets stay on the tensor path.
IMPLEMENTED_OPS = frozenset({
    'sortie', 'form', 'perm_plus', 'count_seed', 'self_harmony',
    'draw_owned_self', 'draw_owned_target', 'draw_forms',
    'heal_target', 'buff_target', 'revive_target', 'heal_player',
    'inspect_top', 'pact_front', 'pact_target',
    'heal_self_full', 'heal_front_full', 'next_followup', 'next_bonus',
    'sacrifice_draw', 'damage_per_downed', 'set_down',
    'reveal_front_hand', 'damage_revealed',
    'ignore_shield', 'team_buff', 'pending_atk', 'share_overflow',
    'hit_front_scaled', 'self_damage', 'draw_one',
    'sortie_allied_extra', 'heal_after_hit',
    'burn_infinite', 'damage_missing_others', 'hp_floor',
})

IMPLEMENTED_TABLES = frozenset({
    'sortie', 'followup', 'form', 'perm_atk', 'perm_hp', 'count_seed', 'self_harmony',
    'draw_owned_self', 'draw_owned_target', 'draw_forms',
    'heal_target', 'buff_target', 'revive_target', 'heal_player',
    'inspect_top', 'pact_front', 'pact_target',
    'heal_self_full', 'heal_front_full', 'next_followup', 'next_bonus_atk', 'next_bonus_sh',
    'sacrifice_draw', 'damage_per_downed', 'set_down',
    'reveal_front_hand', 'damage_revealed',
    'ignore_shield', 'team_buff', 'pending_atk', 'share_overflow',
    'hit_front_base', 'hit_front_printed', 'self_damage', 'draw_one',
    'sortie_allied_extra', 'heal_after_hit',
    'burn_infinite', 'damage_missing_others', 'hp_floor',
})

# Character hooks that can fire with the frozen starter kit. Names must match
# registry timings actually invoked by the generated engine.
IMPLEMENTED_HOOKS = frozenset({
    ('nanali', 'on_turn_begin'),
    ('nanali', 'on_rout'),
    ('nanali', 'after_sortie'),
    ('nanali', 'makes_instant'),
    ('nanali', 'scale_shape'),
    ('nanali', 'payoff_card'),
    ('nanali', 'seed_card'),
    ('iloy', 'after_genesis'),
    ('iloy', 'on_ultimate'),
    ('iloy', 'on_turn_begin'),
    ('iloy', 'on_ally_down'),
    ('bohe', 'combat_overflow'),
    ('bohe', 'incoming_damage'),
    ('bohe', 'after_hits'),
    ('bohe', 'on_turn_begin'),
    ('bohe', 'on_enemy_turn_begin'),
    ('baicang', 'on_damage_taken'),
    ('baicang', 'on_damage_dealt'),
    ('baicang', 'after_hits'),
    ('baicang', 'on_turn_begin'),
    ('zero', 'after_card'),
    ('zero', 'on_turn_end'),
    ('zero', 'shaped_forms_instant'),
    ('zero', 'ultimate_forms_instant'),
    ('jiuyuan', 'genesis_extra'),
    ('jiuyuan', 'on_genesis_damage'),
    ('jiuyuan', 'on_ultimate'),
})

# Starter-relevant kit attributes that are data, not callables.
HOOK_DATA_ATTRS = {
    'nanali': ('seed_card', 'seed_key', 'seed_play_key', 'scale_shape', 'payoff_card', 'followup_shape'),
    'iloy': (),
    'zero': ('shaped_forms_instant', 'ultimate_forms_instant'),
    'jiuyuan': (),
}

STARTER_CALL_HOOKS = {
    'nanali': ('on_turn_begin', 'on_rout', 'after_sortie', 'makes_instant'),
    'iloy': ('after_genesis', 'on_ultimate', 'on_turn_begin', 'on_ally_down'),
    'bohe': ('combat_overflow', 'incoming_damage', 'after_hits', 'on_turn_begin', 'on_enemy_turn_begin'),
    'baicang': ('on_damage_taken', 'on_damage_dealt', 'after_hits', 'on_turn_begin'),
    'zero': ('after_card', 'on_turn_end'),
    'jiuyuan': ('genesis_extra', 'on_genesis_damage', 'on_ultimate'),
}


class CompletenessError(ValueError):
    pass


def _card_program_map() -> dict[str, Any]:
    return {program.card_id: program for program in STARTER_CARDS}


def compiled_rule_hash(deck_id='starter') -> str:
    from hashlib import sha256
    from .layout import ENGINE_VERSION, ROW_WIDTH
    payload = '|'.join((
        ENGINE_VERSION,
        deck_id,
        ','.join(COMPILED_CHARACTERS),
        ','.join(COMPILED_CARDS),
        'public_compiled',
        ','.join(sorted(IMPLEMENTED_OPS)),
        str(ROW_WIDTH),
        ','.join(CARD_IDS),
    ))
    from pathlib import Path
    digest = sha256(payload.encode('utf-8'))
    for name in ('emit_engine.py', 'layout.py', 'starter.py', 'lower_effects.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK
    digest.update(GPU_LOCK.encode())
    return digest.hexdigest()


def _deck(deck_id):
    if deck_id not in ('starter', 'weave-rush'):
        raise CompletenessError(f'Unsupported compiled deck: {deck_id}')
    return next(d for d in STARTER_DECKS if d['id'] == deck_id)


def support_manifest(deck_id='starter') -> dict[str, Any]:
    deck = _deck(deck_id)
    cards = list(public_card_ids())
    return {
        'deck_id': deck_id,
        'deck_name': deck['name'],
        'characters': list(deck['character_ids']),
        'cards': cards,
        'implemented_ops': sorted(IMPLEMENTED_OPS),
        'implemented_hooks': sorted(f'{cid}.{hook}' for cid, hook in IMPLEMENTED_HOOKS),
        'rule_hash': compiled_rule_hash(deck_id),
        'engine_version': __import__('app.modules.card_game.rl.rule_ir.layout', fromlist=['ENGINE_VERSION']).ENGINE_VERSION,
    }


def validate_compiled_deck(deck_id="starter") -> dict[str, Any]:
    """Fail closed: missing lowering or missing engine handler is an error."""
    validate_starter_ir()
    deck = _deck(deck_id)
    manifest = support_manifest(deck_id)
    cards = manifest['cards']
    characters = deck['character_ids']
    errors: list[str] = []
    programs = _card_program_map()
    tables = compile_effect_tables(CARD_IDS)

    if STARTER_DECK.get('id') != COMPILED_DECK_ID:
        errors.append(f'compiled deck id drifted: {STARTER_DECK.get("id")!r}')
    for character_id in characters:
        if character_id not in CHARACTERS or character_id not in KITS:
            errors.append(f'missing starter character {character_id}')
    for card_id in cards:
        if card_id not in CARDS:
            errors.append(f'missing catalog card {card_id}')
            continue
        if card_id not in EFFECTS:
            errors.append(f'content effect missing for {card_id}')
        if card_id not in programs:
            errors.append(f'IR missing for {card_id}')
            continue
        if card_id not in CARD_INDEX:
            errors.append(f'GPU catalog cannot encode {card_id}')
            continue
        row = CARD_INDEX[card_id]
        for op in programs[card_id].ops:
            if op.op not in ALLOWED_OPS:
                errors.append(f'{card_id} IR op {op.op} is not allowed')
            if op.op not in IMPLEMENTED_OPS:
                errors.append(f'{card_id} op {op.op} has no compiled handler')
        # Table lowering must materialize at least one implemented table for this card.
        lowered = False
        for key in IMPLEMENTED_TABLES:
            if key in tables and int(tables[key][row]) != 0:
                lowered = True
                break
        if not lowered:
            errors.append(f'{card_id} IR did not lower to any compiled table')

    for character_id, hooks in STARTER_CALL_HOOKS.items():
        if character_id not in characters:
            continue
        kit = KITS.get(character_id)
        if kit is None:
            errors.append(f'kit missing {character_id}')
            continue
        for hook in hooks:
            if (character_id, hook) not in IMPLEMENTED_HOOKS:
                errors.append(f'{character_id}.{hook} is reachable but not compiled')
            if hook in HOOK_DATA_ATTRS.get(character_id, ()):
                if getattr(kit, hook, None) in (None, ''):
                    errors.append(f'{character_id}.{hook} data missing')
            elif getattr(kit, hook, None) is None:
                errors.append(f'{character_id}.{hook} content handler missing')
        for attr in HOOK_DATA_ATTRS.get(character_id, ()):
            if (character_id, attr) not in IMPLEMENTED_HOOKS and attr in ('scale_shape', 'payoff_card', 'seed_card'):
                errors.append(f'{character_id}.{attr} not marked compiled')
            if getattr(kit, attr, None) in (None, ''):
                errors.append(f'{character_id}.{attr} missing')

    # Starter cards must not require a target policy the engine cannot encode.
    from app.modules.card_game.rl.gpu_duel.catalog import _POLICY
    for card_id in cards:
        policy = TARGET_POLICIES.get(card_id)
        if policy and policy not in _POLICY:
            errors.append(f'{card_id} target policy {policy} is not compiled')

    unknown_table = [key for key in IMPLEMENTED_TABLES if key not in TABLE_KEYS]
    if unknown_table:
        errors.append('compiled tables missing from lowering: ' + ','.join(unknown_table))

    # IR hooks referenced by the engine must exist.
    for key in ('seed_card', 'seed_seat', 'scale_shape', 'payoff_card',
                'after_battle_harmony_seat', 'genesis_extra_seat', 'genesis_pact_shape',
                'extra_genesis_shape', 'turn_end_atk_shape', 'turn_end_heal_shape',
                'ult_energy_next_seat', 'ult_pact_seat'):
        if key not in STARTER_HOOKS:
            errors.append(f'starter hook table missing {key}')

    from pathlib import Path as _Path
    engine_src = _Path(__file__).with_name('emit_engine.py').read_text(encoding='utf-8')
    required_handlers = (
        'step_one', 'rebuild_legal', 'begin_turn', 'end_turn', 'play_card',
        'choose_pending', 'start_ultimate', 'sortie', 'attack_once',
        'trigger_response', 'resolve_genesis', 'knockdown', 'draw_one',
        'grant_combat_resources', 'starter_reset', 'reset_one',
        'public_reset', 'begin_escalation', 'reset_one_with_decks',
    )
    for name in required_handlers:
        if f'static void {name}(' not in engine_src and f'void {name}(' not in engine_src:
            errors.append(f'generated engine missing handler {name}')

    if errors:
        raise CompletenessError('compiled starter incomplete: ' + '; '.join(errors))
    manifest = support_manifest(deck_id)
    manifest['ok'] = True
    manifest['card_count'] = len(cards)
    manifest['character_count'] = len(characters)
    return manifest


def validate_compiled_starter():
    return validate_compiled_deck("starter")
