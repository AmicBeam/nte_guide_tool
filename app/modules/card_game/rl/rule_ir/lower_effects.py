"""Lower starter-card IR to integer tables consumed by the batched kernels."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from .starter import STARTER_CARDS, STARTER_HOOKS, validate_starter_ir

TABLE_KEYS = (
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
)


def compile_effect_tables(card_ids: tuple[str, ...]) -> dict[str, list[int]]:
    validate_starter_ir()
    index = {card_id: i for i, card_id in enumerate(card_ids)}
    n = len(card_ids)
    tables = {key: [0] * n for key in TABLE_KEYS}
    for program in STARTER_CARDS:
        if program.card_id not in index:
            continue
        row = index[program.card_id]
        for op in program.ops:
            if op.op == 'sortie':
                tables['sortie'][row] = 1
                tables['followup'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'form':
                tables['form'][row] = 1
            elif op.op == 'perm_plus':
                tables['perm_atk'][row] = op.args[0] if op.args else 1
                tables['perm_hp'][row] = op.args[1] if len(op.args) > 1 else 1
            elif op.op == 'self_harmony':
                tables['self_harmony'][row] = int(op.args[0])
            elif op.op == 'count_seed':
                tables['count_seed'][row] = 1
            elif op.op == 'draw_owned_self':
                tables['draw_owned_self'][row] = 1
            elif op.op == 'draw_owned_target':
                tables['draw_owned_target'][row] = 1
            elif op.op == 'draw_forms':
                tables['draw_forms'][row] = int(op.args[0]) if op.args else 1
            elif op.op == 'heal_target':
                tables['heal_target'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'buff_target':
                tables['buff_target'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'revive_target':
                tables['revive_target'][row] = 1
            elif op.op == 'heal_player':
                tables['heal_player'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'inspect_top':
                tables['inspect_top'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'pact_front':
                tables['pact_front'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'pact_target':
                tables['pact_target'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'heal_self_full':
                tables['heal_self_full'][row] = 1
            elif op.op == 'heal_front_full':
                tables['heal_front_full'][row] = 1
            elif op.op == 'next_followup':
                tables['next_followup'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'next_bonus':
                tables['next_bonus_atk'][row] = int(op.args[0]) if op.args else 0
                tables['next_bonus_sh'][row] = int(op.args[1]) if len(op.args) > 1 else 0
            elif op.op == 'sacrifice_draw':
                tables['sacrifice_draw'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'damage_per_downed':
                tables['damage_per_downed'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'set_down':
                tables['set_down'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'reveal_front_hand':
                tables['reveal_front_hand'][row] = 1
            elif op.op == 'damage_revealed':
                tables['damage_revealed'][row] = 1
            elif op.op == 'ignore_shield':
                tables['ignore_shield'][row] = 1
            elif op.op == 'team_buff':
                tables['team_buff'][row] = int(op.args[0]) if op.args else 1
            elif op.op == 'pending_atk':
                tables['pending_atk'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'share_overflow':
                tables['share_overflow'][row] = 1
            elif op.op == 'hit_front_scaled':
                tables['hit_front_base'][row] = int(op.args[0]) if op.args else 0
                tables['hit_front_printed'][row] = int(op.args[1]) if len(op.args) > 1 else 0
            elif op.op == 'self_damage':
                tables['self_damage'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'draw_one':
                tables['draw_one'][row] = 1
            elif op.op == 'sortie_allied_extra':
                tables['sortie_allied_extra'][row] = int(op.args[0]) if op.args else 0
                tables['sortie'][row] = 1
            elif op.op == 'heal_after_hit':
                tables['heal_after_hit'][row] = int(op.args[0]) if op.args else 0
            elif op.op == 'burn_infinite':
                tables['burn_infinite'][row] = 1
            elif op.op == 'damage_missing_others':
                tables['damage_missing_others'][row] = 1
            elif op.op == 'hp_floor':
                tables['hp_floor'][row] = int(op.args[0]) if op.args else 1
            else:
                raise ValueError(f'no lowering for {op.op}')
    tables['hooks'] = dict(STARTER_HOOKS)  # type: ignore[assignment]
    return tables


@lru_cache(maxsize=4)
def gpu_tables(device_str: str) -> dict[str, Any]:
    import torch
    from app.modules.card_game.rl.gpu_duel.catalog import CARD_IDS

    raw = compile_effect_tables(CARD_IDS)
    device = torch.device(device_str)
    out: dict[str, Any] = {'hooks': raw['hooks']}
    for key in TABLE_KEYS:
        out[key] = torch.tensor(raw[key], dtype=torch.int32, device=device)
    return out
