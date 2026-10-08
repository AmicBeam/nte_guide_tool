"""GPU-deck effect IR. Card identity lives here; tensor kernels are lowering."""
from __future__ import annotations

from dataclasses import dataclass

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, STARTER_DECK, STARTER_DECKS

ALLOWED_OPS = frozenset({
    'sortie', 'form', 'perm_plus', 'count_seed', 'self_harmony',
    'draw_owned_self', 'draw_owned_target', 'draw_forms',
    'heal_target', 'buff_target', 'revive_target', 'heal_player',
    'inspect_top', 'pact_front', 'pact_target',
    'heal_self_full', 'heal_front_full', 'next_followup', 'next_bonus',
    'sacrifice_draw', 'damage_per_downed', 'set_down',
    'reveal_front_hand', 'damage_revealed',
    'ignore_shield', 'team_buff', 'pending_atk', 'share_overflow',
    'hit_front_scaled', 'self_damage', 'draw_one', 'sortie_allied_extra',
    'heal_after_hit',
    'burn_infinite', 'damage_missing_others', 'hp_floor',
})


@dataclass(frozen=True)
class EffectOp:
    op: str
    args: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.op not in ALLOWED_OPS:
            raise ValueError(f'unsupported effect op {self.op!r}')


@dataclass(frozen=True)
class CardProgram:
    card_id: str
    ops: tuple[EffectOp, ...]


def _appearing_ids() -> tuple[str, ...]:
    ids: list[str] = []
    for deck_id in ('starter', 'weave-rush'):
        preset = next(item for item in STARTER_DECKS if item['id'] == deck_id)
        ids.extend(preset['card_ids'])
    ids.append('NF01')
    for extra in (
        'N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07', 'N08',
        'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06', 'Y07', 'Y08',
        'Z01', 'Z02', 'Z03', 'Z04', 'Z05', 'Z06', 'Z07', 'Z08',
        'J01', 'J02', 'J03', 'J04', 'J05', 'J06', 'J07', 'J08',
        'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08',
        'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
    ):
        ids.append(extra)
    return tuple(dict.fromkeys(ids))


# Closed set for GPU: 创生 + 覆纹快攻 kits, plus NF01.
STARTER_APPEARING = _appearing_ids()

# Kits of the four starter seats. Extra kit cards are registered so a stray
# identity cannot silently no-op; the env still rejects non-starter decks.
_CARD_OPS: dict[str, tuple[EffectOp, ...]] = {
    'N01': (EffectOp('heal_self_full'),),
    'N02': (EffectOp('sortie'),),
    'N03': (EffectOp('sortie', (1,)),),
    'N04': (EffectOp('sortie'),),
    'N05': (EffectOp('next_followup', (3,)),),
    'N06': (EffectOp('revive_target'),),
    'N07': (EffectOp('form'),),
    'N08': (EffectOp('form'),),
    'NF01': (EffectOp('perm_plus', (1, 1)), EffectOp('count_seed'), EffectOp('self_harmony', (1,))),
    'Y01': (EffectOp('damage_per_downed', (3,)),),
    'Y02': (EffectOp('draw_owned_target'),),
    'Y03': (EffectOp('heal_target', (5,)), EffectOp('buff_target', (1,)), EffectOp('heal_player', (5,)),),
    'Y04': (EffectOp('next_bonus', (1, 2)),),
    'Y05': (EffectOp('sacrifice_draw', (2,)), EffectOp('perm_plus', (1, 1)),),
    'Y06': (EffectOp('revive_target'), EffectOp('buff_target', (1,)),),
    'Y07': (EffectOp('form'),),
    'Y08': (EffectOp('form'),),
    'Z01': (EffectOp('heal_front_full'),),
    'Z02': (EffectOp('draw_forms', (2,)),),
    'Z03': (EffectOp('sortie'),),
    'Z04': (EffectOp('sortie'),),
    'Z05': (EffectOp('sortie'),),
    'Z06': (EffectOp('form'),),
    'Z07': (EffectOp('form'),),
    'Z08': (EffectOp('form'),),
    'J01': (EffectOp('inspect_top', (3,)),),
    'J02': (EffectOp('pact_front', (1,)),),
    'J03': (EffectOp('set_down', (3,)),),
    'J04': (EffectOp('pact_target', (2,)),),
    'J05': (EffectOp('damage_revealed'),),
    'J06': (EffectOp('reveal_front_hand'),),
    'J07': (EffectOp('form'),),
    'J08': (EffectOp('form'),),
    'M01': (EffectOp('perm_plus', (1, 0)), EffectOp('sortie'),),
    'M02': (EffectOp('team_buff', (1,)),),
    'M03': (EffectOp('ignore_shield'), EffectOp('sortie'),),
    'M04': (EffectOp('share_overflow'), EffectOp('sortie'),),
    'M05': (EffectOp('pending_atk', (2,)),),
    'M06': (EffectOp('hit_front_scaled', (2, 3)),),
    'M07': (EffectOp('form'),),
    'M08': (EffectOp('form'),),
    'B01': (EffectOp('self_damage', (1,)), EffectOp('draw_one'),),
    'B02': (EffectOp('heal_after_hit', (2,)), EffectOp('sortie'),),
    'B03': (EffectOp('burn_infinite'),),
    'B04': (EffectOp('damage_missing_others'),),
    'B05': (EffectOp('hp_floor', (1,)),),
    'B06': (EffectOp('sortie_allied_extra', (1,)),),
    'B07': (EffectOp('form'),),
    'B08': (EffectOp('form'), EffectOp('self_damage', (4,)),),
}

STARTER_CARDS: tuple[CardProgram, ...] = tuple(
    CardProgram(card_id, _CARD_OPS[card_id]) for card_id in STARTER_APPEARING
)

STARTER_HOOKS = {
    'after_battle_harmony_seat': 'zero',
    'scale_shape': 'N07',
    'payoff_card': 'N03',
    'turn_end_atk_shape': 'Z08',
    'turn_end_atk': 1,
    'turn_end_heal_shape': 'Z07',
    'turn_end_heal': 2,
    'genesis_extra_seat': 'jiuyuan',
    'genesis_pact_shape': 'J08',
    'extra_genesis_shape': 'Y08',
    'seed_card': 'NF01',
    'seed_seat': 'nanali',
    'ult_energy_next_seat': 'iloy',
    'ult_pact_seat': 'jiuyuan',
    'overflow_seat': 'bohe',
    'stack_atk_seat': 'baicang',
    'auto_attack_shape': 'M08',
    'sleep_shape': 'M07',
}


def validate_starter_ir() -> None:
    missing = [card_id for card_id in STARTER_APPEARING if card_id not in _CARD_OPS]
    if missing:
        raise ValueError('starter IR missing ' + ','.join(missing))
    unknown = [card_id for card_id in STARTER_APPEARING if card_id not in CARDS]
    if unknown:
        raise ValueError('starter IR unknown catalog ids ' + ','.join(unknown))
    for program in STARTER_CARDS:
        if not program.ops:
            raise ValueError(f'empty IR for {program.card_id}')
        for op in program.ops:
            if op.op not in ALLOWED_OPS:
                raise ValueError(f'unsupported op {op.op} on {program.card_id}')
    for key in ('scale_shape', 'payoff_card', 'turn_end_atk_shape', 'turn_end_heal_shape',
                'genesis_pact_shape', 'extra_genesis_shape', 'seed_card',
                'auto_attack_shape', 'sleep_shape'):
        card_id = STARTER_HOOKS[key]
        if card_id not in CARDS:
            raise ValueError(f'hook {key} unknown card {card_id}')
    for key in ('after_battle_harmony_seat', 'genesis_extra_seat', 'seed_seat',
                'ult_energy_next_seat', 'ult_pact_seat', 'overflow_seat', 'stack_atk_seat'):
        if STARTER_HOOKS[key] not in CHARACTERS:
            raise ValueError(f'hook {key} unknown seat {STARTER_HOOKS[key]}')
