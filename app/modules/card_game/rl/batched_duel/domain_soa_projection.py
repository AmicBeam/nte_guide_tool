"""Direct named-field public observation and simple legal-action foundation.

Reads stable domain_soa_schema tensors on device. Frozen catalog tables are
prepared outside capture. This is not a VM, native SoA graph, object-graph
walker, or CPU rule callback, and it does not claim the seven 3479-column
observation. Hidden opponent card identities, RNG and deck order never enter
the public tensors. Unhandled action categories are refused rather than omitted.
supports_full_duel_rules stays False.
"""
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType

from app.modules.card_game.content.duel_v2.catalog import CHARACTERS
from app.modules.card_game.rl import seven_lineup
from app.modules.card_game.rl.league_schema import CHAR_FIELDS, OLD_CARDS, SIDE_FIELDS
from .domain_soa_schema import (
    CARD_BOOL_FIELDS,
    CARD_VOCAB,
    CHARACTER_BOOL_FIELDS,
    CHARACTER_INT_FIELDS,
    CHARACTER_VOCAB,
    CHOICE_KINDS,
    DomainSoALayout,
    GAME_PHASES,
    PLAYER_BOOL_FIELDS,
    PLAYER_INT_FIELDS,
    PRESENCE_CATALOG,
    PRESENCE_VALUE,
    array_specs,
    supports_full_duel_rules as _SCHEMA_FULL,
)

INVALID_ROW = 1
COUNT_OVERFLOW = 2
UNHANDLED_ACTION_CATEGORY = 4
UNKNOWN_TEMPLATE = 8
FULL_ACTION_CAPACITY = 800
SEARCH_CANDIDATE_CAP = 16
MAX_HAND_COMBOS = 16
MAX_MULLIGAN_SELECT = 3
TOKEN_WIDTH = 7
supports_full_duel_rules = False
if _SCHEMA_FULL:
    raise RuntimeError('domain_soa_schema must not claim full duel coverage')

ACTION_END_TURN, ACTION_ATTACK, ACTION_ULTIMATE, ACTION_PLAY_CARD = 0, 1, 2, 3
ACTION_CHOOSE, ACTION_MULLIGAN, ACTION_CHOOSE_DECK_ORDER = 4, 5, 6
ACTION_EXIT_ULTIMATE, ACTION_CHOOSE_CARDS, ACTION_CONCEDE = 7, 8, 9
SUPPORTED_ACTIONS = (ACTION_CONCEDE, ACTION_MULLIGAN)
REFUSED_ACTIONS = (
    ACTION_END_TURN, ACTION_ATTACK, ACTION_ULTIMATE, ACTION_PLAY_CARD,
    ACTION_CHOOSE, ACTION_CHOOSE_DECK_ORDER, ACTION_EXIT_ULTIMATE, ACTION_CHOOSE_CARDS,
)
PHASE_MULLIGAN, PHASE_PLAYING, PHASE_CHOICE, PHASE_FINISHED = range(4)
ZONE_HAND, ZONE_DECK, ZONE_DISCARD = 0, 1, 2
SEATS = seven_lineup.SEATS
NUM_SEATS = len(SEATS)
CARD_IDS = seven_lineup.CARD_IDS
NUM_CARDS = len(CARD_IDS)
NUM_OLD_CARDS = len(OLD_CARDS)
SUMMON_SEAT = seven_lineup.SEAT_INDEX['summon']
CHOICES = seven_lineup.CHOICES
NEW_SIDE = seven_lineup.NEW_SIDE
PINT = {name: i for i, name in enumerate(PLAYER_INT_FIELDS)}
PBOOL = {name: i for i, name in enumerate(PLAYER_BOOL_FIELDS)}
CINT = {name: i for i, name in enumerate(CHARACTER_INT_FIELDS)}
CBOOL = {name: i for i, name in enumerate(CHARACTER_BOOL_FIELDS)}
KBOOL = {name: i for i, name in enumerate(CARD_BOOL_FIELDS)}
HEADER_CHOICE_CODE = {'inspect_top': 1, 'discard': 2, 'enemy_hand': 3}
IMPLEMENTED_SIDE = (
    'hp', 'shield', 'ap', 'front', 'last_front', 'turn_count', 'normal_atk',
    'ultimate_ok', 'extra_genesis', 'nanali_seed', 'nanali_seed_played',
    'hand_n', 'deck_n', 'discard_n', 'fatigue', 'extra_ap', 'surplus',
    'harmony_damage', 'genesis_damage_bonus',
)
IMPLEMENTED_CHAR = (
    'ch_present', 'ch_hp', 'ch_max_hp', 'ch_shield', 'ch_harmony', 'ch_energy',
    'ch_down', 'ch_awakened', 'ch_ult_turns', 'ch_growth', 'ch_energy_max',
    'ch_order', 'ch_jingu', 'ch_jingu_loan',
)
IMPLEMENTED_CARD_ZONES = (0, 1, 2, 3)
PUBLIC_FLAG_NAMES = (
    'observation_public', 'actions_complete', 'viewer_mulligan_done', 'finished',
)
REQUIRED_FIELDS = (
    'row_valid', 'phase', 'turn', 'active_side', 'first_side', 'escalation_enabled',
    'winner_presence', 'pending_choice_present', 'choice_kind', 'choice_side',
    'choice_card_count', 'choice_card_index', 'operation_present', 'op_card_index',
    'mulligan_count', 'mulligan_sides', 'player_valid', 'player_hp', 'player_max_hp',
    'player_shield', 'player_ap', 'player_front_presence', 'player_front_slot',
    'player_last_front_presence', 'player_last_front_slot', 'player_int_presence',
    'player_int_value', 'player_bool_presence', 'player_bool_value',
    'character_valid', 'character_template', 'character_hp', 'character_shield',
    'character_energy', 'character_int_presence', 'character_int_value',
    'character_bool_presence', 'character_bool_value', 'zone_count',
    'zone_card_index', 'card_valid', 'card_template', 'card_bool_presence',
    'card_bool_value', 'rng',
)
_TORCH_DTYPES = {
    'bool': 'bool', 'int8': 'int8', 'int16': 'int16', 'int32': 'int32',
    'int64': 'int64', 'uint8': 'uint8', 'float64': 'float64',
}


def _implemented_feature_names():
    names = ['turn', 'first', 'escalation', 'phase', 'choice_kind', 'resolving', 'choice_count']
    names += [f'{key}:{side}' for key in SIDE_FIELDS for side in (0, 1) if key in IMPLEMENTED_SIDE]
    names += [
        f'{key}:{side}:{cid}'
        for key in CHAR_FIELDS for side in (0, 1) for cid in SEATS
        if key in IMPLEMENTED_CHAR
    ]
    names += [f'cards:{zone}:{cid}' for zone in IMPLEMENTED_CARD_ZONES for cid in CARD_IDS]
    names += [f'seven:{key}:{side}' for side in (0, 1) for key in NEW_SIDE]
    names += [f'seven:choice:{kind}' for kind in CHOICES]
    names += [f'seven:choice_card:{cid}' for cid in CARD_IDS]
    return tuple(names)


IMPLEMENTED_FEATURE_NAMES = _implemented_feature_names()
_FEATURE_INDEX = {name: i for i, name in enumerate(seven_lineup.all_features())}
IMPLEMENTED_FEATURE_INDEX = tuple(_FEATURE_INDEX[name] for name in IMPLEMENTED_FEATURE_NAMES)


def coverage():
    return MappingProxyType({
        'supports_full_duel_rules': False,
        'full_3479_observation': False,
        'observation_dim_claimed': len(IMPLEMENTED_FEATURE_NAMES),
        'seven_observation_dim': len(seven_lineup.all_features()),
        'implemented_features': IMPLEMENTED_FEATURE_NAMES,
        'supported_actions': SUPPORTED_ACTIONS,
        'refused_actions': REFUSED_ACTIONS,
        'full_action_capacity': FULL_ACTION_CAPACITY,
        'search_candidate_cap': SEARCH_CANDIDATE_CAP,
        'max_mulligan_hand': MAX_HAND_COMBOS,
        'hidden_excluded': ('opponent hidden hand identities', 'rng', 'deck order', 'deck identities'),
        'remaining': (
            'attack/play/ultimate/choice/end_turn complete legality',
            'kit fire/decide and targets',
            'full 3479 seven_lineup observation',
            'resistances, dots, timed attacks, marks, substituted faces',
        ),
    })


def _combo_table(max_hand=MAX_HAND_COMBOS, max_select=MAX_MULLIGAN_SELECT):
    rows = []
    for size in range(max_select + 1):
        for group in combinations(range(max_hand), size):
            rows.append(group + (-1,) * (max_select - size))
    return tuple(rows)


MULLIGAN_COMBOS = _combo_table()


def prepare_catalog_tables(device, torch_module=None):
    """Frozen public-17 -> seven seat/card tables. Call outside capture."""
    t = torch_module
    if t is None:
        import torch as t
    device = t.device(device)
    n_char = max(CHARACTER_VOCAB.values()) + 1
    char_to_seat = t.full((n_char,), -1, dtype=t.int32, device=device)
    card_to_seven = t.full((max(CARD_VOCAB.values()) + 1,), -1, dtype=t.int32, device=device)
    char_max_hp = t.zeros(n_char, dtype=t.int32, device=device)
    char_energy_max = t.zeros(n_char, dtype=t.int32, device=device)
    char_attack = t.zeros(n_char, dtype=t.int32, device=device)
    char_int_defaults = t.zeros((n_char, len(CINT)), dtype=t.int32, device=device)
    for cid, vocab in CHARACTER_VOCAB.items():
        char_to_seat[vocab] = seven_lineup.SEAT_INDEX[cid]
        spec = CHARACTERS[cid]
        char_max_hp[vocab] = int(spec.get('max_hp') or 0)
        char_energy_max[vocab] = int(spec.get('energy_max') or 0)
        char_attack[vocab] = int(spec.get('attack') or 0)
        for name, index in CINT.items():
            value = spec.get(name)
            char_int_defaults[vocab, index] = value if type(value) is int else 0
    for card_id, vocab in CARD_VOCAB.items():
        card_to_seven[vocab] = seven_lineup.CARD_INDEX[card_id]
    header_choice = t.full((len(CHOICE_KINDS),), -1, dtype=t.int32, device=device)
    seven_choice = t.full((len(CHOICE_KINDS),), -1, dtype=t.int32, device=device)
    for i, kind in enumerate(CHOICE_KINDS):
        if kind in HEADER_CHOICE_CODE:
            header_choice[i] = HEADER_CHOICE_CODE[kind]
        if kind in CHOICES:
            seven_choice[i] = CHOICES.index(kind)
        elif kind == 'deck_order':
            seven_choice[i] = CHOICES.index('hand_deck_placement')
    return FrozenCatalogTables(
        character_to_seat=char_to_seat, card_to_seven=card_to_seven,
        header_choice=header_choice, seven_choice=seven_choice,
        character_int_defaults=char_int_defaults, device=device)


@dataclass(frozen=True)
class FrozenCatalogTables:
    character_to_seat: object
    character_int_defaults: object
    card_to_seven: object
    header_choice: object
    seven_choice: object
    device: object


@dataclass(frozen=True)
class DomainPublicProjection:
    named: object
    packed_named: object
    packed_names: tuple
    packed_index: tuple
    actors: object
    tokens: object
    mask: object
    counts: object
    mulligan_ordinals: object
    public_flags: object
    errors: object
    active: object


def cpu_oracle(layout=None, **kwargs):
    kwargs.pop('device', None)
    kwargs.pop('allow_cpu_oracle', None)
    return DomainSoAProjection(layout=layout, device='cpu', allow_cpu_oracle=True, **kwargs)


class DomainSoAProjection:
    supports_full_duel_rules = False

    def __init__(self, layout=None, catalog=None, *, max_actions=FULL_ACTION_CAPACITY,
                 device='cuda', allow_cpu_oracle=False):
        import torch
        if layout is None:
            layout = DomainSoALayout()
        if not isinstance(layout, DomainSoALayout):
            raise TypeError('DomainSoALayout required')
        if type(max_actions) is not int or not 1 <= max_actions <= FULL_ACTION_CAPACITY:
            raise ValueError('Full legal-action capacity must be in 1..800')
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Actual CUDA required; no CPU projection fallback')
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
        elif not (self.device.type == 'cpu' and allow_cpu_oracle is True):
            raise ValueError('CPU is an explicit projection oracle only')
        if catalog is None:
            catalog = prepare_catalog_tables(self.device, torch)
        if not isinstance(catalog, FrozenCatalogTables) or catalog.device != self.device:
            raise ValueError('Frozen catalog tables must already reside on the projection device')
        self.layout = layout
        self.catalog = catalog
        self.max_actions = max_actions
        self.combo_index = torch.tensor(MULLIGAN_COMBOS, dtype=torch.int32, device=self.device)
        pad = torch.full(
            (self.combo_index.shape[0], SEARCH_CANDIDATE_CAP - MAX_MULLIGAN_SELECT),
            -1, dtype=torch.int32, device=self.device)
        self.combo_select = torch.cat((self.combo_index, pad), dim=1)
        self.n_combos = int(self.combo_index.shape[0])
        self.specs = array_specs(layout)

    def _fields(self, fields):
        if hasattr(fields, 'arrays') and isinstance(fields.arrays, dict):
            fields = fields.arrays
        if not isinstance(fields, dict):
            raise TypeError('Named domain SoA field mapping required')
        missing = [name for name in REQUIRED_FIELDS if name not in fields]
        if missing:
            raise TypeError('Missing domain SoA fields: ' + ','.join(missing))
        return fields

    def _tensor(self, value, dtype, shape, name):
        t = self.t
        if (not isinstance(value, t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != tuple(shape)):
            raise ValueError('Resident ' + name + ' required')
        return value

    def _torch_dtype(self, np_dtype):
        name = getattr(np_dtype, '__name__', None) or getattr(np_dtype, 'name', str(np_dtype))
        name = str(name).replace('_', '')
        if name == 'bool':
            return self.t.bool
        return getattr(self.t, _TORCH_DTYPES[name])

    def _column(self, fields, name, rows):
        dtype, tail = self.specs[name]
        return self._tensor(fields[name], self._torch_dtype(dtype), (rows,) + tuple(tail), name)

    def _ratio(self, value, denominator):
        return (value.to(dtype=self.t.float64) / float(denominator)).to(dtype=self.t.float32)

    def _gather_sides(self, value, side_index):
        idx = side_index
        while idx.ndim < value.ndim:
            idx = idx.unsqueeze(-1)
        return value.gather(1, idx.expand_as(value))

    def _optional_int(self, presence, values, field):
        take = presence[..., field] == PRESENCE_VALUE
        return self.t.where(take, values[..., field], self.t.zeros_like(values[..., field]))

    def _optional_bool(self, presence, values, field):
        return values[..., field] & (presence[..., field] == PRESENCE_VALUE)

    def _scatter_classes(self, indices, valid, n_classes):
        t = self.t
        safe = indices.clamp(0, n_classes - 1).long()
        one = t.nn.functional.one_hot(safe, n_classes)
        return (valid.unsqueeze(-1) & (indices >= 0).unsqueeze(-1) & one.bool()).sum(-2)

    def project(self, fields, viewers, active, *, errors=None):
        t = self.t
        fields = self._fields(fields)
        if (not isinstance(viewers, t.Tensor) or viewers.device != self.device
                or viewers.dtype != t.int64 or viewers.ndim != 1):
            raise ValueError('Resident int64 viewers required')
        rows = int(viewers.shape[0])
        if not 0 <= rows <= self.layout.game_capacity:
            raise ValueError('Viewer batch exceeds layout.game_capacity')
        active = self._tensor(active, t.bool, (rows,), 'active')
        if errors is None:
            errors = t.zeros(rows, dtype=t.int32, device=self.device)
        else:
            errors = self._tensor(errors.clone(), t.int32, (rows,), 'errors')
        col = {name: self._column(fields, name, rows) for name in REQUIRED_FIELDS}
        hidden_rng = col['rng']
        C = self.layout.max_characters
        K = self.layout.max_cards
        Z = self.layout.max_zone
        CH = self.layout.max_choice_cards
        batch = t.arange(rows, device=self.device)
        errors = errors | (active & ((viewers < 0) | (viewers > 1) | ~col['row_valid']
                                     | ~col['player_valid'].all(1))).to(t.int32) * INVALID_ROW
        own = viewers.clamp(0, 1)
        side_index = t.stack((own, 1 - own), dim=1)
        phase = col['phase'].to(t.int32)
        finished = (phase == PHASE_FINISHED) | (col['winner_presence'].to(t.int32) == PRESENCE_VALUE)
        pending = col['pending_choice_present']
        unknown_phase = (phase < 0) | (phase >= len(GAME_PHASES))
        unhandled = active & ~finished & (
            unknown_phase | (phase == PHASE_CHOICE)
            | ((phase == PHASE_PLAYING) & (own == col['active_side'].to(t.int64)))
            | ((phase == PHASE_MULLIGAN) & (pending | col['operation_present']))
        )
        errors = errors | unhandled.to(t.int32) * UNHANDLED_ACTION_CATEGORY
        char_valid = col['character_valid']
        char_tmpl = col['character_template'].to(t.int64)
        char_map = self.catalog.character_to_seat
        in_char = (char_tmpl >= 0) & (char_tmpl < char_map.shape[0])
        seats = t.where(in_char, char_map[char_tmpl.clamp(0, char_map.shape[0] - 1)].to(t.int64),
                        t.full_like(char_tmpl, -1))
        summoned = self._optional_bool(
            col['character_bool_presence'], col['character_bool_value'], CBOOL['summoned'])
        seats = t.where(char_valid & summoned, t.full_like(seats, SUMMON_SEAT), seats)
        errors = errors | (active.view(rows, 1, 1) & char_valid & ((seats < 0) | ~in_char)
                           ).flatten(1).any(1).to(t.int32) * UNKNOWN_TEMPLATE
        seat_ok = char_valid & (seats >= 0) & (seats < NUM_SEATS)
        one_seat = t.nn.functional.one_hot(seats.clamp(0, NUM_SEATS - 1), NUM_SEATS).bool()
        seat_hits = (seat_ok.unsqueeze(-1) & one_seat).sum(2)
        errors = errors | (active & (seat_hits > 1).flatten(1).any(1)).to(t.int32) * INVALID_ROW

        def scatter_int(values, default=0):
            fill = t.full((rows, 2, NUM_SEATS), default, dtype=t.int32, device=self.device)
            packed = (values.to(t.int32).unsqueeze(-1) * seat_ok.to(t.int32).unsqueeze(-1)
                      * one_seat.to(t.int32)).sum(2)
            return t.where(seat_hits > 0, packed, fill)

        def scatter_bool(values):
            return (values.unsqueeze(-1) & seat_ok.unsqueeze(-1) & one_seat).any(2)

        cint_p, cint_v = col['character_int_presence'], col['character_int_value']
        cbool_p, cbool_v = col['character_bool_presence'], col['character_bool_value']
        char_hp = scatter_int(col['character_hp'])
        char_shield = scatter_int(col['character_shield'])
        char_energy = scatter_int(col['character_energy'])
        char_present = scatter_bool(char_valid)
        def character_int(name):
            index = CINT[name]
            raw = self._optional_int(cint_p, cint_v, index)
            defaults = self.catalog.character_int_defaults[char_tmpl.clamp(0, char_map.shape[0] - 1), index]
            return t.where(cint_p[..., index] == PRESENCE_CATALOG, defaults, raw)

        char_max_hp = scatter_int(character_int('max_hp'))
        char_energy_max = scatter_int(character_int('energy_max'))
        char_harmony = scatter_int(character_int('harmony'))
        char_down = scatter_int(character_int('down_turns'))
        char_ult = scatter_int(character_int('ultimate_turns'))
        char_growth = scatter_int(character_int('growth'))
        char_jingu = scatter_int(character_int('jingu'))
        char_awakened = scatter_bool(self._optional_bool(cbool_p, cbool_v, CBOOL['awakened']))
        char_jingu_loan = scatter_bool(self._optional_bool(cbool_p, cbool_v, CBOOL['jingu_loan']))
        order_slots = t.arange(C, device=self.device, dtype=t.int32).view(1, 1, C).expand(rows, 2, C)
        char_order = t.where(char_present, scatter_int(order_slots, default=-1),
                             t.full((rows, 2, NUM_SEATS), -1, dtype=t.int32, device=self.device))

        def occupant_seat(presence, slot):
            present = presence.to(t.int32) == PRESENCE_VALUE
            safe = slot.to(t.int64).clamp(0, max(C - 1, 0))
            gathered = char_tmpl.gather(2, safe.unsqueeze(-1)).squeeze(-1)
            in_map = (gathered >= 0) & (gathered < char_map.shape[0])
            mapped = t.where(in_map, char_map[gathered.clamp(0, char_map.shape[0] - 1)].to(t.int64),
                             t.full_like(gathered, -1))
            occupied = char_valid.gather(2, safe.unsqueeze(-1)).squeeze(-1)
            bad = present & ((slot < 0) | (slot >= C) | (mapped < 0) | ~occupied)
            return t.where(present, mapped, t.full_like(mapped, -1)), bad

        front, bad_front = occupant_seat(col['player_front_presence'], col['player_front_slot'])
        last_front, bad_last = occupant_seat(col['player_last_front_presence'], col['player_last_front_slot'])
        errors = errors | (active & (bad_front | bad_last).any(1)).to(t.int32) * INVALID_ROW
        pint_p, pint_v = col['player_int_presence'], col['player_int_value']
        pbool_p, pbool_v = col['player_bool_presence'], col['player_bool_value']
        zone_count = col['zone_count'].to(t.int32)
        hand_n = zone_count[:, :, ZONE_HAND]
        deck_n = zone_count[:, :, ZONE_DECK]
        discard_n = zone_count[:, :, ZONE_DISCARD]
        card_map = self.catalog.card_to_seven
        card_tmpl = col['card_template'].to(t.int64)
        in_card = (card_tmpl >= 0) & (card_tmpl < card_map.shape[0])
        card_seven = t.where(in_card, card_map[card_tmpl.clamp(0, card_map.shape[0] - 1)].to(t.int64),
                             t.full_like(card_tmpl, -1))
        copy_flag = self._optional_bool(col['card_bool_presence'], col['card_bool_value'], KBOOL['copy'])

        def zone_kinds(side_idx, zone, public_mask=None):
            count = zone_count[batch, side_idx, zone]
            pos = t.arange(Z, device=self.device).unsqueeze(0).expand(rows, Z)
            in_zone = pos < count.unsqueeze(1).clamp(min=0, max=Z)
            index = col['zone_card_index'][batch, side_idx, zone].to(t.int64)
            index_ok = (index >= 0) & (index < K)
            safe = index.clamp(0, max(K - 1, 0))
            valid_card = in_zone & index_ok & col['card_valid'].gather(1, safe)
            kinds = card_seven.gather(1, safe)
            if public_mask is not None:
                valid_card = valid_card & public_mask.gather(1, safe)
            unknown = active.unsqueeze(1) & in_zone & (
                ~index_ok | (kinds < 0) | ~col['card_valid'].gather(1, safe))
            return kinds, valid_card, unknown

        own_hand_k, own_hand_v, unk_h = zone_kinds(own, ZONE_HAND)
        opp_hand_k, opp_hand_v, unk_oh = zone_kinds(1 - own, ZONE_HAND, public_mask=copy_flag)
        own_disc_k, own_disc_v, unk_d = zone_kinds(own, ZONE_DISCARD)
        opp_disc_k, opp_disc_v, unk_od = zone_kinds(1 - own, ZONE_DISCARD)
        errors = errors | (unk_h | unk_oh | unk_d | unk_od).any(1).to(t.int32) * UNKNOWN_TEMPLATE
        errors = errors | (active & ((zone_count < 0) | (zone_count > Z)).any(2).any(1)).to(
            t.int32) * COUNT_OVERFLOW
        choice_kind = col['choice_kind'].to(t.int64)
        choice_in = (choice_kind >= 0) & (choice_kind < len(CHOICE_KINDS))
        header_choice = t.where(
            pending & choice_in,
            self.catalog.header_choice[choice_kind.clamp(0, len(CHOICE_KINDS) - 1)].to(t.int64),
            t.full((rows,), -1, dtype=t.int64, device=self.device))
        viewer_choice = pending & (col['choice_side'].to(t.int64) == own)
        choice_n = t.where(viewer_choice, col['choice_card_count'].to(t.int32),
                           t.zeros(rows, dtype=t.int32, device=self.device))
        cpos = t.arange(CH, device=self.device).unsqueeze(0).expand(rows, CH)
        choice_in_range = viewer_choice.unsqueeze(1) & (cpos < choice_n.unsqueeze(1).clamp(min=0, max=CH))
        cindex = col['choice_card_index'].to(t.int64)
        c_ok = (cindex >= 0) & (cindex < K)
        c_safe = cindex.clamp(0, max(K - 1, 0))
        choice_kinds = card_seven.gather(1, c_safe)
        choice_valid = choice_in_range & c_ok & col['card_valid'].gather(1, c_safe) & (choice_kinds >= 0)
        errors = errors | (active.unsqueeze(1) & choice_in_range & ~choice_valid).any(1).to(
            t.int32) * UNKNOWN_TEMPLATE
        resolving = t.full((rows,), -1, dtype=t.int64, device=self.device)
        op_idx = col['op_card_index'].to(t.int64)
        op_ok = col['operation_present'] & (op_idx >= 0) & (op_idx < K)
        resolving = t.where(
            op_ok, card_seven.gather(1, op_idx.clamp(0, max(K - 1, 0)).unsqueeze(1)).squeeze(1), resolving)
        errors = errors | (active & col['operation_present'] & ((resolving < 0) | ~op_ok)).to(
            t.int32) * UNKNOWN_TEMPLATE
        slot = t.arange(2, device=self.device)
        m_count = col['mulligan_count'].to(t.int32)
        m_in = slot.unsqueeze(0) < m_count.unsqueeze(1).clamp(min=0, max=2)
        m_sides = col['mulligan_sides'].to(t.int64)
        viewer_done = (m_in & (m_sides == own.unsqueeze(1))).any(1)
        done0 = (m_in & (m_sides == 0)).any(1)
        done1 = (m_in & (m_sides == 1)).any(1)
        mulligan_actor = t.where(~done0, t.zeros(rows, dtype=t.int64, device=self.device),
                                 t.where(~done1, t.ones(rows, dtype=t.int64, device=self.device),
                                         col['active_side'].to(t.int64)))
        actors = t.where(phase == PHASE_MULLIGAN, mulligan_actor, col['active_side'].to(t.int64))
        actors = t.where(finished | ~active, own, actors)
        own_hand_n = hand_n[batch, own]
        work = active & ~finished & (errors == 0)
        want_mulligan = work & (phase == PHASE_MULLIGAN) & ~viewer_done
        want_concede = work & ((phase == PHASE_MULLIGAN)
                               | ((phase == PHASE_PLAYING) & (own != col['active_side'].to(t.int64))))
        combo_sel = self.combo_index.unsqueeze(0).expand(rows, self.n_combos, MAX_MULLIGAN_SELECT)
        combo_ok = ((combo_sel < 0) | (combo_sel < own_hand_n.view(rows, 1, 1))).all(-1) & want_mulligan.unsqueeze(1)
        raw_count = want_concede.to(t.int32) + combo_ok.to(t.int32).sum(1)
        overflow = work & ((own_hand_n > MAX_HAND_COMBOS) | (own_hand_n < 0) | (raw_count > self.max_actions))
        errors = errors | overflow.to(t.int32) * COUNT_OVERFLOW
        admit = combo_ok & (errors == 0).unsqueeze(1)
        dest = combo_ok.to(t.int32).cumsum(1) - 1 + want_concede.unsqueeze(1).to(t.int32)
        width = self.max_actions
        tokens = t.full((rows, width + 1, TOKEN_WIDTH), -1, dtype=t.int32, device=self.device)
        ordinals = t.full((rows, width + 1, SEARCH_CANDIDATE_CAP), -1, dtype=t.int32, device=self.device)
        concede_ok = want_concede & (errors == 0)
        tokens[:, 0, 0] = t.where(
            concede_ok, t.full((), ACTION_CONCEDE, dtype=t.int32, device=self.device), tokens[:, 0, 0])
        b_idx = batch.unsqueeze(1).expand(rows, self.n_combos)
        dummy = t.full((), width, dtype=t.int64, device=self.device)
        dest_i = t.where(admit, dest.to(t.int64), dummy)
        tokens[b_idx, dest_i, 0] = t.where(
            admit, t.full((), ACTION_MULLIGAN, dtype=t.int32, device=self.device), tokens[b_idx, dest_i, 0])
        select = self.combo_select.unsqueeze(0).expand(rows, self.n_combos, SEARCH_CANDIDATE_CAP)
        ordinals[b_idx, dest_i] = t.where(admit.unsqueeze(-1), select, ordinals[b_idx, dest_i])
        tokens = tokens[:, :width]
        ordinals = ordinals[:, :width]
        counts = t.where(active & (errors == 0), raw_count.clamp(max=width),
                         t.zeros(rows, dtype=t.int32, device=self.device))
        mask = t.arange(width, device=self.device).unsqueeze(0) < counts.unsqueeze(1)
        tokens = t.where(mask.unsqueeze(-1), tokens, t.full_like(tokens, -1))
        ordinals = t.where(mask.unsqueeze(-1), ordinals, t.full_like(ordinals, -1))

        def g(value):
            return self._gather_sides(value, side_index)

        g_hp = g(col['player_hp'])
        named = {
            'turn': col['turn'].to(t.int32),
            'phase': phase,
            'first': (own == col['first_side'].to(t.int64)).to(t.int32),
            'escalation': col['escalation_enabled'].to(t.int32),
            'choice_kind': header_choice.to(t.int32),
            'resolving': resolving.to(t.int32),
            'choice_count': choice_n,
            'player_hp': g_hp,
            'player_shield': g(col['player_shield']),
            'player_ap': g(col['player_ap']),
            'player_max_hp': g(col['player_max_hp']),
            'player_front_seat': g(front).to(t.int32),
            'player_last_front_seat': g(last_front).to(t.int32),
            'player_hand_n': g(hand_n),
            'player_deck_n': g(deck_n),
            'player_discard_n': g(discard_n),
            'char_present': g(char_present.to(t.int32)),
            'char_hp': g(char_hp),
            'char_shield': g(char_shield),
            'char_energy': g(char_energy),
            'char_order': g(char_order),
            'own_hand_card_kinds': own_hand_k.to(t.int32),
            'own_hand_valid': own_hand_v,
            'choice_card_kinds': choice_kinds.to(t.int32),
            'choice_card_valid': choice_valid,
        }
        side_values = {
            'hp': g_hp, 'shield': g(col['player_shield']), 'ap': g(col['player_ap']),
            'front': g(front), 'last_front': g(last_front),
            'turn_count': g(self._optional_int(pint_p, pint_v, PINT['turn_count'])),
            'normal_atk': g(self._optional_bool(pbool_p, pbool_v, PBOOL['normal_attack_available']).to(t.int32)),
            'ultimate_ok': g(self._optional_bool(pbool_p, pbool_v, PBOOL['ultimate_available']).to(t.int32)),
            'extra_genesis': g(self._optional_bool(pbool_p, pbool_v, PBOOL['extra_genesis_pending']).to(t.int32)),
            'nanali_seed': g(self._optional_int(pint_p, pint_v, PINT['nanali_family'])),
            'nanali_seed_played': g(self._optional_int(pint_p, pint_v, PINT['nanali_family_played'])),
            'hand_n': g(hand_n), 'deck_n': g(deck_n), 'discard_n': g(discard_n),
            'fatigue': g(self._optional_int(pint_p, pint_v, PINT['fatigue'])),
            'extra_ap': g(self._optional_int(pint_p, pint_v, PINT['extra_ap'])),
            'surplus': g(self._optional_bool(pbool_p, pbool_v, PBOOL['surplus']).to(t.int32)),
            'harmony_damage': g(self._optional_bool(pbool_p, pbool_v, PBOOL['harmony_damage']).to(t.int32)),
            'genesis_damage_bonus': g(self._optional_int(pint_p, pint_v, PINT['genesis_damage_bonus'])),
        }
        char_values = {
            'ch_present': g(char_present.to(t.int32)), 'ch_hp': g(char_hp),
            'ch_max_hp': g(char_max_hp), 'ch_shield': g(char_shield),
            'ch_harmony': g(char_harmony), 'ch_energy': g(char_energy),
            'ch_down': g(char_down), 'ch_awakened': g(char_awakened.to(t.int32)),
            'ch_ult_turns': g(char_ult), 'ch_growth': g(char_growth),
            'ch_energy_max': g(char_energy_max), 'ch_order': g(char_order),
            'ch_jingu': g(char_jingu), 'ch_jingu_loan': g(char_jingu_loan.to(t.int32)),
        }
        card_blocks = [
            self._scatter_classes(own_hand_k, own_hand_v, NUM_CARDS),
            self._scatter_classes(own_disc_k, own_disc_v, NUM_CARDS),
            self._scatter_classes(opp_disc_k, opp_disc_v, NUM_CARDS),
            self._scatter_classes(opp_hand_k, opp_hand_v, NUM_CARDS),
        ]
        seven_side = t.stack((
            g(col['player_max_hp']),
            g(2 + self._optional_int(pint_p, pint_v, PINT['insight_bonus'])),
            g(self._optional_int(pint_p, pint_v, PINT['insight_played'])),
            g(t.zeros_like(col['player_hp'])),
            g(self._optional_int(pint_p, pint_v, PINT['heiyu_divinations'])),
        ), dim=2)
        mapped_choice = t.where(
            pending & choice_in,
            self.catalog.seven_choice[choice_kind.clamp(0, len(CHOICE_KINDS) - 1)].to(t.int64),
            t.full((rows,), -1, dtype=t.int64, device=self.device))
        seven_choice = t.zeros((rows, len(CHOICES)), dtype=t.float32, device=self.device)
        choice_hot = (mapped_choice >= 0) & (mapped_choice < len(CHOICES))
        seven_choice.scatter_(
            1, mapped_choice.clamp(0, max(len(CHOICES) - 1, 0)).unsqueeze(1),
            choice_hot.unsqueeze(1).to(t.float32))
        packed_parts = [
            self._ratio(col['turn'], 30.0).unsqueeze(1),
            named['first'].to(t.float32).unsqueeze(1),
            named['escalation'].to(t.float32).unsqueeze(1),
            self._ratio(phase, 10.0).unsqueeze(1),
            self._ratio(header_choice, 10.0).unsqueeze(1),
            t.where(resolving >= 0, self._ratio(resolving, float(NUM_OLD_CARDS)),
                    t.full((rows,), -1.0 / float(NUM_OLD_CARDS), dtype=t.float32, device=self.device)).unsqueeze(1),
            self._ratio(choice_n, 10.0).unsqueeze(1),
        ]
        for key in SIDE_FIELDS:
            if key in IMPLEMENTED_SIDE:
                packed_parts.append(self._ratio(side_values[key], 10.0))
        for key in CHAR_FIELDS:
            if key in IMPLEMENTED_CHAR:
                packed_parts.append(self._ratio(char_values[key], 10.0).reshape(rows, 2 * NUM_SEATS))
        for block in card_blocks:
            packed_parts.append(self._ratio(block, 2.0))
        packed_parts.append(self._ratio(seven_side.reshape(rows, 2 * len(NEW_SIDE)), 10.0))
        packed_parts.append(seven_choice)
        packed_parts.append(self._ratio(self._scatter_classes(choice_kinds, choice_valid, NUM_CARDS), 2.0))
        packed = t.cat(packed_parts, dim=1)
        if packed.shape[1] != len(IMPLEMENTED_FEATURE_NAMES):
            raise RuntimeError('Implemented named-column pack drifted from the declared feature list')
        packed = t.where(active.unsqueeze(1), packed, t.zeros_like(packed))
        flags = t.stack((
            active & ((errors & (INVALID_ROW | UNKNOWN_TEMPLATE)) == 0),
            active & (errors == 0) & ~unhandled,
            viewer_done,
            finished,
        ), dim=1)
        inactive = ~active
        errors = t.where(inactive, t.zeros_like(errors), errors)
        counts = t.where(inactive, t.zeros_like(counts), counts)
        mask = t.where(inactive.unsqueeze(1), t.zeros_like(mask), mask)
        tokens = t.where(inactive.view(rows, 1, 1), t.full_like(tokens, -1), tokens)
        ordinals = t.where(inactive.view(rows, 1, 1), t.full_like(ordinals, -1), ordinals)
        actors = t.where(inactive, t.full_like(actors, -1), actors)
        del hidden_rng
        return DomainPublicProjection(
            named=MappingProxyType(named), packed_named=packed,
            packed_names=IMPLEMENTED_FEATURE_NAMES, packed_index=IMPLEMENTED_FEATURE_INDEX,
            actors=actors, tokens=tokens, mask=mask, counts=counts,
            mulligan_ordinals=ordinals, public_flags=flags, errors=errors, active=active)
