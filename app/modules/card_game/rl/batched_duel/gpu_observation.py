"""GPU batch observation encoding for Duel V2.

Targets all 2420 columns of cross_lineup.all_features() / cross_lineup.encode(view, actions)[0].
Consumes only sanitized public view dictionaries (not raw simulation snapshot).
Compact raw scalar / character ID / card ID / zone / mark / timed-effect tokens are extracted on host,
while seat scattering, absence defaults, normalizations, 5-zone card counting, Jingu mark delta
counting, fixed extras, timed-attack left 1/2 summing, Murk debuffs/mirage/blood-mist boolean comparisons,
and substituted hand counts are executed purely via PyTorch tensor operations on GPU.

No CPU fallback: cuda unavailable raises RuntimeError unless device='cpu' is explicitly specified.
Module import does not unconditionally require PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

torch = None

from app.modules.card_game.rl import cross_lineup, league_schema, murk_lineup

# Schema and feature constants
SCHEMA: str = cross_lineup.SCHEMA
OBS_DIM: int = 2420

SEATS: tuple[str, ...] = cross_lineup.SEATS
SEAT_INDEX: dict[str, int] = cross_lineup.SEAT_INDEX
NUM_SEATS: int = len(SEATS)

CARD_IDS: tuple[str, ...] = cross_lineup.CARD_IDS
CARD_INDEX: dict[str, int] = cross_lineup.CARD_INDEX
NUM_CARDS: int = len(CARD_IDS)

OLD_CARDS: tuple[str, ...] = league_schema.OLD_CARDS
NUM_OLD_CARDS: int = len(OLD_CARDS)

SIDE_FIELDS: tuple[str, ...] = league_schema.SIDE_FIELDS
NUM_SIDE_FIELDS: int = len(SIDE_FIELDS)

CHAR_FIELDS: tuple[str, ...] = league_schema.CHAR_FIELDS
NUM_CHAR_FIELDS: int = len(CHAR_FIELDS)

EXTRA_SIDE: tuple[str, ...] = cross_lineup.EXTRA_SIDE
NUM_EXTRA_SIDE: int = len(EXTRA_SIDE)

EXTRA_CHAR: tuple[str, ...] = cross_lineup.EXTRA_CHAR
NUM_EXTRA_CHAR: int = len(EXTRA_CHAR)

DOT_KINDS: tuple[str, ...] = murk_lineup.DOT_KINDS

FEATURE_NAMES: list[str] = cross_lineup.all_features()
if len(FEATURE_NAMES) != OBS_DIM:
    raise RuntimeError(f"Expected {OBS_DIM} features, got {len(FEATURE_NAMES)}")

SCHEMA_SHA: str = hashlib.sha256(
    f"{SCHEMA}:{OBS_DIM}:{','.join(FEATURE_NAMES)}".encode('utf-8')
).hexdigest()

# Fixed capacity limits for compact token tables
MAX_PRESENT: int = 8
MAX_TIMED: int = 8
MAX_ZONE_CARDS: int = 64
MAX_HAND_CARDS: int = 32

# Slice boundaries in 2420 observation vector
HEADER_START: int = 0
HEADER_LEN: int = 7

SIDE_START: int = HEADER_START + HEADER_LEN
SIDE_LEN: int = NUM_SIDE_FIELDS * 2  # 30 * 2 = 60

CHAR_START: int = SIDE_START + SIDE_LEN
CHAR_LEN: int = NUM_CHAR_FIELDS * 2 * NUM_SEATS  # 35 * 2 * 15 = 1050

CARDS_START: int = CHAR_START + CHAR_LEN
CARDS_LEN: int = 5 * NUM_CARDS  # 5 * 115 = 575

MARKS_START: int = CARDS_START + CARDS_LEN
MARKS_LEN: int = 3 * NUM_CARDS  # 3 * 115 = 345

EXTRA_SIDE_START: int = MARKS_START + MARKS_LEN
EXTRA_SIDE_LEN: int = NUM_EXTRA_SIDE * 2  # 7 * 2 = 14

EXTRA_CHAR_START: int = EXTRA_SIDE_START + EXTRA_SIDE_LEN
EXTRA_CHAR_LEN: int = NUM_EXTRA_CHAR * 2 * NUM_SEATS  # 8 * 2 * 15 = 240

MURK_START: int = EXTRA_CHAR_START + EXTRA_CHAR_LEN
MURK_LEN: int = 14

SUBSTITUTED_START: int = MURK_START + MURK_LEN
SUBSTITUTED_LEN: int = NUM_CARDS  # 115

assert SUBSTITUTED_START + SUBSTITUTED_LEN == OBS_DIM


def _require_torch() -> Any:
    global torch
    if torch is None:
        import torch as torch_module
        torch = torch_module
    return torch


def _resolve_device(device: Optional[Union[str, Any]] = None) -> Any:
    t = _require_torch()
    if device is None:
        if not t.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available and no explicit device was provided (no CPU fallback)"
            )
        return t.device("cuda", t.cuda.current_device())

    dev = t.device(device) if isinstance(device, str) else device
    if dev.type == "cuda":
        if not t.cuda.is_available():
            raise RuntimeError(f"CUDA device '{device}' requested but CUDA is not available")
        if dev.index is None:
            dev = t.device('cuda', t.cuda.current_device())
    elif dev.type != "cpu":
        raise ValueError("Only explicit CPU oracle or CUDA is supported")
    return dev


def _rewrite_character_id(cid: Optional[str]) -> Optional[str]:
    if isinstance(cid, str) and cid.startswith('summon_'):
        return 'summon'
    return cid


@dataclass(frozen=True)
class PackedObservationBatch:
    """Padded torch observation batch tensor and associated metadata."""

    observations: Any  # torch.Tensor [N, OBS_DIM], float32
    mask: Any          # torch.Tensor [N], bool
    schema_sha: str    # Hex digest representing observation schema
    device: Any        # torch.device
    request_ids: tuple = ()

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.observations.shape[0]), int(self.observations.shape[1]))

    @property
    def dtype(self) -> Any:
        return self.observations.dtype

    def __getitem__(self, idx: Any) -> Any:
        return self.observations[idx]

    def to(self, device: Any) -> PackedObservationBatch:
        dev = _resolve_device(device)
        return PackedObservationBatch(
            observations=self.observations.to(dev),
            mask=self.mask.to(dev),
            schema_sha=self.schema_sha,
            device=dev,
            request_ids=self.request_ids,
        )

    def cpu(self) -> PackedObservationBatch:
        return self.to("cpu")

    def cuda(self) -> PackedObservationBatch:
        return self.to("cuda")

    def numpy(self) -> np.ndarray:
        return self.observations.detach().cpu().numpy()


@dataclass
class RawObservationTokens:
    """Compact raw token tables extracted from public views on host.

    Can also be directly emitted by future GPU rule kernels.
    No divisions by 10/30, no .5 counts, no one-hot, no aggregations,
    and no turn comparisons are performed during host extraction.
    """

    # [N, 9]: [turn, viewer_side (0/1), first_side (0/1), escalation (0/1), phase (0..3),
    #          pending_kind (-1,1..3), pending_choices_count, resolving_idx, has_pending_resolving]
    global_tokens: Union[np.ndarray, Any]
    # [N, 2, 6]: [front_seat, last_front_seat, burn_by_side, dark_by_side, extra_genesis_actor_seat, delay_by_side]
    side_int_tokens: Union[np.ndarray, Any]
    # [N, 2, 39]: raw side scalars
    side_float_tokens: Union[np.ndarray, Any]
    # [N, 2, MAX_PRESENT]: bool, character active in slot
    char_present: Union[np.ndarray, Any]
    # [N, 2, MAX_PRESENT]: int64, seat index in 0..14 (-1 if unused)
    char_seats: Union[np.ndarray, Any]
    # [N, 2, MAX_PRESENT, 2]: int64, [shape_card_idx, collapse_by_side (0 for 'a', 1 for 'b', -1 for none)]
    char_int_tokens: Union[np.ndarray, Any]
    # [N, 2, MAX_PRESENT, 38]: float32, raw character scalars and flags
    char_float_tokens: Union[np.ndarray, Any]
    # [N, 2, MAX_PRESENT, MAX_TIMED, 2]: float32, [amount, left]
    char_timed_attacks: Union[np.ndarray, Any]
    # [N, 5, MAX_ZONE_CARDS]: int64, card_idx in 0..114 (-1 if unused)
    zone_cards: Union[np.ndarray, Any]
    # [N, MAX_HAND_CARDS]: int64, card_idx in 0..114
    mark_cards: Union[np.ndarray, Any]
    # [N, MAX_HAND_CARDS]: int64, delta in -1, 0, 1
    mark_deltas: Union[np.ndarray, Any]
    # [N, MAX_HAND_CARDS]: bool
    mark_valid: Union[np.ndarray, Any]
    # [N, MAX_HAND_CARDS]: int64, substituted face card_idx
    substituted_cards: Union[np.ndarray, Any]
    # [N, MAX_HAND_CARDS]: bool
    substituted_valid: Union[np.ndarray, Any]
    # [N]: bool
    mask: Union[np.ndarray, Any]
    # [N]: int64
    counts: Union[np.ndarray, Any]
    request_ids: tuple = ()


def extract_raw_observation_tokens(
    views: Sequence[Dict[str, Any]],
    *,
    capacity: Optional[int] = None,
) -> RawObservationTokens:
    """Extract compact token tables from sanitized public views on host.

    Consumes only public view structures (not raw engine snapshots).
    Filters hidden opponent hand cards without reading their identities.
    Summon aliases are adapted per murk._adapt semantics (only front or latest summon).
    """
    n = len(views)
    if n>16384:raise ValueError("Observation batch exceeds bounded capacity")
    if capacity is not None:
        if type(capacity) is not int or not 0 <= capacity <= 16384:
            raise ValueError("Observation capacity must be an integer in 0..16384")
        if n > capacity:
            raise ValueError(f"Batch views count {n} exceeds explicit capacity {capacity}")
        total_len = capacity
    else:
        total_len = n

    global_tokens = np.zeros((total_len, 9), dtype=np.float32)
    side_int_tokens = np.full((total_len, 2, 6), -1, dtype=np.int64)
    side_float_tokens = np.zeros((total_len, 2, 39), dtype=np.float32)

    char_present = np.zeros((total_len, 2, MAX_PRESENT), dtype=bool)
    char_seats = np.full((total_len, 2, MAX_PRESENT), -1, dtype=np.int64)
    char_int_tokens = np.full((total_len, 2, MAX_PRESENT, 2), -1, dtype=np.int64)
    char_float_tokens = np.zeros((total_len, 2, MAX_PRESENT, 38), dtype=np.float32)
    char_timed_attacks = np.zeros((total_len, 2, MAX_PRESENT, MAX_TIMED, 2), dtype=np.float32)

    zone_cards = np.full((total_len, 5, MAX_ZONE_CARDS), -1, dtype=np.int64)
    mark_cards = np.full((total_len, MAX_HAND_CARDS), -1, dtype=np.int64)
    mark_deltas = np.zeros((total_len, MAX_HAND_CARDS), dtype=np.int64)
    mark_valid = np.zeros((total_len, MAX_HAND_CARDS), dtype=bool)
    substituted_cards = np.full((total_len, MAX_HAND_CARDS), -1, dtype=np.int64)
    substituted_valid = np.zeros((total_len, MAX_HAND_CARDS), dtype=bool)

    mask = np.zeros(total_len, dtype=bool)
    counts = np.zeros(total_len, dtype=np.int64)

    for i in range(n):
        view = views[i]
        viewer = view.get('viewer_side')
        if viewer not in ('a', 'b'):
            raise ValueError(f"Invalid viewer_side: {viewer}")
        order = (viewer, 'b' if viewer == 'a' else 'a')

        first_side = view.get('first_side')
        if first_side not in ('a', 'b'):
            raise ValueError(f"Invalid first_side: {first_side}")

        public = view.get('policy_state') or {}
        if not public:
            raise ValueError("Missing policy_state in view")

        mask[i] = True
        counts[i] = 1

        # Global header tokens
        turn = float(view.get('turn', 0))
        viewer_code = 0.0 if viewer == 'a' else 1.0
        first_code = 0.0 if first_side == 'a' else 1.0
        escalation_enabled = float(bool(public.get('escalation_enabled')))

        phase_str = view.get('phase', 'playing')
        phase_map = {'mulligan': 0.0, 'playing': 1.0, 'choice': 2.0, 'finished': 3.0}
        if phase_str not in phase_map:
            raise ValueError(f"Unknown phase: {phase_str}")
        phase_val = phase_map[phase_str]

        pending = view.get('pending_choice') or {}
        resolving = view.get('resolving_card') or {}

        kind_map = {'inspect_top': 1.0, 'discard': 2.0, 'enemy_hand': 3.0}
        pending_kind = kind_map.get(pending.get('kind'), -1.0)
        choices_count = float(len(pending.get('choices', [])))

        resolving_idx = -1.0
        has_pending_resolving = 0.0
        if pending and resolving and resolving.get('card_id'):
            res_cid = resolving['card_id']
            if res_cid not in CARD_INDEX:
                raise ValueError(f"Unknown resolving card_id: {res_cid}")
            resolving_idx = float(CARD_INDEX[res_cid])
            has_pending_resolving = 1.0

        global_tokens[i] = [
            turn, viewer_code, first_code, escalation_enabled,
            phase_val, pending_kind, choices_count, resolving_idx, has_pending_resolving
        ]

        # Side & Character state
        for side_idx, side_key in enumerate(order):
            team = view.get('sides', {}).get(side_key, {})
            extra = public.get('sides', {}).get(side_key, {})
            zones = team.get('front_debuff') or {}

            # Summon adaptation per murk._adapt semantics
            chars = list(team.get('characters') or [])
            summons = [
                h for h in chars
                if h.get('summoned') or str(h.get('id', '')).startswith('summon_')
            ]
            non_summons = [h for h in chars if h not in summons]
            active_chars: List[Tuple[Dict[str, Any], str]] = []
            if summons:
                chosen = next((h for h in summons if team.get('front') == h.get('id')), summons[-1])
                old_id = chosen['id']
                chosen_adapted = dict(chosen)
                chosen_adapted['id'] = 'summon'
                active_chars = [(h, h['id']) for h in non_summons] + [(chosen_adapted, old_id)]
            else:
                active_chars = [(h, h['id']) for h in non_summons]

            # Side integer tokens
            front_id = _rewrite_character_id(team.get('front'))
            last_front_id = _rewrite_character_id(team.get('last_front'))
            front_seat = SEAT_INDEX.get(front_id, -1) if front_id else -1
            last_front_seat = SEAT_INDEX.get(last_front_id, -1) if last_front_id else -1

            burn_zone = zones.get('burn') or {}
            burn_by_side = -1 if not burn_zone else (0 if burn_zone.get('by') == 'a' else 1)

            star_zone = zones.get('star') or {}
            dark_by_side = -1 if not star_zone else (0 if star_zone.get('by') == 'a' else 1)

            extra_actor_id = _rewrite_character_id(extra.get('extra_genesis_actor'))
            extra_genesis_actor_seat = SEAT_INDEX.get(extra_actor_id, -1) if extra_actor_id else -1

            delay_zone = zones.get('delay') or {}
            delay_by_side = -1 if not delay_zone else (0 if delay_zone.get('by') == 'a' else 1)

            side_int_tokens[i, side_idx] = [
                front_seat, last_front_seat, burn_by_side, dark_by_side,
                extra_genesis_actor_seat, delay_by_side
            ]

            # Side float tokens
            dots = zones.get('dots') or {}
            dot_attack_turn = -999.0
            mirage_val = 0.0
            for h, _ in active_chars:
                if h.get('id') == 'canhong':
                    eid = h.get('entity_id')
                    used_map = team.get('used') or {}
                    val = used_map.get(f"canhong_dot_attack_turn:{eid}")
                    if val is not None:
                        dot_attack_turn = float(val)
                    effects = h.get('effects') or []
                    if any(eff.get('definition_id') == 'canhong.mirage' for eff in effects):
                        mirage_val = 1.0

            side_float_tokens[i, side_idx] = [
                float(team.get('hp', 0)),
                float(team.get('shield', 0)),
                float(team.get('ap', 0)),
                float(team.get('turn_count', 0)),
                float(team.get('fatigue', 0)),
                float(bool(team.get('normal_attack_available'))),
                float(bool(team.get('ultimate_available'))),
                float(bool((team.get('used') or {}).get('instant'))),
                float(team.get('ultimate_remaining', 0)),
                float(team.get('hand_count', 0)),
                float(team.get('deck_count', 0)),
                float(len(team.get('discard', []))),
                float(bool(zones.get('weave'))),
                float(burn_zone.get('left', 0)),
                float(bool(burn_zone.get('infinite'))),
                float(delay_zone.get('end', 0)),
                float(extra.get('light_resistance', 0)),
                float(star_zone.get('left', 0)),
                float(extra.get('extra_ap', 0) or 0),
                float(bool(extra.get('extra_genesis_pending'))),
                float(bool(extra.get('surplus'))),
                float(int(extra.get('genesis_damage_bonus') or 0)),
                float(bool(extra.get('harmony_damage'))),
                float(extra.get('nanali_family', 0) or 0),
                float(extra.get('nanali_family_played', 0) or 0),
                float(delay_zone.get('left', 0)),
                float(delay_zone.get('slow', 0)),
                float(delay_zone.get('slow_turn', -999)),
                float(extra.get('delay_turns', 0) or 0),
                float(bool(extra.get('skip_normal_attack'))),
                float(extra['genesis_turn']) if extra.get('genesis_turn') is not None else -999.0,
                float(extra.get('genesis_player_hits', 0) or 0),
                float(int(burn_zone.get('stacks') or 0)),
                float(int((dots.get('nightmare') or {}).get('stacks') or 0)),
                float(int((dots.get('etch') or {}).get('stacks') or 0)),
                float(int((dots.get('venom') or {}).get('stacks') or 0)),
                float(int((dots.get('guard') or {}).get('stacks') or 0)),
                mirage_val,
                dot_attack_turn,
            ]

            # Characters in slots
            if len(active_chars) > MAX_PRESENT:
                raise ValueError("Active characters count exceeds token capacity")

            meta_chars = extra.get('characters', {})
            for p, (h, lookup_id) in enumerate(active_chars):
                cid = h.get('id')
                if cid not in SEAT_INDEX:
                    raise ValueError(f"Model only supports public characters: {cid}")
                seat_idx = SEAT_INDEX[cid]
                char_present[i, side_idx, p] = True
                char_seats[i, side_idx, p] = seat_idx

                shape_id = h.get('shape_id')
                if shape_id:
                    if shape_id not in CARD_INDEX:
                        raise ValueError(f"Unknown shape_id: {shape_id}")
                    shape_idx = CARD_INDEX[shape_id]
                else:
                    shape_idx = -1

                meta_hero = meta_chars.get(lookup_id, {})
                flags = meta_hero.get('flags', {})
                collapse = flags.get('collapse') or {}
                collapse_until = float(collapse.get('until', 0))
                if not collapse:
                    collapse_by_side = -1
                else:
                    col_side = collapse.get('side')
                    if col_side not in ('a', 'b'):
                        raise ValueError(f"Unknown collapse side: {col_side}")
                    collapse_by_side = 0 if col_side == 'a' else 1

                char_int_tokens[i, side_idx, p] = [shape_idx, collapse_by_side]

                char_float_tokens[i, side_idx, p] = [
                    float(h.get('hp', 0)),
                    float(h.get('max_hp', 0)),
                    float(meta_hero.get('base_attack', 0)),
                    float(h.get('growth', 0)),
                    float(flags.get('atk_buff', 0) or 0),
                    float(h.get('shield', 0)),
                    float(h.get('harmony', 0)),
                    float(h.get('energy', 0)),
                    float(h.get('down_turns', 0)),
                    float(h.get('awakened', 0)),
                    float(h.get('ultimate_turns', 0)),
                    float(flags.get('next_bonus', 0) or 0),
                    float(flags.get('next_shield', 0) or 0),
                    float(flags.get('next_followup', 0) or 0),
                    float(flags.get('pact', 0) or 0),
                    float(flags.get('energy_next', 0) or 0),
                    float(flags.get('allied_hurt', 0) or 0),
                    float(flags.get('pending_atk', 0) or 0),
                    float(flags.get('share_overflow', 0) or 0),
                    float(flags.get('hp_floor', 0) or 0),
                    float(flags.get('collapse_count', 0) or 0),
                    float(meta_hero.get('base_max_hp', 0)),
                    float(h.get('energy_max', 0)),
                    float(h.get('harmony_max', 0)),
                    float(meta_hero.get('order', -1)),
                    float(meta_hero.get('harmonized', 0)),
                    float(h.get('jingu', 0)),
                    float(bool(meta_hero.get('jingu_loan'))),
                    float(meta_hero.get('ultimate_used_turn', -999)),
                    float(bool(meta_hero.get('attacked_this_turn'))),
                    float(meta_hero.get('light_resistance', 0)),
                    collapse_until,
                    float(h.get('beast_fangs', 0) or 0),
                    float(h.get('intimidation', 0) or 0),
                    float(h.get('extra_attacks', 0) or 0),
                    float(h.get('attack', 0) or 0),
                    float(h.get('damage_immune', 0) or 0),
                    float(h.get('damage_limit', 0) or 0),
                ]

                timed_entries = flags.get('timed_attack') or []
                if len(timed_entries) > MAX_TIMED:
                    raise ValueError("Timed attack entries exceed token capacity")
                for t_idx, entry in enumerate(timed_entries):
                    char_timed_attacks[i, side_idx, p, t_idx] = [
                        float(entry.get('amount', 0)),
                        float(entry.get('left', 0)),
                    ]

        # Card zones:
        # Zone 0: viewer's hand
        # Zone 1: viewer's discard
        # Zone 2: opponent's discard
        # Zone 3: opponent's hand (revealed/not hidden)
        # Zone 4: viewer's hand (revealed/not hidden)
        own_team = view['sides'][viewer]
        opp_team = view['sides'][order[1]]

        z0_cards = own_team.get('hand', [])
        z1_cards = own_team.get('discard', [])
        z2_cards = opp_team.get('discard', [])
        z3_cards = [c for c in opp_team.get('hand', []) if not c.get('hidden')]
        z4_cards = [c for c in own_team.get('hand', []) if not c.get('hidden') and c.get('revealed')]

        zones_list = [z0_cards, z1_cards, z2_cards, z3_cards, z4_cards]
        for z_idx, cards_group in enumerate(zones_list):
            if len(cards_group) > MAX_ZONE_CARDS:
                raise ValueError("Zone cards count exceeds token capacity")
            for c_idx, c in enumerate(cards_group):
                cid = c.get('card_id')
                if cid not in CARD_INDEX:
                    raise ValueError(f"Unknown card_id: {cid}")
                zone_cards[i, z_idx, c_idx] = CARD_INDEX[cid]

        # Jingu marks & substituted original cards in viewer's hand
        if len(z0_cards) > MAX_HAND_CARDS:
            raise ValueError("Hand cards count exceeds token capacity")

        for c_idx, c in enumerate(z0_cards):
            cid = c.get('card_id')
            if cid not in CARD_INDEX:
                raise ValueError(f"Unknown card_id: {cid}")
            c_kind_idx = CARD_INDEX[cid]

            if 'jingu_mark' in c:
                delta = int(c['jingu_mark'].get('delta', 0))
                if delta not in (-1, 0, 1):
                    raise ValueError(f"Invalid jingu mark delta: {delta}")
                mark_cards[i, c_idx] = c_kind_idx
                mark_deltas[i, c_idx] = delta
                mark_valid[i, c_idx] = True

            face = c.get('hand_face') or {}
            face_cid = face.get('card_id')
            if face_cid and face_cid != cid:
                if face_cid not in CARD_INDEX:
                    raise ValueError(f"Unknown hand_face card_id: {face_cid}")
                substituted_cards[i, c_idx] = CARD_INDEX[face_cid]
                substituted_valid[i, c_idx] = True

    return RawObservationTokens(
        global_tokens=global_tokens,
        side_int_tokens=side_int_tokens,
        side_float_tokens=side_float_tokens,
        char_present=char_present,
        char_seats=char_seats,
        char_int_tokens=char_int_tokens,
        char_float_tokens=char_float_tokens,
        char_timed_attacks=char_timed_attacks,
        zone_cards=zone_cards,
        mark_cards=mark_cards,
        mark_deltas=mark_deltas,
        mark_valid=mark_valid,
        substituted_cards=substituted_cards,
        substituted_valid=substituted_valid,
        mask=mask,
        counts=counts,
    )


def encode_observation_tokens(
    tokens: RawObservationTokens,
    *,
    device: Optional[Union[str, Any]] = None,
) -> PackedObservationBatch:
    """Encode RawObservationTokens into 2420 observation tensor purely on GPU.

    Executes seat scattering, default assignments for absent seats, normalizations,
    card counting, mark aggregation, extra attributes, and substituted hand counts
    strictly using PyTorch tensor operations.
    """
    t = _require_torch()
    dev = _resolve_device(device)

    if not isinstance(tokens, RawObservationTokens):
        raise TypeError("RawObservationTokens required")

    # Validate types before conversion: float IDs must never be truncated to integers.
    specs={
        'global_tokens':('float',(9,)), 'side_int_tokens':('int',(2,6)),
        'side_float_tokens':('float',(2,39)), 'char_present':('bool',(2,MAX_PRESENT)),
        'char_seats':('int',(2,MAX_PRESENT)), 'char_int_tokens':('int',(2,MAX_PRESENT,2)),
        'char_float_tokens':('float',(2,MAX_PRESENT,38)),
        'char_timed_attacks':('float',(2,MAX_PRESENT,MAX_TIMED,2)),
        'zone_cards':('int',(5,MAX_ZONE_CARDS)), 'mark_cards':('int',(MAX_HAND_CARDS,)),
        'mark_deltas':('int',(MAX_HAND_CARDS,)), 'mark_valid':('bool',(MAX_HAND_CARDS,)),
        'substituted_cards':('int',(MAX_HAND_CARDS,)), 'substituted_valid':('bool',(MAX_HAND_CARDS,)),
        'mask':('bool',()),'counts':('int',())}
    arrays={};size=None
    allowed={'int':(t.int32,t.int64),'float':(t.float32,t.float64),'bool':(t.bool,)}
    target={'int':t.int64,'float':t.float32,'bool':t.bool}
    for name,(kind,tail) in specs.items():
        raw=getattr(tokens,name)
        if not isinstance(raw,np.ndarray) and not isinstance(raw,t.Tensor):raise TypeError('Numeric token array required: '+name)
        value=t.as_tensor(raw,device=dev)
        if value.dtype not in allowed[kind]:raise ValueError('Invalid token dtype: '+name)
        if value.ndim!=1+len(tail) or tuple(value.shape[1:])!=tail:raise ValueError('Invalid token shape: '+name)
        if size is None:size=value.shape[0]
        if value.shape[0]!=size or size>16384:raise ValueError('Unaligned/big observation batch')
        arrays[name]=value.to(dtype=target[kind])
    global_tok=arrays['global_tokens'];side_int=arrays['side_int_tokens'];side_flt=arrays['side_float_tokens']
    ch_pres=arrays['char_present'];ch_seats=arrays['char_seats'];ch_int=arrays['char_int_tokens'];ch_flt=arrays['char_float_tokens']
    ch_timed=arrays['char_timed_attacks'];z_cards=arrays['zone_cards'];m_cards=arrays['mark_cards'];m_deltas=arrays['mark_deltas']
    m_valid=arrays['mark_valid'];sub_cards=arrays['substituted_cards'];sub_valid=arrays['substituted_valid'];mask=arrays['mask'];counts=arrays['counts']
    def bounded(value,lo,hi,label):
        if bool(((value<lo)|(value>hi)).any()):raise ValueError('Invalid token domain: '+label)
    if not bool(t.equal(counts,mask.long())):raise ValueError('Observation mask/count mismatch')
    bounded(ch_seats,-1,NUM_SEATS-1,'character seats');bounded(ch_int[:,:,:,0],-1,NUM_CARDS-1,'shapes')
    bounded(ch_int[:,:,:,1],-1,1,'collapse side');bounded(z_cards,-1,NUM_CARDS-1,'zone cards')
    bounded(m_cards,-1,NUM_CARDS-1,'marked cards');bounded(sub_cards,-1,NUM_CARDS-1,'physical cards');bounded(m_deltas,-1,1,'mark deltas')
    for field in (0,1,4):bounded(side_int[:,:,field],-1,NUM_SEATS-1,'side character')
    for field in (2,3,5):bounded(side_int[:,:,field],-1,1,'side identity')
    if bool((ch_pres&(ch_seats<0)).any()):raise ValueError('Active character without seat')
    seats=ch_seats.unsqueeze(-1)==ch_seats.unsqueeze(-2)
    active=ch_pres.unsqueeze(-1)&ch_pres.unsqueeze(-2)
    identity=t.eye(MAX_PRESENT,dtype=t.bool,device=dev).view(1,1,MAX_PRESENT,MAX_PRESENT)
    if bool((seats&active&~identity).any()):raise ValueError('Duplicate character seat')
    if bool((m_valid&(m_cards<0)).any()) or bool((sub_valid&(sub_cards<0)).any()):raise ValueError('Active card token missing identity')
    for field,lo,hi in ((1,0,1),(2,0,1),(3,0,1),(4,0,3),(5,-1,3),(7,-1,NUM_CARDS-1),(8,0,1)):
        value=global_tok[:,field];bounded(value,lo,hi,'header')
        if bool((value!=value.floor()).any()):raise ValueError('Nonintegral header identity')
    n = int(global_tok.shape[0])
    ids = tuple(tokens.request_ids)
    if not ids:
        epoch = uuid4().hex
        ids = tuple(epoch + ':' + str(i) for i in range(n))
    if len(ids) != n or len(set(ids)) != n or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Versioned unique observation request IDs required")

    if n == 0:
        empty_obs = t.zeros((0, OBS_DIM), dtype=t.float32, device=dev)
        return PackedObservationBatch(
            observations=empty_obs,
            mask=mask,
            schema_sha=SCHEMA_SHA,
            device=dev,
            request_ids=ids,
        )

    # Finite check on inputs
    if not bool(
        t.isfinite(global_tok).all() and t.isfinite(side_flt).all() and
        t.isfinite(ch_flt).all() and t.isfinite(ch_timed).all()
    ):
        raise ValueError("Non-finite input tokens detected")

    # 1. Header features [N, 7]
    turn = global_tok[:, 0]
    viewer_code = global_tok[:, 1].long()
    first_code = global_tok[:, 2].long()
    escalation = global_tok[:, 3]
    phase_val = global_tok[:, 4]
    pending_kind = global_tok[:, 5]
    choices_count = global_tok[:, 6]
    resolving_idx = global_tok[:, 7]
    has_pending_resolving = global_tok[:, 8]

    col0 = turn / 30.0
    col1 = (viewer_code == first_code).float()
    col2 = escalation
    col3 = phase_val / 10.0
    col4 = pending_kind / 10.0
    col5 = t.where(
        has_pending_resolving > 0.5,
        resolving_idx / float(NUM_OLD_CARDS),
        t.tensor(-1.0 / float(NUM_OLD_CARDS), device=dev)
    )
    col6 = choices_count / 10.0
    header_block = t.stack([col0, col1, col2, col3, col4, col5, col6], dim=1)

    # 2. Side fields [N, 60]
    # order: for each key in SIDE_FIELDS (30 fields): s=0 (viewer), s=1 (opponent)
    # Relative side comparison:
    # burn_by: -1 if burn_by_side < 0 else int(burn_by_side != side_id)
    # dark_by: -1 if dark_by_side < 0 else int(dark_by_side != side_id)
    # side_id for viewer (s=0) is viewer_code; for opponent (s=1) is (1 - viewer_code)
    side_ids = t.stack([viewer_code, 1 - viewer_code], dim=1)  # [N, 2]

    front_seat = side_int[:, :, 0].float()
    last_front_seat = side_int[:, :, 1].float()
    burn_by_side = side_int[:, :, 2]
    dark_by_side = side_int[:, :, 3]
    extra_genesis_actor_seat = side_int[:, :, 4].float()

    burn_by = t.where(
        burn_by_side < 0,
        t.tensor(-1.0, device=dev),
        (burn_by_side != side_ids).float()
    )
    dark_by = t.where(
        dark_by_side < 0,
        t.tensor(-1.0, device=dev),
        (dark_by_side != side_ids).float()
    )

    # 30 field values for s in (0, 1), each divided by 10.0
    side_values_list = [
        side_flt[:, :, 0] / 10.0,   # hp
        side_flt[:, :, 1] / 10.0,   # shield
        side_flt[:, :, 2] / 10.0,   # ap
        front_seat / 10.0,          # front
        last_front_seat / 10.0,     # last_front
        side_flt[:, :, 3] / 10.0,   # turn_count
        side_flt[:, :, 5] / 10.0,   # normal_atk
        side_flt[:, :, 6] / 10.0,   # ultimate_ok
        side_flt[:, :, 7] / 10.0,   # used_instant
        side_flt[:, :, 19] / 10.0,  # extra_genesis
        extra_genesis_actor_seat / 10.0,  # extra_genesis_actor
        side_flt[:, :, 23] / 10.0,  # nanali_seed
        side_flt[:, :, 24] / 10.0,  # nanali_seed_played
        side_flt[:, :, 9] / 10.0,   # hand_n
        side_flt[:, :, 10] / 10.0,  # deck_n
        side_flt[:, :, 11] / 10.0,  # discard_n
        side_flt[:, :, 12] / 10.0,  # weave
        side_flt[:, :, 4] / 10.0,   # fatigue
        side_flt[:, :, 18] / 10.0,  # extra_ap
        side_flt[:, :, 20] / 10.0,  # surplus
        side_flt[:, :, 8] / 10.0,   # ultimate_remaining
        side_flt[:, :, 13] / 10.0,  # burn_left
        side_flt[:, :, 14] / 10.0,  # burn_infinite
        side_flt[:, :, 15] / 10.0,  # delay_end
        side_flt[:, :, 22] / 10.0,  # harmony_damage
        burn_by / 10.0,             # burn_by
        side_flt[:, :, 21] / 10.0,  # genesis_damage_bonus
        side_flt[:, :, 16] / 10.0,  # light_resistance
        side_flt[:, :, 17] / 10.0,  # dark_left
        dark_by / 10.0,             # dark_by
    ]
    # side_values_list has 30 tensors of shape [N, 2]
    # Stack along field dimension: [N, 30, 2] -> reshape [N, 60]
    side_block = t.stack(side_values_list, dim=1).reshape(n, NUM_SIDE_FIELDS * 2)

    # 3. Character fields [N, 1050]
    # 35 fields in CHAR_FIELDS:
    # 0: ch_present, 1: ch_hp, 2: ch_max_hp, 3: ch_base_atk, 4: ch_growth, 5: ch_atk_buff,
    # 6: ch_shield, 7: ch_harmony, 8: ch_energy, 9: ch_down, 10: ch_awakened, 11: ch_ult_turns,
    # 12: ch_shape, 13: ch_next_bonus, 14: ch_next_shield, 15: ch_next_followup, 16: ch_pact,
    # 17: ch_harmonized, 18: ch_collapse_count, 19: ch_collapse_until, 20: ch_allied_hurt,
    # 21: ch_energy_next, 22: ch_base_max, 23: ch_energy_max, 24: ch_harmony_max,
    # 25: ch_pending_atk, 26: ch_share_overflow, 27: ch_hp_floor, 28: ch_order,
    # 29: ch_collapse_by, 30: ch_jingu, 31: ch_jingu_loan, 32: ch_ultimate_used,
    # 33: ch_attacked, 34: ch_light_resistance.
    # Default for absent seats: shape=-1, order=-1, collapse_by=-1, others 0.
    char_grid = t.zeros((n, 2, NUM_SEATS, NUM_CHAR_FIELDS), dtype=t.float32, device=dev)
    char_grid[:, :, :, 12] = -1.0  # ch_shape
    char_grid[:, :, :, 28] = -1.0  # ch_order
    char_grid[:, :, :, 29] = -1.0  # ch_collapse_by

    turn_expanded = turn.view(n, 1, 1).expand(n, 2, MAX_PRESENT)
    side_ids_expanded = side_ids.view(n, 2, 1).expand(n, 2, MAX_PRESENT)

    shape_col = ch_int[:, :, :, 0].float()
    col_by_side = ch_int[:, :, :, 1]
    ch_collapse_by = t.where(
        col_by_side < 0,
        t.tensor(-1.0, device=dev),
        (col_by_side != side_ids_expanded).float()
    )
    ult_used = (ch_flt[:, :, :, 28] == turn_expanded).float()

    char_extracted = [
        t.ones_like(ch_flt[:, :, :, 0]),  # 0: ch_present
        ch_flt[:, :, :, 0],               # 1: ch_hp
        ch_flt[:, :, :, 1],               # 2: ch_max_hp
        ch_flt[:, :, :, 2],               # 3: ch_base_atk
        ch_flt[:, :, :, 3],               # 4: ch_growth
        ch_flt[:, :, :, 4],               # 5: ch_atk_buff
        ch_flt[:, :, :, 5],               # 6: ch_shield
        ch_flt[:, :, :, 6],               # 7: ch_harmony
        ch_flt[:, :, :, 7],               # 8: ch_energy
        ch_flt[:, :, :, 8],               # 9: ch_down
        ch_flt[:, :, :, 9],               # 10: ch_awakened
        ch_flt[:, :, :, 10],              # 11: ch_ult_turns
        shape_col,                        # 12: ch_shape
        ch_flt[:, :, :, 11],              # 13: ch_next_bonus
        ch_flt[:, :, :, 12],              # 14: ch_next_shield
        ch_flt[:, :, :, 13],              # 15: ch_next_followup
        ch_flt[:, :, :, 14],              # 16: ch_pact
        ch_flt[:, :, :, 25],              # 17: ch_harmonized
        ch_flt[:, :, :, 20],              # 18: ch_collapse_count
        ch_flt[:, :, :, 31],              # 19: ch_collapse_until
        ch_flt[:, :, :, 16],              # 20: ch_allied_hurt
        ch_flt[:, :, :, 15],              # 21: ch_energy_next
        ch_flt[:, :, :, 21],              # 22: ch_base_max
        ch_flt[:, :, :, 22],              # 23: ch_energy_max
        ch_flt[:, :, :, 23],              # 24: ch_harmony_max
        ch_flt[:, :, :, 17],              # 25: ch_pending_atk
        ch_flt[:, :, :, 18],              # 26: ch_share_overflow
        ch_flt[:, :, :, 19],              # 27: ch_hp_floor
        ch_flt[:, :, :, 24],              # 28: ch_order
        ch_collapse_by,                   # 29: ch_collapse_by
        ch_flt[:, :, :, 26],              # 30: ch_jingu
        ch_flt[:, :, :, 27],              # 31: ch_jingu_loan
        ult_used,                         # 32: ch_ultimate_used
        ch_flt[:, :, :, 29],              # 33: ch_attacked
        ch_flt[:, :, :, 30],              # 34: ch_light_resistance
    ]
    # [N, 2, MAX_PRESENT, 35]
    char_rows = t.stack(char_extracted, dim=-1)

    # Scatter active character rows into fixed seat positions
    valid_slots = ch_pres & (ch_seats >= 0) & (ch_seats < NUM_SEATS)
    if valid_slots.any():
        b_idx, s_idx, p_idx = t.where(valid_slots)
        target_seat = ch_seats[b_idx, s_idx, p_idx]
        char_grid[b_idx, s_idx, target_seat, :] = char_rows[b_idx, s_idx, p_idx, :]

    # Normalize by 10.0 and permute to: for key in char_fields, for s in (0, 1), for c in SEATS
    char_grid = char_grid / 10.0
    char_block = char_grid.permute(0, 3, 1, 2).reshape(n, NUM_CHAR_FIELDS * 2 * NUM_SEATS)

    # 4. Five zone card counts [N, 575]
    zone_counts = t.zeros((n, 5, NUM_CARDS), dtype=t.float32, device=dev)
    valid_z = (z_cards >= 0) & (z_cards < NUM_CARDS)
    clamped_z = z_cards.clamp(min=0)
    zone_counts.scatter_add_(dim=2, index=clamped_z, src=valid_z.float() * 0.5)
    zone_block = zone_counts.reshape(n, 5 * NUM_CARDS)

    # 5. Jingu marks in viewer hand [N, 345]
    mark_counts = t.zeros((n, 3, NUM_CARDS), dtype=t.float32, device=dev)
    delta_idx = m_deltas + 1
    valid_m = m_valid & (m_cards >= 0) & (m_cards < NUM_CARDS) & (delta_idx >= 0) & (delta_idx < 3)
    clamped_m = m_cards.clamp(min=0)
    for d in range(3):
        d_mask = valid_m & (delta_idx == d)
        mark_counts[:, d, :].scatter_add_(dim=1, index=clamped_m, src=d_mask.float() * 0.5)
    mark_block = mark_counts.reshape(n, 3 * NUM_CARDS)

    # 6. Fixed extra side fields [N, 14]
    # EXTRA_SIDE = ('delay_left', 'delay_by', 'delay_slow', 'delay_turns',
    #               'skip_normal_attack', 'genesis_this_turn', 'genesis_player_hits')
    # delay_by: -1 if delay_by_side < 0 else int(delay_by_side != side_id)
    delay_by_side = side_int[:, :, 5]
    delay_by = t.where(
        delay_by_side < 0,
        t.tensor(-1.0, device=dev),
        (delay_by_side != side_ids).float()
    )
    delay_slow = t.where(
        side_flt[:, :, 27] == turn.view(n, 1),
        side_flt[:, :, 26],
        t.tensor(0.0, device=dev)
    )
    genesis_this_turn = (side_flt[:, :, 30] == turn.view(n, 1)).float()

    extra_side_list = [
        side_flt[:, :, 25] / 10.0,   # delay_left
        delay_by / 10.0,             # delay_by
        delay_slow / 10.0,           # delay_slow
        side_flt[:, :, 28] / 10.0,   # delay_turns
        side_flt[:, :, 29] / 10.0,   # skip_normal_attack
        genesis_this_turn / 10.0,    # genesis_this_turn
        side_flt[:, :, 31] / 10.0,   # genesis_player_hits
    ]
    # [N, 7, 2] -> reshape [N, 14]
    extra_side_block = t.stack(extra_side_list, dim=1).reshape(n, NUM_EXTRA_SIDE * 2)

    # 7. Fixed extra character fields [N, 240]
    # EXTRA_CHAR = ('beast_fangs', 'intimidation', 'extra_attacks', 'attack',
    #               'timed_attack_1', 'timed_attack_2', 'damage_immune', 'damage_limit')
    # Default absent is 0.0
    extra_char_grid = t.zeros((n, 2, NUM_SEATS, NUM_EXTRA_CHAR), dtype=t.float32, device=dev)

    # timed_attack left 1 and left 2 summing
    timed_left = ch_timed[:, :, :, :, 1]
    timed_amount = ch_timed[:, :, :, :, 0]
    timed_1 = t.where(timed_left == 1.0, timed_amount, t.tensor(0.0, device=dev)).sum(dim=-1)
    timed_2 = t.where(timed_left == 2.0, timed_amount, t.tensor(0.0, device=dev)).sum(dim=-1)

    extra_char_extracted = [
        ch_flt[:, :, :, 32],  # beast_fangs
        ch_flt[:, :, :, 33],  # intimidation
        ch_flt[:, :, :, 34],  # extra_attacks
        ch_flt[:, :, :, 35],  # attack
        timed_1,              # timed_attack_1
        timed_2,              # timed_attack_2
        ch_flt[:, :, :, 36],  # damage_immune
        ch_flt[:, :, :, 37],  # damage_limit
    ]
    # [N, 2, MAX_PRESENT, 8]
    extra_char_rows = t.stack(extra_char_extracted, dim=-1)

    if valid_slots.any():
        b_idx, s_idx, p_idx = t.where(valid_slots)
        target_seat = ch_seats[b_idx, s_idx, p_idx]
        extra_char_grid[b_idx, s_idx, target_seat, :] = extra_char_rows[b_idx, s_idx, p_idx, :]

    extra_char_grid = extra_char_grid / 10.0
    extra_char_block = extra_char_grid.permute(0, 3, 1, 2).reshape(n, NUM_EXTRA_CHAR * 2 * NUM_SEATS)

    # 8. Murk zone extras, mirage, and dot_attack_used [N, 14]
    # order: viewer 5 debuffs / 10, opp 5 debuffs / 10, viewer mirage, opp mirage, viewer dot, opp dot
    murk_debuffs_v = side_flt[:, 0, 32:37] / 10.0  # burn_stacks, nightmare, etch, venom, guard
    murk_debuffs_o = side_flt[:, 1, 32:37] / 10.0
    mirage_v = side_flt[:, 0, 37:38]
    mirage_o = side_flt[:, 1, 37:38]
    dot_attack_v = (side_flt[:, 0, 38:39] == turn.view(n, 1)).float()
    dot_attack_o = (side_flt[:, 1, 38:39] == turn.view(n, 1)).float()

    murk_block = t.cat([
        murk_debuffs_v, murk_debuffs_o,
        mirage_v, mirage_o,
        dot_attack_v, dot_attack_o,
    ], dim=1)

    # 9. Substituted hand original card identity [N, 115]
    sub_counts = t.zeros((n, NUM_CARDS), dtype=t.float32, device=dev)
    valid_sub = sub_valid & (sub_cards >= 0) & (sub_cards < NUM_CARDS)
    clamped_sub = sub_cards.clamp(min=0)
    sub_counts.scatter_add_(dim=1, index=clamped_sub, src=valid_sub.float() * 0.5)
    substituted_block = sub_counts

    # 10. Concatenate all 2420 columns
    observations = t.cat([
        header_block,
        side_block,
        char_block,
        zone_block,
        mark_block,
        extra_side_block,
        extra_char_block,
        murk_block,
        substituted_block,
    ], dim=1)

    if observations.shape != (n, OBS_DIM):
        raise ValueError(f"Observation dimension mismatch: {observations.shape} != {(n, OBS_DIM)}")

    # 11. Mask out padded games
    observations = observations * mask.view(n, 1).float()

    # 12. Finite check
    if not t.isfinite(observations).all():
        raise ValueError("Non-finite model observation detected")

    return PackedObservationBatch(
        observations=observations,
        mask=mask,
        schema_sha=SCHEMA_SHA,
        device=dev,
        request_ids=ids,
    )


def encode_observation_batch(
    views: Sequence[Dict[str, Any]],
    *,
    capacity: Optional[int] = None,
    device: Optional[Union[str, Any]] = None,
) -> PackedObservationBatch:
    """Batch GPU observation encoder.

    Extracts compact raw tokens on host and encodes observation vectors
    on GPU via PyTorch tensor operations.
    """
    tokens = extract_raw_observation_tokens(views, capacity=capacity)
    return encode_observation_tokens(tokens, device=device)


def encode_single_observation(
    view: Dict[str, Any],
    *,
    capacity: Optional[int] = None,
    device: Optional[Union[str, Any]] = None,
) -> PackedObservationBatch:
    """Convenience helper for encoding a single view."""
    return encode_observation_batch([view], capacity=capacity, device=device)
