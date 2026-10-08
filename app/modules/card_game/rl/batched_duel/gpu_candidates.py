"""GPU batch candidate action encoding for Duel V2.

Targets all columns of cross_lineup.encode(view, actions)[1]; parity requires independent tests.
Consumes only sanitized public view dictionaries and legal action structures.
Raw action/item IDs/booleans are extracted into compact token tables on host,
while card template gathering, mulligan scatter, option one-hot, context arithmetic,
and physical identity one-hot are executed purely via PyTorch tensor operations on GPU.

No CPU fallback: cuda unavailable raises RuntimeError unless device='cpu' is explicitly specified.
Module import does not unconditionally require PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from app.modules.card_game.content.duel_v2.catalog import CARDS
from app.modules.card_game.rl import cross_lineup
from app.modules.card_game.rl.league_schema import (
    ACT_ATTACK,
    ACT_CHOOSE,
    ACT_END,
    ACT_MULLIGAN,
    ACT_PLAY,
    ACT_ULTIMATE,
    OPTION_IDS,
)

# Contract constants
CARD_IDS: tuple[str, ...] = cross_lineup.CARD_IDS
CARD_INDEX: dict[str, int] = cross_lineup.CARD_INDEX
NUM_CARDS: int = len(CARD_IDS)

SEATS: tuple[str, ...] = cross_lineup.SEATS
SEAT_INDEX: dict[str, int] = cross_lineup.SEAT_INDEX
NUM_SEATS: int = len(SEATS)

OPTION_IDS_TUPLE: tuple[str, ...] = OPTION_IDS
OPTION_INDEX: dict[str, int] = {opt: idx for idx, opt in enumerate(OPTION_IDS_TUPLE)}
NUM_OPTIONS: int = len(OPTION_IDS_TUPLE)

CANDIDATE_CONTEXT: tuple[str, ...] = cross_lineup.CANDIDATE_CONTEXT
NUM_CONTEXT: int = len(CANDIDATE_CONTEXT)

CAND_DIM: int = cross_lineup.CAND_DIM
CANDIDATE_NAMES: list[str] = cross_lineup.candidate_names()

ACTION_TYPE_MAP: dict[str, int] = {
    'end_turn': ACT_END,
    'attack': ACT_ATTACK,
    'play_card': ACT_PLAY,
    'ultimate': ACT_ULTIMATE,
    'mulligan': ACT_MULLIGAN,
    'choose': ACT_CHOOSE,
}

# Column slice offsets
BASE_START: int = 0
BASE_LEN: int = 11

MULLIGAN_START: int = 11
MULLIGAN_LEN: int = NUM_CARDS

OPTION_START: int = MULLIGAN_START + MULLIGAN_LEN
OPTION_LEN: int = NUM_OPTIONS

MARK_START: int = OPTION_START + OPTION_LEN
MARK_LEN: int = 4

CONTEXT_START: int = CAND_DIM - NUM_CARDS - NUM_CONTEXT
CONTEXT_LEN: int = NUM_CONTEXT

PHYSICAL_START: int = CAND_DIM - NUM_CARDS
PHYSICAL_LEN: int = NUM_CARDS

MAX_MULLIGAN_SELECTIONS: int = 16

# Stable schema SHA
SCHEMA_SHA: str = hashlib.sha256(
    f"{cross_lineup.SCHEMA}:{CAND_DIM}:{','.join(CANDIDATE_NAMES)}".encode('utf-8')
).hexdigest()

_TEMPLATE_CACHE: dict[str, Any] = {}


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for gpu_candidates operations")
    return torch


def _resolve_device(device: Optional[Union[str, Any]] = None) -> Any:
    t = _require_torch()
    if device is None:
        if not t.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available and no explicit device was provided (no CPU fallback)"
            )
        return t.device("cuda",t.cuda.current_device())

    dev = t.device(device) if isinstance(device, str) else device
    if dev.type == "cuda":
        if not t.cuda.is_available():
            raise RuntimeError(f"CUDA device '{device}' requested but CUDA is not available")
        if dev.index is None:dev=t.device('cuda',t.cuda.current_device())
    elif dev.type != "cpu":
        raise ValueError('Only explicit CPU oracle or CUDA is supported')
    return dev


def _rewrite_character_id(cid: Optional[str]) -> Optional[str]:
    if isinstance(cid, str) and cid.startswith('summon_'):
        return 'summon'
    return cid


def _parse_target_id(target_id: Optional[str], viewer: str) -> Tuple[int, int]:
    """Parse target_id into (target_side, target_seat_idx).

    target_side: 0 if own side, 1 if opponent side, -1 if no target.
    target_seat_idx: SEAT_INDEX if seat, -1 if player or no target.
    """
    if target_id is None:
        return -1, -1
    if not isinstance(target_id, str) or ':' not in target_id:
        raise ValueError(f"Invalid target_id format: {target_id}")
    side, cid = target_id.split(':', 1)
    if side not in ('a', 'b'):
        raise ValueError(f"Unknown target side in target_id: {target_id}")
    target_side = int(side != viewer)
    cid_rewritten = _rewrite_character_id(cid)
    if cid_rewritten == 'player':
        target_seat_idx = -1
    elif cid_rewritten in SEAT_INDEX:
        target_seat_idx = SEAT_INDEX[cid_rewritten]
    else:
        raise ValueError(f"Unknown target character in target_id: {cid}")
    return target_side, target_seat_idx


def get_card_template_table(device: Any) -> Any:
    """Return static card template tensor on device, initialized once and cached.

    Shape: [NUM_CARDS + 1, 6]
    Columns: [cost, attack, shield, hp, instant, derived]
    Row index NUM_CARDS is dummy zero row for card_kind_idx == -1.
    """
    t = _require_torch()
    dev_key = str(device)
    if dev_key not in _TEMPLATE_CACHE:
        raw = np.zeros((NUM_CARDS + 1, 6), dtype=np.float32)
        for i, cid in enumerate(CARD_IDS):
            card = CARDS[cid]
            raw[i, 0] = float(int(card.get('cost') or 0))
            raw[i, 1] = float(int(card.get('attack') or 0))
            raw[i, 2] = float(int(card.get('shield') or 0))
            raw[i, 3] = float(int(card.get('hp') or 0))
            raw[i, 4] = float(int(bool(card.get('instant'))))
            raw[i, 5] = float(int(bool(card.get('derived'))))
        _TEMPLATE_CACHE[dev_key] = t.tensor(raw, dtype=t.float32, device=device)
    return _TEMPLATE_CACHE[dev_key]


@dataclass(frozen=True)
class PackedCandidateBatch:
    """Padded torch candidate batch tensor and associated metadata."""

    candidates: Any  # torch.Tensor [N, M, CAND_DIM], float32
    mask: Any        # torch.Tensor [N, M], bool
    counts: Any      # torch.Tensor [N], int64
    schema_sha: str  # Hex digest representing candidate schema
    device: Any      # torch.device
    request_ids: tuple = ()

    @property
    def batch_size(self) -> int:
        return int(self.candidates.shape[0])

    @property
    def max_candidates(self) -> int:
        return int(self.candidates.shape[1])

    @property
    def cand_dim(self) -> int:
        return int(self.candidates.shape[2])

    def to(self, device: Any) -> PackedCandidateBatch:
        dev = _resolve_device(device)
        return PackedCandidateBatch(
            candidates=self.candidates.to(dev),
            mask=self.mask.to(dev),
            counts=self.counts.to(dev),
            schema_sha=self.schema_sha,
            device=dev,
            request_ids=self.request_ids,
        )

    def cpu(self) -> PackedCandidateBatch:
        return self.to("cpu")

    def cuda(self) -> PackedCandidateBatch:
        return self.to("cuda")


@dataclass
class RawCandidateTokens:
    """Compact raw token tables extracted from public views and actions on host.

    Can also be directly emitted by future GPU rule kernels.
    """

    # [N, M, 7]: [action_type, actor_idx, card_kind_idx, target_side, target_seat_idx, option_idx, physical_card_idx]
    int_tokens: Union[np.ndarray, Any]
    # [N, M, 8]: [revealed, marked, mark_delta, set_attack, item_instant, is_copy, raw_expires_turn, antique_investment]
    float_tokens: Union[np.ndarray, Any]
    # [N, M, 2]: [has_item, item_derived]
    item_flags: Union[np.ndarray, Any]
    # [N]: viewer's own turn count
    turn_counts: Union[np.ndarray, Any]
    # [N, M, MAX_MULLIGAN_SELECTIONS]: card_kind indices to redraw (-1 for unused)
    mulligan_cards: Union[np.ndarray, Any]
    # [N]: legal action count per item
    counts: Union[np.ndarray, Any]
    # [N, M]: validity mask
    mask: Union[np.ndarray, Any]
    request_ids: tuple = ()


def extract_raw_tokens(
    views: Sequence[Dict[str, Any]],
    actions_batch: Sequence[Sequence[Dict[str, Any]]],
    *,
    capacity: Optional[int] = None,
) -> RawCandidateTokens:
    """Extract compact token tables from public views and actions on host.

    Entity IDs (instance_id, choice_id) are used ONLY for looking up visible
    objects in the viewer's hand or pending choices, and are immediately discarded.
    """
    n = len(views)
    if len(actions_batch) != n:
        raise ValueError(f"Batch views count ({n}) != actions_batch count ({len(actions_batch)})")

    counts_arr = np.array([len(actions) for actions in actions_batch], dtype=np.int64)

    if capacity is not None:
        if type(capacity) is not int or not 0<=capacity<=16384:
            raise ValueError('Candidate capacity must be an integer in 0..16384')
        if capacity < 0:
            raise ValueError(f"Explicit capacity cannot be negative: {capacity}")
        for idx, count in enumerate(counts_arr):
            if count > capacity:
                raise ValueError(
                    f"Candidate count {count} at index {idx} exceeds explicit capacity {capacity}"
                )
        m = capacity
    else:
        m = int(counts_arr.max()) if n > 0 and len(counts_arr) > 0 else 0

    int_tokens = np.full((n, m, 7), -1, dtype=np.int64)
    float_tokens = np.zeros((n, m, 8), dtype=np.float32)
    item_flags = np.zeros((n, m, 2), dtype=np.float32)
    turn_counts = np.zeros(n, dtype=np.float32)
    mulligan_cards = np.full((n, m, MAX_MULLIGAN_SELECTIONS), -1, dtype=np.int64)
    mask = np.zeros((n, m), dtype=bool)

    for i in range(n):
        view = views[i]
        actions = actions_batch[i]
        viewer = view.get('viewer_side')
        if viewer not in ('a', 'b'):
            raise ValueError(f"Invalid viewer_side: {viewer}")

        own_side = view.get('sides', {}).get(viewer, {})
        turn_counts[i] = float(own_side.get('turn_count', 0))

        hand_cards = {c['instance_id']: c for c in own_side.get('hand', []) if 'instance_id' in c}
        pending_choices = {
            item['id']: item.get('card', {})
            for item in (view.get('pending_choice') or {}).get('choices', [])
            if 'id' in item
        }

        for j in range(len(actions)):
            a = actions[j]
            action_type_str = a.get('type')
            if action_type_str not in ACTION_TYPE_MAP:
                raise ValueError(f"Unknown action type: {action_type_str}")

            action_type = ACTION_TYPE_MAP[action_type_str]
            mask[i, j] = True

            actor_idx = -1
            card_kind_idx = -1
            target_side = -1
            target_seat_idx = -1
            revealed = 0.0

            marked = 0.0
            mark_delta = 0.0
            set_attack = 0.0
            item_instant = 0.0

            is_copy = 0.0
            raw_expires = 0.0
            item_derived = 0.0
            antique = 0.0
            has_item = 0.0
            physical_card_idx = -1

            # Option handling
            opt = a.get('option_id')
            if opt is not None:
                if opt not in OPTION_INDEX:
                    raise ValueError(f"Unknown play option: {opt}")
                option_idx = OPTION_INDEX[opt]
            else:
                option_idx = -1

            if action_type == ACT_PLAY:
                inst_id = a.get('card_id')
                if not inst_id or inst_id not in hand_cards:
                    raise ValueError(f"Card instance '{inst_id}' not found in viewer's hand")
                item = hand_cards[inst_id]

                item_char = _rewrite_character_id(item.get('character_id'))
                if item_char not in SEAT_INDEX:
                    raise ValueError(f"Unknown card character: {item.get('character_id')}")
                actor_idx = SEAT_INDEX[item_char]

                card_id = item.get('card_id')
                if card_id not in CARD_INDEX:
                    raise ValueError(f"Unknown card_id: {card_id}")
                card_kind_idx = CARD_INDEX[card_id]

                target_side, target_seat_idx = _parse_target_id(a.get('target_id'), viewer)
                revealed = float(bool(item.get('revealed')))

                mark = item.get('jingu_mark')
                if mark is not None:
                    marked = 1.0
                    mark_delta = float(mark.get('delta', 0))
                set_attack = float(item.get('attack_mode') == 'set')
                item_instant = float(bool(item.get('instant')))

                is_copy = float(bool(item.get('copy')))
                raw_expires = float(int(item.get('expires_turn') or 0))
                item_derived = float(bool(item.get('derived')))
                antique = float(item.get('antique_investment') or 0.0)
                has_item = 1.0

                face = item.get('hand_face') or {}
                face_cid = face.get('card_id')
                if face_cid and face_cid != card_id:
                    if face_cid not in CARD_INDEX:
                        raise ValueError(f"Unknown hand_face card_id: {face_cid}")
                    physical_card_idx = CARD_INDEX[face_cid]

            elif action_type == ACT_CHOOSE:
                ch_id = a.get('choice_id')
                if not ch_id or ch_id not in pending_choices:
                    raise ValueError(f"Choice id '{ch_id}' not found in pending choices")
                item = pending_choices[ch_id]

                card_id = item.get('card_id')
                if card_id not in CARD_INDEX:
                    raise ValueError(f"Unknown choice card_id: {card_id}")
                card_kind_idx = CARD_INDEX[card_id]

                revealed = float(bool(item.get('revealed')))
                is_copy = float(bool(item.get('copy')))
                raw_expires = float(int(item.get('expires_turn') or 0))
                item_derived = float(bool(item.get('derived')))
                antique = float(item.get('antique_investment') or 0.0)
                has_item = 1.0

                face = item.get('hand_face') or {}
                face_cid = face.get('card_id')
                if face_cid and face_cid != card_id:
                    if face_cid not in CARD_INDEX:
                        raise ValueError(f"Unknown choice hand_face card_id: {face_cid}")
                    physical_card_idx = CARD_INDEX[face_cid]

            elif action_type in (ACT_ATTACK, ACT_ULTIMATE):
                char_id = _rewrite_character_id(a.get('character_id'))
                if char_id not in SEAT_INDEX:
                    raise ValueError(f"Unknown action character: {a.get('character_id')}")
                actor_idx = SEAT_INDEX[char_id]
                target_side, target_seat_idx = _parse_target_id(a.get('target_id'), viewer)

            elif action_type == ACT_MULLIGAN:
                if len(a.get('card_ids',[]))>MAX_MULLIGAN_SELECTIONS:raise ValueError('Mulligan payload exceeds token capacity')
                for sel_idx, sel_inst_id in enumerate(a.get('card_ids', [])):
                    if sel_inst_id not in hand_cards:
                        raise ValueError(f"Mulligan card instance '{sel_inst_id}' not found in hand")
                    m_card_id = hand_cards[sel_inst_id].get('card_id')
                    if m_card_id not in CARD_INDEX:
                        raise ValueError(f"Unknown mulligan card_id: {m_card_id}")
                    if sel_idx < MAX_MULLIGAN_SELECTIONS:
                        mulligan_cards[i, j, sel_idx] = CARD_INDEX[m_card_id]

            elif action_type == ACT_END:
                pass

            int_tokens[i, j, 0] = action_type
            int_tokens[i, j, 1] = actor_idx
            int_tokens[i, j, 2] = card_kind_idx
            int_tokens[i, j, 3] = target_side
            int_tokens[i, j, 4] = target_seat_idx
            int_tokens[i, j, 5] = option_idx
            int_tokens[i, j, 6] = physical_card_idx

            float_tokens[i, j, 0] = revealed
            float_tokens[i, j, 1] = marked
            float_tokens[i, j, 2] = mark_delta
            float_tokens[i, j, 3] = set_attack
            float_tokens[i, j, 4] = item_instant
            float_tokens[i, j, 5] = is_copy
            float_tokens[i, j, 6] = raw_expires
            float_tokens[i, j, 7] = antique

            item_flags[i, j, 0] = has_item
            item_flags[i, j, 1] = item_derived

    return RawCandidateTokens(
        int_tokens=int_tokens,
        float_tokens=float_tokens,
        item_flags=item_flags,
        turn_counts=turn_counts,
        mulligan_cards=mulligan_cards,
        counts=counts_arr,
        mask=mask,
    )


def encode_tokens(
    tokens: RawCandidateTokens,
    *,
    device: Optional[Union[str, Any]] = None,
) -> PackedCandidateBatch:
    """Encode RawCandidateTokens purely on GPU via PyTorch tensor operations.

    No host candidate matrix creation. Emits PackedCandidateBatch with finite checks.
    """
    t = _require_torch()
    dev = _resolve_device(device)

    if not isinstance(tokens,RawCandidateTokens):raise TypeError('RawCandidateTokens required')
    for value in (tokens.int_tokens,tokens.mulligan_cards,tokens.counts):
        if isinstance(value,np.ndarray):valid=value.dtype in (np.dtype('int32'),np.dtype('int64'))
        else:valid=isinstance(value,t.Tensor) and value.dtype in (t.int32,t.int64)
        if not valid:raise ValueError('Integer candidate ID/count buffers required')
    if isinstance(tokens.mask,np.ndarray):valid=tokens.mask.dtype==np.bool_
    else:valid=isinstance(tokens.mask,t.Tensor) and tokens.mask.dtype==t.bool
    if not valid:raise ValueError('Boolean candidate mask required')
    # Ingress token buffers to torch tensors on device
    if isinstance(tokens.int_tokens, np.ndarray):
        int_tok = t.as_tensor(tokens.int_tokens, dtype=t.int64, device=dev)
        flt_tok = t.as_tensor(tokens.float_tokens, dtype=t.float32, device=dev)
        flags = t.as_tensor(tokens.item_flags, dtype=t.float32, device=dev)
        turns = t.as_tensor(tokens.turn_counts, dtype=t.float32, device=dev)
        mull_cards = t.as_tensor(tokens.mulligan_cards, dtype=t.int64, device=dev)
        counts = t.as_tensor(tokens.counts, dtype=t.int64, device=dev)
        mask = t.as_tensor(tokens.mask, dtype=t.bool, device=dev)
    else:
        int_tok = tokens.int_tokens.to(dev, dtype=t.int64)
        flt_tok = tokens.float_tokens.to(dev, dtype=t.float32)
        flags = tokens.item_flags.to(dev, dtype=t.float32)
        turns = tokens.turn_counts.to(dev, dtype=t.float32)
        mull_cards = tokens.mulligan_cards.to(dev, dtype=t.int64)
        counts = tokens.counts.to(dev, dtype=t.int64)
        mask = tokens.mask.to(dev, dtype=t.bool)

    if int_tok.ndim!=3 or int_tok.shape[2]!=7:raise ValueError('Candidate token shape must be N,M,7')
    n,m=int_tok.shape[:2]
    expected=((flt_tok,(n,m,8)),(flags,(n,m,2)),(turns,(n,)),(counts,(n,)),(mask,(n,m)))
    if any(tuple(value.shape)!=shape for value,shape in expected):raise ValueError('Candidate token buffers are misaligned')
    if mull_cards.ndim!=3 or tuple(mull_cards.shape[:2])!=(n,m) or mull_cards.shape[2]>MAX_MULLIGAN_SELECTIONS:
        raise ValueError('Invalid mulligan token shape')
    if not bool(t.isfinite(flt_tok).all() and t.isfinite(flags).all() and t.isfinite(turns).all()):
        raise ValueError('Non-finite raw candidate token')
    if bool(((counts<0)|(counts>m)).any()) or not t.equal(mask,t.arange(m,device=dev)[None,:]<counts[:,None]):
        raise ValueError('Candidate count/mask mismatch')
    limits=(5,len(SEAT_INDEX)-1,NUM_CARDS-1,1,len(SEAT_INDEX)-1,NUM_OPTIONS-1,NUM_CARDS-1)
    for col,maximum in enumerate(limits):
        if bool(((int_tok[:,:,col]<-1)|(int_tok[:,:,col]>maximum)).any()):raise ValueError('Candidate token ID outside schema')
    if bool((mask&(int_tok[:,:,0]<0)).any()):raise ValueError('Unknown active action token')
    if bool(((mull_cards<-1)|(mull_cards>=NUM_CARDS)).any()):raise ValueError('Invalid mulligan card token')
    ids=tuple(tokens.request_ids)
    if not ids:
        epoch=uuid4().hex;ids=tuple(epoch+':'+str(i) for i in range(n))
    if len(ids)!=n or len(set(ids))!=n or any(not isinstance(x,str) or not x for x in ids):
        raise ValueError('Versioned unique candidate request IDs required')
    n, m = int_tok.shape[0], int_tok.shape[1]
    if n == 0 or m == 0:
        candidates = t.zeros((n, m, CAND_DIM), dtype=t.float32, device=dev)
        return PackedCandidateBatch(
            candidates=candidates,
            mask=mask,
            counts=counts,
            schema_sha=SCHEMA_SHA,
            device=dev,
            request_ids=ids,
        )

    card_templates = get_card_template_table(dev)

    # 1. Unpack integer tokens
    action_type = int_tok[:, :, 0]
    actor_seat_idx = int_tok[:, :, 1]
    card_kind_idx = int_tok[:, :, 2]
    target_side = int_tok[:, :, 3]
    target_seat_idx = int_tok[:, :, 4]
    option_idx = int_tok[:, :, 5]
    physical_card_idx = int_tok[:, :, 6]

    # 2. Unpack float tokens
    revealed = flt_tok[:, :, 0]
    marked = flt_tok[:, :, 1]
    mark_delta = flt_tok[:, :, 2]
    set_attack = flt_tok[:, :, 3]
    item_instant = flt_tok[:, :, 4]
    is_copy = flt_tok[:, :, 5]
    raw_expires = flt_tok[:, :, 6]
    antique = flt_tok[:, :, 7]

    has_item = flags[:, :, 0]
    item_derived = flags[:, :, 1]

    # 3. Gather static card template attributes on GPU
    clamped_card_idx = t.where(
        card_kind_idx >= 0,
        card_kind_idx,
        t.tensor(NUM_CARDS, dtype=t.int64, device=dev),
    )
    gathered = card_templates[clamped_card_idx]  # [N, M, 6]
    card_cost = gathered[:, :, 0]
    card_attack = gathered[:, :, 1]
    card_shield = gathered[:, :, 2]
    card_hp = gathered[:, :, 3]
    card_instant = gathered[:, :, 4]
    static_derived = gathered[:, :, 5]

    candidates = t.zeros((n, m, CAND_DIM), dtype=t.float32, device=dev)

    # 4. Fill base features (cols 0..10)
    is_mulligan = (action_type == ACT_MULLIGAN)
    col0 = action_type.float()
    col1 = t.where(is_mulligan, t.tensor(-1.0, device=dev), actor_seat_idx.float())
    col2 = t.where(is_mulligan, t.tensor(-1.0, device=dev), card_kind_idx.float())
    col3 = t.where(is_mulligan, t.tensor(-1.0, device=dev), target_side.float())
    col4 = t.where(is_mulligan, t.tensor(-1.0, device=dev), target_seat_idx.float())
    col5 = t.where(is_mulligan, t.tensor(0.0, device=dev), card_cost)
    col6 = t.where(is_mulligan, t.tensor(0.0, device=dev), card_attack)
    col7 = t.where(is_mulligan, t.tensor(0.0, device=dev), card_shield)
    col8 = t.where(is_mulligan, t.tensor(0.0, device=dev), card_hp)
    col9 = t.where(is_mulligan, t.tensor(0.0, device=dev), card_instant)
    col10 = t.where(is_mulligan, t.tensor(0.0, device=dev), revealed)

    candidates[:, :, 0:11] = t.stack(
        [col0, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10],
        dim=-1,
    )

    # 5. Mulligan card count scatter on GPU (cols 11..11+NUM_CARDS)
    mulligan_slice = candidates[:, :, MULLIGAN_START:MULLIGAN_START + NUM_CARDS]
    valid_mull=(mull_cards>=0)&(mull_cards<NUM_CARDS)
    mulligan_slice.scatter_add_(2,mull_cards.clamp(min=0),valid_mull.float()*0.5)

    # 6. Option one-hot on GPU (cols OPTION_START..OPTION_START+NUM_OPTIONS)
    option_slice = candidates[:, :, OPTION_START:OPTION_START + NUM_OPTIONS]
    valid_opt = (option_idx >= 0) & (option_idx < NUM_OPTIONS)
    clamped_opt = t.clamp(option_idx, min=0)
    option_slice.scatter_(
        dim=2,
        index=clamped_opt.unsqueeze(-1),
        src=valid_opt.unsqueeze(-1).float(),
    )

    # 7. Dynamic mark / delta / set_attack / instant (cols MARK_START..MARK_START+4)
    candidates[:, :, MARK_START:MARK_START + 4] = t.stack(
        [marked, mark_delta, set_attack, item_instant],
        dim=-1,
    )

    # 8. Candidate context (cols CONTEXT_START..CONTEXT_START+NUM_CONTEXT)
    turn_count_2d = turns.view(n, 1)
    expires_in = t.clamp(raw_expires - turn_count_2d, min=0.0) / 10.0
    derived_diff = (item_derived != static_derived).float()
    context_block = t.stack(
        [is_copy, expires_in, derived_diff, antique],
        dim=-1,
    ) * has_item.unsqueeze(-1)
    candidates[:, :, CONTEXT_START:CONTEXT_START + NUM_CONTEXT] = context_block

    # 9. Physical original identity one-hot on GPU (cols PHYSICAL_START..CAND_DIM)
    physical_slice = candidates[:, :, PHYSICAL_START:PHYSICAL_START + NUM_CARDS]
    valid_phys = (physical_card_idx >= 0) & (physical_card_idx < NUM_CARDS) & (has_item > 0.5)
    clamped_phys = t.clamp(physical_card_idx, min=0)
    physical_slice.scatter_(
        dim=2,
        index=clamped_phys.unsqueeze(-1),
        src=valid_phys.unsqueeze(-1).float(),
    )

    # 10. Mask out padded candidates
    candidates = candidates * mask.unsqueeze(-1).float()

    # 11. Finite check
    if not t.isfinite(candidates).all():
        raise ValueError("Non-finite candidate features detected in GPU encoding")

    return PackedCandidateBatch(
        candidates=candidates,
        mask=mask,
        counts=counts,
        schema_sha=SCHEMA_SHA,
        device=dev,
        request_ids=ids,
    )


def encode_candidate_batch(
    views: Sequence[Dict[str, Any]],
    actions_batch: Sequence[Sequence[Dict[str, Any]]],
    *,
    capacity: Optional[int] = None,
    device: Optional[Union[str, Any]] = None,
) -> PackedCandidateBatch:
    """Batch GPU candidate action encoder.

    Extracts compact raw tokens on host and encodes candidate action matrices
    on GPU via PyTorch tensor operations.
    """
    tokens = extract_raw_tokens(views, actions_batch, capacity=capacity)
    return encode_tokens(tokens, device=device)


def encode_single(
    view: Dict[str, Any],
    actions: Sequence[Dict[str, Any]],
    *,
    capacity: Optional[int] = None,
    device: Optional[Union[str, Any]] = None,
) -> PackedCandidateBatch:
    """Convenience helper for encoding a single (view, actions) pair."""
    return encode_candidate_batch([view], [actions], capacity=capacity, device=device)
