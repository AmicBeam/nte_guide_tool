"""Observer-only, simulation-free Zhenhong action prior.

Consumes the existing seven-team public encoding and effective legal candidate
identities. Scores are bounded logit hints, never WDL, rewards, terminal labels
or a post-search action override. No rule engine, world sampler or VM is called.
"""
from dataclasses import dataclass
import math


VERSION = 'zhenhong_array_prior_v1'


@dataclass(frozen=True)
class ZhenhongPriorWeights:
    surplus_opportunity: float = 2.0
    delay_setup: float = 1.0
    fang_setup: float = .75
    zero_setup: float = 1.0
    draw_setup: float = .25
    resource_cost: float = .1
    opening_keep: float = 1.0
    max_logit_shift: float = 2.0

    def __post_init__(self):
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
               for v in self.__dict__.values()):
            raise ValueError('Finite nonnegative prior weights required')


@dataclass(frozen=True)
class ArrayPriorResult:
    shifts: object
    errors: object
    applicable: object


class ArrayZhenhongPrior:
    """Prepared static indices; hot path is tensor operations over B and M.

    Hints deliberately describe opportunities, not predicted trigger counts:
    interrupts, harmony overrides and later draws remain the search's job.
    Both dense and compact candidates use the same effective card identity.
    """
    version = VERSION
    simulations = 0
    changes_value = False
    changes_final_action = False

    def __init__(self, *, device='cuda', weights=None, allow_cpu_reference=False):
        import torch
        from .. import seven_lineup as schema
        from ...content.duel_v2 import CARDS, CHARACTERS
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda' and not (self.device.type == 'cpu' and allow_cpu_reference):
            raise ValueError('CUDA or explicit CPU prior reference required')
        self.weights = weights or ZhenhongPriorWeights()
        if not isinstance(self.weights, ZhenhongPriorWeights):
            raise TypeError('Versioned Zhenhong prior weights required')
        self.schema = schema
        self.names = {name: i for i, name in enumerate(schema.all_features())}
        self.obs_dim, self.cand_dim = len(self.names), schema.CAND_DIM
        self.battle = torch.tensor([CARDS[c]['type'] == 'battle' for c in schema.CARD_IDS]
                                  + [False], device=self.device)
        self.attribute = torch.tensor([
            ('光', '灵', '相').index(CHARACTERS[c]['attribute']) + 1
            if c in CHARACTERS and CHARACTERS[c]['attribute'] in ('光', '灵', '相') else 0
            for c in schema.SEATS] + [0], dtype=torch.int64, device=self.device)
        self._char_cols = {key: torch.tensor([self.names[f'{key}:0:{cid}'] for cid in schema.SEATS],
                                           dtype=torch.int64, device=self.device)
                           for key in ('ch_hp', 'ch_present', 'ch_harmony', 'ch_harmony_max',
                                       'ch_collapse_until')}
        self._fang_ids = tuple(schema.CARD_INDEX[c] for c in ('I01', 'I02', 'I03', 'I04'))
        self._yi_battle_ids = tuple(schema.CARD_INDEX[c] for c in ('I01', 'I02'))
        self._zero_battle_ids = tuple(schema.CARD_INDEX[c] for c in schema.CARD_IDS
                                     if CARDS[c]['character_id'] == 'zero' and CARDS[c]['type'] == 'battle')
        self._backup_ids = tuple(schema.CARD_INDEX[c] for c in ('R01', 'Y02'))
        self._hand_cols = torch.tensor([self.names[f'cards:0:{c}'] for c in schema.CARD_IDS],
                                       dtype=torch.int64, device=self.device)
        self._yi_battle_index = torch.tensor(self._yi_battle_ids, dtype=torch.int64, device=self.device)
        self._opening_groups = tuple((ids, torch.tensor(ids, dtype=torch.int64, device=self.device))
                                     for ids in (self._fang_ids, self._zero_battle_ids))

    def _has_id(self, ids, group):
        # Static schema loop, independent of batch size and legal action count.
        out = self.t.zeros_like(ids, dtype=self.t.bool)
        for cid in group:
            out = out | (ids == cid)
        return out

    def score_device(self, observations, candidates, valid, active):
        t, s, w = self.t, self.schema, self.weights
        if (not isinstance(observations, t.Tensor) or observations.device != self.device
                or observations.dtype != t.float32 or observations.ndim != 2
                or observations.shape[1] != self.obs_dim):
            raise ValueError('Exact seven-team public observation matrix required')
        b = observations.shape[0]
        if (not isinstance(valid, t.Tensor) or valid.device != self.device or valid.dtype != t.bool
                or valid.ndim != 2 or valid.shape[0] != b):
            raise ValueError('Device legal candidate mask required')
        if (not isinstance(active, t.Tensor) or active.device != self.device
                or active.dtype != t.bool or active.shape != (b,)):
            raise ValueError('Device active rows required')
        m = valid.shape[1]
        tokens = getattr(candidates, 'tokens', None)
        if tokens is not None:
            raw = tokens.int_tokens
            if raw.shape != (b, m, 7) or raw.device != self.device or raw.dtype not in (t.int32, t.int64):
                raise ValueError('Exact compact seven candidate tokens required')
            kind, actor, card = raw[:, :, 0], raw[:, :, 1], raw[:, :, 2]
            outgoing = tokens.mulligan_cards
            cost = candidates.template_table[card.clamp(0, len(s.CARD_IDS)), 0]
            instant = ((tokens.float_tokens[:, :, 4] > .5)
                       | (candidates.template_table[card.clamp(0, len(s.CARD_IDS)), 4] > .5))
            discarded = None
        else:
            if (not isinstance(candidates, t.Tensor) or candidates.device != self.device
                    or candidates.dtype != t.float32 or candidates.shape != (b, m, self.cand_dim)):
                raise ValueError('Dense or compact seven candidates required')
            raw = candidates[:, :, :3]
            kind, actor, card = raw[:, :, 0].long(), raw[:, :, 1].long(), raw[:, :, 2].long()
            cost = candidates[:, :, 5]
            instant = ((candidates[:, :, 9] > .5)
                       | (candidates[:, :, 11 + len(s.CARD_IDS) + len(s.old.OPTION_IDS) + 3] > .5))
            discarded = candidates[:, :, 11:11 + len(s.CARD_IDS)]
            outgoing = None
        bad = ~t.isfinite(observations).all(-1)
        bad = bad | (valid & ((kind < 0) | (kind >= len(s.ACTION_TYPES))
                    | (actor < -1) | (actor >= len(s.SEATS))
                    | (card < -1) | (card >= len(s.CARD_IDS))
                    | ~t.isfinite(cost))).any(-1)
        if tokens is None:
            bad = bad | (valid & (~t.isfinite(candidates).all(-1)
                        | (raw != raw.round()).any(-1))).any(-1)
        else:
            bad = bad | ((outgoing < -1) | (outgoing >= len(s.CARD_IDS))).flatten(1).any(-1)
        safe_x = t.where(t.isfinite(observations), observations, 0.)
        def x(name):
            return safe_x[:, self.names[name]]
        playing = (x('phase') - .1).abs() < .001
        opening = x('phase').abs() < .001
        applicable = (active & ~bad & (x('ch_present:0:zhenhong') > 0)
                      & (x('ch_hp:0:zhenhong') > 0) & (x('ch_awakened:0:zhenhong') == 0)
                      & (x('ch_present:0:yi') > 0) & (playing | opening))
        safe_actor = actor.clamp(0, len(s.SEATS)-1)
        hp = safe_x[:, self._char_cols['ch_hp']]
        living = (actor >= 0) & (hp.gather(1, safe_actor) > 0)
        front = (x('front:0') * 10).round().long()
        last = (x('last_front:0') * 10).round().long()
        payer = t.where(front >= 0, front, last)
        safe_payer = payer.clamp(0, len(s.SEATS)-1)[:, None]
        harmony = safe_x[:, self._char_cols['ch_harmony']].gather(1, safe_payer)[:, 0]
        cap = safe_x[:, self._char_cols['ch_harmony_max']].gather(1, safe_payer)[:, 0]
        collapse = safe_x[:, self._char_cols['ch_collapse_until']].gather(1, safe_payer)[:, 0]
        ready = ((payer >= 0) & (hp.gather(1, safe_payer)[:, 0] > 0)
                 & (cap > 0) & (harmony >= cap) & (collapse == 0))
        payer_attr = self.attribute[payer.clamp(0, len(s.SEATS))][:, None]
        actor_attr = self.attribute[safe_actor]
        pair_light = (payer_attr == 1) | (actor_attr == 1)
        genesis = pair_light & ((payer_attr == 2) | (actor_attr == 2))
        delay_pair = pair_light & ((payer_attr == 3) | (actor_attr == 3))
        battle = self.battle[card.clamp(0, len(s.CARD_IDS))] & (card >= 0)
        sortie = ((kind == s.ACTION_TYPES.index('attack'))
                  | ((kind == s.ACTION_TYPES.index('play_card')) & battle)) & living
        entry = sortie & (actor != payer[:, None]) & ready[:, None]
        delayed = x('fixed:delay_left:1') > 0
        fangs = x('fixed:beast_fangs:0:yi') > 0
        play = kind == s.ACTION_TYPES.index('play_card')
        fang_card = play & self._has_id(card, self._fang_ids) & living
        zero_battle = play & battle & (actor == s.SEAT_INDEX['zero']) & living
        hand = safe_x.index_select(1, self._hand_cols)
        has_yi_battle = hand.index_select(1, self._yi_battle_index).sum(-1) > 0
        surplus = entry & genesis & delayed[:, None]
        delay_setup = entry & delay_pair & ~delayed[:, None]
        fang_setup = fang_card & ~fangs[:, None]
        zero_setup = (zero_battle & ~delayed[:, None] & has_yi_battle[:, None]
                      & ((x('ap:0') >= .2) & (x('ch_hp:0:yi') > 0))[:, None])
        draw_setup = play & self._has_id(card, self._backup_ids) & (~fangs | ~delayed)[:, None]
        score = (w.surplus_opportunity * surplus + w.delay_setup * delay_setup
                 + w.fang_setup * fang_setup + w.zero_setup * zero_setup
                 + w.draw_setup * draw_setup).to(t.float32)
        # Cost is a tradeoff only on hinted preparation; no universal end-turn
        # penalty or survival rule that overrides the learned game objective.
        paid = t.where(instant & (x('used_instant:0') == 0)[:, None], 0., cost.clamp_min(0))
        score = t.where(score > 0, score - w.resource_cost * paid, score)
        score = t.where(playing[:, None], score, 0.)
        # Opening preference keeps a core piece, not every duplicate. Counts
        # come from this player's visible hand, never unknown replacement draws.
        opening_score = t.zeros_like(score)
        for group, indices in self._opening_groups:
            held = hand.index_select(1, indices).sum(-1) * 2
            if outgoing is not None:
                removed = self._has_id(outgoing, group).sum(-1)
            else:
                removed = discarded.index_select(2, indices).sum(-1) * 2
            lost_core = (held > 0)[:, None] & (held[:, None] - removed < .5)
            opening_score = opening_score - w.opening_keep * lost_core
        score = t.where(opening[:, None] & (kind == s.ACTION_TYPES.index('mulligan')),
                        opening_score, score)
        finite = t.isfinite(score).all(-1)
        errors = ((bad | ~finite) & active).to(t.int32)
        admitted = applicable & (errors == 0)
        score = t.where(admitted[:, None] & valid, score, 0.)
        return ArrayPriorResult(score.clamp(-w.max_logit_shift, w.max_logit_shift), errors, admitted)

    def score_numpy(self, observations, candidates):
        """Explicit CPU search oracle only; never a CUDA hot-path export."""
        if self.device.type != 'cpu':
            raise RuntimeError('NumPy guidance is a CPU reference, not a device fallback')
        import numpy as np
        x = self.t.as_tensor(np.asarray(observations, dtype=np.float32)[None])
        c = self.t.as_tensor(np.asarray(candidates, dtype=np.float32)[None])
        valid = self.t.ones(c.shape[:2], dtype=self.t.bool)
        hint = self.score_device(x, c, valid, self.t.ones(1, dtype=self.t.bool))
        if bool(hint.errors.any()):
            raise ValueError('Invalid CPU-reference prior inputs')
        return hint.shifts[0].numpy().copy()
