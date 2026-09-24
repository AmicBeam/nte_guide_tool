"""Split action type from the card or actor chosen inside that type.

The coupled candidate network is not added into the score. Play actions add
the card head; every other type adds the actor head. Padding is masked the
same way as the coupled scorer, so an empty slot cannot enter the softmax.
"""
import torch
from torch import nn

from .cross_lineup import CARD_IDS, SEATS
from .league_schema import ACT_PLAY

SELECTION_KEYS = ('starter', 'weave-rush', 'zhenhong')


class FactoredActionScore(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        hidden = net.score.in_features
        self.kind_head = nn.Linear(hidden, 6)
        self.card_head = nn.Linear(hidden, len(CARD_IDS) + 1)
        self.actor_head = nn.Linear(hidden, len(SEATS) + 1)
        for head in (self.kind_head, self.card_head, self.actor_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for parameter in list(net.cand_net.parameters()) + list(net.score.parameters()):
            parameter.requires_grad_(False)

    def state_net(self, x):
        return self.net.state_net(x)

    def wdl(self, hidden):
        return self.net.wdl(hidden)

    def forward(self, x, candidates, mask):
        hidden = self.net.state_net(x)
        kind_logit = self.kind_head(hidden)
        card_logit = self.card_head(hidden)
        actor_logit = self.actor_head(hidden)
        kinds = candidates[..., 0].long().clamp(0, 5)
        cards = (candidates[..., 2].long() + 1).clamp(0, card_logit.shape[-1] - 1)
        actors = (candidates[..., 1].long() + 1).clamp(0, actor_logit.shape[-1] - 1)
        type_term = _gather(kind_logit, kinds)
        card_term = _gather(card_logit, cards)
        actor_term = _gather(actor_logit, actors)
        within = torch.where(kinds == ACT_PLAY, card_term, actor_term)
        logits = type_term + within
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min), hidden.new_zeros(hidden.shape[0])


def _gather(table, index):
    expanded = table.unsqueeze(1).expand(-1, index.shape[1], -1)
    return torch.gather(expanded, 2, index.unsqueeze(-1)).squeeze(-1)


def screening_counts(records):
    attack = [row for row in records if row['target_kind'] == 'attack']
    end = [row for row in records if row['target_kind'] == 'end_turn']
    target_play = [row for row in records if row['target_play']]
    return dict(
        n=len(records),
        attack_n=len(attack),
        attack_called_play=sum(row['model_kind'] == 'play_card' for row in attack),
        end_n=len(end),
        end_called_play=sum(row['model_kind'] == 'play_card' for row in end),
        target_play_n=len(target_play),
        same_card=sum(row['same_card'] for row in target_play),
        play_top=sum(row['play_top'] for row in records),
        end_top=sum(row['model_kind'] == 'end_turn' for row in records),
    )


def scheme_passes(by_rate):
    """One predeclared learning rate must beat the coupled scorer on every selection team.

    In-sample teams are ignored here. A higher play rate cannot create a pass,
    because ``factored_beats`` is the only flag this function reads.
    """
    for flags in by_rate.values():
        if all(bool(flags.get(key)) for key in SELECTION_KEYS):
            return True
    return False


def factored_beats(baseline, factored):
    """True only when factoring cuts both mislabels and raises same-card hits.

    A higher overall play rate is not a reason to pass. Always-play and
    always-end are failures even if the other counts move the right way.
    """
    if baseline['n'] == 0 or baseline['n'] != factored['n']:
        return False
    if min(baseline['attack_n'], baseline['end_n'], baseline['target_play_n']) == 0:
        return False
    if factored['play_top'] >= 0.95 * factored['n'] or factored['end_top'] >= 0.95 * factored['n']:
        return False
    return (factored['attack_called_play'] < baseline['attack_called_play']
            and factored['end_called_play'] < baseline['end_called_play']
            and factored['same_card'] > baseline['same_card'])
