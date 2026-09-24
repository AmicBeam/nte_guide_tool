"""Classify archived end-turn labels. This module does not replay games or train.

A leftover of 2 action points is not automatically a bad label. Leaving 1 point
can pay a later response; leaving 2 is a different fact. Search selection and
the improved-policy top action are recorded separately and neither one is
treated as proof that ending was best.
"""

CLASSES = ('explained', 'suspicious', 'insufficient')


def ap_bucket(ap):
    ap = int(ap)
    if ap < 0:
        raise ValueError('Negative action points')
    if ap >= 3:
        return '3+'
    return str(ap)


def turn_stage(own_turn, global_turn):
    """Own-turn buckets. Global turn 1 is the only 1-point opening turn."""
    if int(global_turn) == 1:
        return 'first_opening'
    if int(own_turn) <= 1:
        return 'second_opening'
    if int(own_turn) <= 4:
        return 'own_2_to_4'
    if int(own_turn) <= 8:
        return 'own_5_to_8'
    return 'own_9_plus'


def _combat(option):
    if option.get('kind') == 'attack':
        return True
    return option.get('kind') == 'play_card' and option.get('card_type') == 'battle'


def _paying(option):
    return int(option.get('ap_cost') or 0) >= 1 and option.get('kind') in ('attack', 'play_card')


def _safe_combat(option):
    """Immediate damage, attacker still standing, and no simultaneous counter damage."""
    return (_paying(option) and _combat(option) and int(option.get('damage') or 0) > 0
            and option.get('survives') and int(option.get('counter') or 0) == 0
            and not option.get('immune_target'))


def _lethal(option):
    return _paying(option) and _combat(option) and not option.get('survives')


def _immune_blank(option):
    return _paying(option) and _combat(option) and option.get('immune_target') and int(option.get('damage') or 0) == 0


def _zero_damage(option):
    return _paying(option) and _combat(option) and int(option.get('damage') or 0) == 0 and not option.get('immune_target')


def _trade(option):
    return (_paying(option) and _combat(option) and option.get('survives')
            and int(option.get('counter') or 0) > 0 and int(option.get('damage') or 0) > 0
            and not option.get('immune_target'))


def _unsimulated_card(option):
    return _paying(option) and option.get('kind') == 'play_card' and option.get('card_type') != 'battle'


def classify_leftover(options):
    """Classify one real end that still had an attack or a card.

    ``explained`` requires every action point that could be spent to be either
    absent or blocked by immunity, a knockdown, or a zero-damage combat.
    A card the archive cannot simulate keeps the decision in ``insufficient``.
    One safe combat makes the end ``suspicious``; it does not prove the end is wrong.
    """
    options = list(options or [])
    paying = [option for option in options if _paying(option)]
    if any(_safe_combat(option) for option in paying):
        return 'suspicious', ['safe_immediate_combat']
    if any(_unsimulated_card(option) for option in paying):
        return 'insufficient', ['unsimulated_card']
    if any(_trade(option) for option in paying):
        return 'insufficient', ['counter_trade']
    if any(_zero_damage(option) for option in paying):
        return 'insufficient', ['zero_damage_combat']
    if not paying:
        return 'explained', ['only_nonspending_action']
    reasons = []
    if any(_lethal(option) for option in paying):
        reasons.append('counter_knockdown')
    if any(_immune_blank(option) for option in paying):
        reasons.append('immune_front')
    if reasons and all(_lethal(option) or _immune_blank(option) for option in paying):
        return 'explained', reasons
    return 'insufficient', reasons or ['unclassified_option']


def count_ends(rows):
    """One playing-phase end action is one turn. Policy preference is not a second turn."""
    seen = set()
    kept = []
    for row in rows:
        if row.get('phase') != 'playing' or row.get('selected') != 'end_turn':
            continue
        key = (row['corpus'], row['game_id'], row['step'])
        if key in seen:
            raise ValueError(f'Duplicate end decision {key}')
        seen.add(key)
        kept.append(row)
    return kept


def question_rows(rows):
    """Real training turns that ended on exactly 2 points with an attack or a card still legal."""
    return [row for row in count_ends(rows)
            if row['corpus'] == 'training' and int(row['ap']) == 2 and row.get('has_attack_or_play')]
