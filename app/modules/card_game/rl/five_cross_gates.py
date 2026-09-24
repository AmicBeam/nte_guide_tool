"""Stage gates for five-preset cross training. Decisions only; no self-play."""


def schedule_gates(planned_batches):
    """Place the three gates from a batch count frozen before the run."""
    if type(planned_batches) is not int or planned_batches < 3:
        raise ValueError('Three gates need at least three planned batches')
    first = 1
    mid = max(first + 1, planned_batches // 3)
    late = max(mid + 1, (2 * planned_batches) // 3)
    if late >= planned_batches:
        late = planned_batches - 1
    if not first < mid < late < planned_batches:
        raise ValueError('Gate cycles must sit inside the planned batches')
    return dict(first=first, mid=mid, late=late)


def launch_allowed(*, seconds_available, batch_seconds, eval_seconds, gate_seconds, reserve_seconds):
    """A claimed-complete run needs the first gate, one later selection, and final eval."""
    needed = batch_seconds + gate_seconds + batch_seconds + gate_seconds + eval_seconds + reserve_seconds
    margin = 1.25 * (eval_seconds + 2 * batch_seconds)
    return seconds_available >= needed and seconds_available >= margin + reserve_seconds + gate_seconds


def first_gate(metrics, *, min_play_rows=8):
    """P0 if any team is non-finite, has no policy gradient, or never plays a legal card."""
    reasons = []
    for key, item in metrics.items():
        if not item['finite']:
            reasons.append(dict(key=key, reason='nonfinite'))
        if not item['policy_gradient']:
            reasons.append(dict(key=key, reason='no_policy_gradient'))
        if item['play_legal'] >= min_play_rows and item['play_top1'] == 0:
            reasons.append(dict(key=key, reason='no_play'))
        reference = item.get('reference_play_top1', 0)
        if reference - item['play_top1'] >= 0.10 and reference > 0 and item['play_legal'] >= min_play_rows:
            reasons.append(dict(key=key, reason='regression'))
        if item['play_legal'] < min_play_rows:
            reasons.append(dict(key=key, reason='insufficient_play_rows'))
    blocking = [row for row in reasons if row['reason'] != 'insufficient_play_rows']
    if blocking:
        return dict(decision='p0', reasons=reasons)
    if any(row['reason'] == 'insufficient_play_rows' for row in reasons):
        return dict(decision='insufficient', reasons=reasons)
    return dict(decision='pass', reasons=[])


def mid_gate(pure, screen, confirm):
    """Promote only after the pure network passes and a second seed batch confirms search."""
    if pure == 'p0':
        return 'p0'
    if pure != 'pass':
        return 'keep'
    if not screen or not confirm or not screen.get('complete') or not confirm.get('complete'):
        return 'keep'
    if screen['candidate'] > screen['best'] and confirm['candidate'] > confirm['best']:
        return 'promote'
    return 'keep'


def late_gate(pure, search_confirms, prior_no_gain, rollback_confirmed):
    """Promote, roll back to best, or stop on a second check with no material gain."""
    if pure == 'p0':
        return 'p0'
    if rollback_confirmed:
        return 'rollback'
    if prior_no_gain and pure != 'improved':
        return 'plateau'
    if pure == 'improved' and search_confirms:
        return 'promote'
    return 'keep'
