"""Evaluate configured rotation windows at request time."""
from datetime import datetime, time, timedelta, timezone

BEIJING = timezone(timedelta(hours=8))


def current_time():
    return datetime.now(timezone.utc)


def is_released_challenge(challenge, now=None):
    """Keep permanent and already-started seasons; hide unknown release dates."""
    if challenge.get('is_permanent'):
        return True
    try:
        begin = datetime.combine(datetime.strptime(challenge['begin_date'], '%Y-%m-%d').date(),
                                 time(5), tzinfo=BEIJING)
    except (KeyError, TypeError, ValueError):
        return False
    return begin <= (now if now is not None else current_time())



def _period_boundary(record, side, kind=None):
    """Prefer explicit zoned timestamps; otherwise use the category's date precision."""
    stamp = record.get(side + '_at')
    if stamp:
        boundary = datetime.fromisoformat(stamp)
        if boundary.tzinfo is None:
            raise ValueError('Period timestamp must include a timezone')
        return boundary
    day = datetime.strptime(record[side + '_date'], '%Y-%m-%d').date()
    category = kind or record.get('kind')
    if category == 'challenges':
        return datetime.combine(day, time(5), tzinfo=BEIJING)
    # Date-only event ranges include the entire end date. Never guess an opening hour.
    if side == 'end':
        day += timedelta(days=1)
    return datetime.combine(day, time.min, tzinfo=BEIJING)


def is_current_period(record, now=None):
    """Highlight only an active configured window, never the newest record by default."""
    if record.get('is_permanent'):
        return False
    try:
        begin = _period_boundary(record, 'begin')
        end = _period_boundary(record, 'end')
        current = now if now is not None else current_time()
        return begin <= current < end
    except (KeyError, TypeError, ValueError):
        return False


def period_sort_key(record, kind=None):
    try:
        return _period_boundary(record, 'begin', kind)
    except (KeyError, TypeError, ValueError):
        # Undated entries (including permanent challenges) follow dated seasons.
        return datetime.min.replace(tzinfo=timezone.utc)


def is_current_challenge(challenge, now=None):
    return is_current_period({**challenge, 'kind': 'challenges'}, now)
