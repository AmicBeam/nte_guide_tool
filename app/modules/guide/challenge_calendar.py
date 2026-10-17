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


def is_current_challenge(challenge, now=None):
    if challenge.get('is_permanent'):
        return False
    try:
        begin = datetime.combine(datetime.strptime(challenge['begin_date'], '%Y-%m-%d').date(),
                                 time(5), tzinfo=BEIJING)
        end = datetime.combine(datetime.strptime(challenge['end_date'], '%Y-%m-%d').date(),
                               time(5), tzinfo=BEIJING)
    except (KeyError, TypeError, ValueError):
        return False
    current = now if now is not None else current_time()
    return begin <= current < end
