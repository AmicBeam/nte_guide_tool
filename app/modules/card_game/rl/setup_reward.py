"""Temporary training-only credit for establishing one's own delay plus beast fangs."""
import math


def ready(state, side):
    foe='b' if side=='a' else 'a'
    delay=(state['sides'][foe].get('front_debuff') or {}).get('delay') or {}
    yi=state['sides'][side]['characters'].get('yi') or {}
    return delay.get('by')==side and delay.get('left',0)>0 and yi.get('beast_fangs',0)>0


def coefficient(initial, completed_updates, decay_steps):
    if not math.isfinite(initial) or initial<0 or type(completed_updates)!=int or completed_updates<0 or type(decay_steps)!=int or decay_steps<=0:
        raise ValueError('Invalid setup reward schedule')
    return initial*max(0.,1-completed_updates/decay_steps)


def learning_return(row, setup_coefficient=None):
    base=row.get('reward_to_go')
    if base is None:base=10*row['z']
    bonus=row.get('setup_reward',0.) if setup_coefficient is None else setup_coefficient
    future=row.get('setup_future',0)
    if future not in (0,1) or not math.isfinite(bonus) or bonus<0:
        raise ValueError('Invalid setup reward target')
    return base+bonus*future
