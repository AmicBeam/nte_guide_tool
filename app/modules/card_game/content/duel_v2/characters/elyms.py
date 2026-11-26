"""Test-only Elyms kit; named cards intentionally have no special effects."""
from random import Random

from .common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.modifiers import add_effect, deadline
from app.modules.card_game.engine.duel_v2.state import alive, hero, other, team_order


def _empty(c):
    pass


@register
class Elyms:
    id = 'elyms'
    text_name = '曳尔沐'
    effects = {**{f'MU{i:02d}': _empty for i in range(1, 6)},
               **{f'MU{i:02d}': set_shape for i in range(6, 9)}}

    @staticmethod
    def on_game_start(c):
        c.character['signal_level'] = 0

    @staticmethod
    def delay_renewals(c, current=0, **_data):
        return max(1, int(current))

    @staticmethod
    def on_surplus(c, **_data):
        foe = other(c.side)
        candidates = [cid for cid in team_order(c.state, foe) if alive(c.state, foe, cid)] + ['player']
        rng = Random(c.state['rng'])
        cid = rng.choice(candidates)
        c.state['rng'] = rng.getrandbits(63)
        target = c.state['sides'][foe] if cid == 'player' else hero(c.state, foe, cid)
        add_effect(target, 'elyms.resistance', 'damage.resistance', -1,
                   stacking='replace', stack_key='elyms.resistance:光',
                   source_entity_id=c.entity_id, source_key='surplus',
                   expiry=deadline(c.state, c.side, 1), label='光抗性 -1',
                   conditions=[{'id': 'damage_attribute', 'args': {'attribute': '光'}}])
        c.history('【盈蓄】强化：光属性抗性 -1，持续到下个己方回合开始时。',
                  target_side=foe, target_id=cid)

    @staticmethod
    def on_ultimate(c):
        level = max(0, min(2, int(c.character.get('signal_level', 0))))
        c.character['signal_level'] = 0
        for cid in team_order(c.state, c.side):
            if not alive(c.state, c.side, cid) or hero(c.state, c.side, cid).get('summoned'):
                continue
            add_effect(hero(c.state, c.side, cid), 'elyms.ultimate_attack', 'attack.panel', level,
                       stacking='replace', source_entity_id=c.entity_id, source_key='ultimate',
                       label='曳尔沐终结', expiry=deadline(c.state, c.side, 2), show_marker=bool(level))
        c.history(f'信号强度重置为低信号，降低 {level} 级；全体己方异能者攻击 +{level}，持续 2 个己方回合。')
