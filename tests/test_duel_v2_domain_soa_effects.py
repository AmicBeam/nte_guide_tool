import unittest
import torch

from app.modules.card_game.rl.batched_duel.domain_soa_effects import (
    CONDITIONS, SCOPES, EffectEntities, EffectTable, EffectQuery,
    UNKNOWN_CONDITION, INVALID_REFERENCE, match_effects,
)


def i(value):
    return torch.tensor(value, dtype=torch.int32)


def b(value):
    return torch.tensor(value, dtype=torch.bool)


def l(value):
    return torch.tensor(value, dtype=torch.int64)


class SoAEffectMatchingTest(unittest.TestCase):
    def fixture(self):
        # Two real players and one character per side; preserve formal order.
        from app.modules.card_game.content.duel_v2.characters import adler  # register its declared condition
        from app.modules.card_game.engine.duel_v2.modifiers import contribution
        chars = {
            'zero': dict(entity_id='a-zero', hp=3, awakened=True, shape='Z08', energy=3, side='a', attribute='光'),
            'baicang': dict(entity_id='b-white', hp=0, awakened=False, shape=None, energy=0, side='b', attribute='咒'),
        }
        state = dict(active_side='b', sides={
            'a': dict(entity_id='a-player', side='a', hp=30, turn_count=3, front='zero', characters={'zero': chars['zero']}),
            'b': dict(entity_id='b-player', side='b', hp=20, turn_count=2, front=None, characters={'baicang': chars['baicang']}),
        })
        objects = [state['sides']['a'], chars['zero'], state['sides']['b'], chars['baicang']]
        identities = {obj['entity_id']: index for index, obj in enumerate(objects)}
        effects, holder_slots = [], []
        condition_args = dict(source_front={}, source_alive={}, source_awakened={},
            source_shape={'card_id': 'Z08'}, enemy_turn={},
            resource_at_least={'resource': 'energy', 'amount': 2},
            damage_attribute={'attribute': '光'}, card_owner_other={}, attribute_is={'value': '光'})
        for holder_slot in range(4):
            for scope in SCOPES:
                for condition, args in condition_args.items():
                    e = contribution('test', 'attack.panel', 1, scope=scope,
                        source_entity_id='a-zero', conditions=[dict(id=condition, args=args)])
                    effects.append(e)
                    holder_slots.append(holder_slot)
        effects.extend([
            contribution('missing', 'attack.panel', 1, source_entity_id='missing', kind='aura', scope='all'),
            contribution('expired', 'attack.panel', 1, scope='all', expiry=dict(side='a', turn=3, phase='turn_start')),
            contribution('end', 'attack.panel', 1, scope='all', expiry=dict(side='a', turn=3, phase='turn_end')),
            contribution('own', 'attack.panel', 1, scope='all', source_entity_id=None),
        ])
        holder_slots.extend([0] * 4)
        E = len(effects)
        table = EffectTable(b([[True]*E]), i([holder_slots]),
            l([[identities.get(e.get('source_entity_id'), -1 if not e.get('source_entity_id') else 99) for e in effects]]),
            i([[0]*E]), i([[SCOPES.index(e['scope']) for e in effects]]),
            b([[e['kind'] == 'aura' for e in effects]]),
            i([[{'a': 0, 'b': 1}.get((e.get('expiry') or {}).get('side'), -1) for e in effects]]),
            i([[(e.get('expiry') or {}).get('turn', 0) for e in effects]]),
            i([[{'turn_start': 0, 'turn_end': 1}.get((e.get('expiry') or {}).get('phase'), -1) for e in effects]]),
            i([[[CONDITIONS.index(e['conditions'][0]['id']) if e.get('conditions') else -1] for e in effects]]),
            i([[[0] for e in effects]]),
            i([[[2] for e in effects]]))
        entities = EffectEntities(b([[True]*4]), l([[0, 1, 2, 3]]), i([[0, 0, 1, 1]]),
            i([[30, 3, 20, 0]]), b([[False, True, False, False]]), i([[-1, 0, -1, -1]]),
            i([[[0], [3], [0], [0]]]), attribute=i([[-1, 0, -1, 1]]))
        query = EffectQuery(i([1]), i([0]), i([0]), i([1]), l([[1, -1]]), i([[3, 2]]), i([0]), b([True]))
        return state, objects, effects, holder_slots, entities, table, query

    def test_all_registered_conditions_and_scopes_match_formal(self):
        from app.modules.card_game.engine.duel_v2.modifiers import _applies, CONDITIONS as formal_conditions
        self.assertEqual(set(CONDITIONS), set(formal_conditions))
        state, objects, effects, holders, entities, table, query = self.fixture()
        for slot in range(4):
            query.subject.fill_(slot)
            query.side.fill_(0 if slot < 2 else 1)
            expected = [_applies(e, objects[h], 'a' if h < 2 else 'b', objects[slot],
                         'a' if slot < 2 else 'b', state, {'attribute': '光'}) for e, h in zip(effects, holders)]
            result = match_effects(entities, table, query)
            self.assertEqual(result.applies[0].tolist(), expected)
            self.assertEqual(result.errors.tolist(), [0])
        # Pure query stages must not change canonical effects, clocks or tensors.
        self.assertEqual(state['sides']['a']['turn_count'], 3)
        self.assertEqual(table.present.sum().item(), len(effects))

    def test_unknown_or_missing_reference_is_not_silently_dropped(self):
        _, _, _, _, entities, table, query = self.fixture()
        table.condition[0, 0, 0] = 98
        result = match_effects(entities, table, query)
        self.assertEqual(result.errors.tolist(), [UNKNOWN_CONDITION])
        self.assertFalse(result.applies.any())
        table.condition[0, 0, 0] = 0
        table.holder[0, 0] = 999
        result = match_effects(entities, table, query)
        self.assertEqual(result.errors.tolist(), [INVALID_REFERENCE])
        self.assertFalse(result.applies.any())
        query.active.zero_()
        result = match_effects(entities, table, query)
        self.assertEqual(result.errors.tolist(), [0])
        self.assertFalse(result.applies.any())

    def test_absent_identity_self_uses_object_slot_and_unknown_source_is_allowed(self):
        _, _, _, _, entities, table, query = self.fixture()
        entities.identity.fill_(-1)
        table.source_identity.fill_(-1)
        table.condition.fill_(-1)
        result = match_effects(entities, table, query)
        expected = (table.holder == query.subject[:, None])
        self_scope = table.scope == SCOPES.index('self')
        # Expiry still takes precedence for the explicit timed examples.
        self.assertEqual(result.applies[self_scope].tolist(), expected[self_scope].tolist())
        self.assertEqual(result.errors.tolist(), [0])


if __name__ == '__main__':
    unittest.main()
