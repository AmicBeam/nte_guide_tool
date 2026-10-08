"""User rules: immediate mirage, combat immunity and nightmare role targets."""
import json
import unittest

from tests import test_duel_v2_continuous_cast as continuous
from app.modules.card_game.engine.duel_v2 import apply_action, observe
from app.modules.card_game.engine.duel_v2.continuous import apply_dot, tick_owner_end
from app.modules.card_game.engine.duel_v2.entities import clone_state
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.state import card_instance, damage, front_debuff, heal, hero, knockdowns, reduce_max_hp
from app.modules.card_game.engine.duel_v2.summons import summon_front


class MirageNightmareTest(unittest.TestCase):
    def game(self):
        return continuous.ContinuousCastTest().game()

    def play(self, s, card_id):
        return continuous.ContinuousCastTest().play(s, card_id)

    def test_first_battle_card_enters_mirage_before_attack_and_applies_etch(self):
        s = self.game()
        s['sides']['b']['shield'] = 0
        s = self.play(s, 'C04')
        self.assertEqual(front_debuff(s, 'b')['dots']['etch']['stacks'], 2)
        mirage = [e for e in s['events'] if e['text'] == '残虹进入幻境，持续 3 个己方回合。']
        self.assertEqual(len(mirage), 1)
        attack = next(e for e in s['events'] if e['type'] == 'attack')
        self.assertLess(mirage[0]['seq'], attack['seq'])

    def test_blood_flower_adds_one_and_blocks_counter_damage_without_extra_energy(self):
        s = self.game()
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero').update(hp=20, max_hp=20, attribute='魂')
        hero(s, 'a', 'canhong')['shield'] = 2
        front_debuff(s, 'a')['stain'] = {'by': 'b', 'left': 1}
        s = self.play(s, 'C05')
        hit = next(e for e in reversed(s['events']) if e['type'] == 'attack')
        self.assertEqual((hit['attack'], hit['counter']), (3, 2))
        h = hero(s, 'a', 'canhong')
        self.assertEqual((h['hp'], h['shield'], h['energy']), (6, 2, 2))
        self.assertFalse(any(e['definition_id'] == 'canhong.combat_immune' for e in h['effects']))
        damage(s, 'a', 'canhong', 1, kind='combat')
        self.assertEqual(h['shield'], 1)

    def test_nightmare_distributes_to_player_living_esper_and_summon(self):
        reached = set()
        for seed in range(4):
            s = self.game()
            for cid in ('iloy', 'jiuyuan', 'adler'):
                hero(s, 'b', cid)['hp'] = 0
            knockdowns(s)
            hero(s, 'b', 'zero').update(hp=20, max_hp=20)
            summon = summon_front(s, 'b', name='靶子', attack=0, hp=20)
            s['sides']['b']['shield'] = 0
            s['rng'] = seed
            apply_dot(s, 'b', 'nightmare', by='a', source_cid='anhunqu', stacks=10)
            seq = s['event_seq']
            again = clone_state(json.loads(json.dumps(s)))
            tick_owner_end(s, 'a')
            tick_owner_end(again, 'a')
            hits = [e for e in s['events'] if e['seq'] > seq and e['type'] == 'damage']
            self.assertEqual(sum(e['amount'] for e in hits), 10)
            self.assertTrue(all(e['target'] in ('b:player', 'b:zero', 'b:' + summon) for e in hits))
            reached.update('summon' if e['target'].endswith(summon) else e['target'] for e in hits)
            self.assertEqual(capture_public_board(s), capture_public_board(again))
        self.assertEqual(reached, {'b:player', 'b:zero', 'summon'})

    def test_rose_reduces_player_maximum_and_healing_respects_it_in_replay(self):
        s = self.play(self.game(), 'A08')
        for cid in s['sides']['b']['order']:
            hero(s, 'b', cid)['hp'] = 0
        knockdowns(s)
        s['sides']['b'].update(hp=20, max_hp=30, shield=1)
        hero(s, 'a', 'anhunqu')['hp'] = 2
        apply_dot(s, 'b', 'nightmare', by='a', source_cid='anhunqu', stacks=3)
        card = card_instance(s, 'A03', 'a')
        s['sides']['a']['hand'] = [card]
        opening, seq = capture_public_board(s), s['event_seq']
        s = apply_action(s, 'a', {'type': 'play_card', 'card_id': card['instance_id']})
        self.assertEqual((s['sides']['b']['hp'], s['sides']['b']['max_hp'], s['sides']['b']['shield']), (15, 27, 0))
        self.assertEqual(hero(s, 'a', 'anhunqu')['hp'], 5)
        for viewer in ('a', 'b'):
            self.assertEqual(observe(s, viewer, include_previews=False)['sides']['b']['max_hp'], 27)
        self.assertEqual(replay_public_board(opening, [e for e in s['events'] if e['seq'] > seq]), capture_public_board(s))
        s = clone_state(json.loads(json.dumps(s)))
        heal(s, 'b', 'player', 99)
        self.assertEqual(s['sides']['b']['hp'], 27)

    def test_player_maximum_cut_uses_legacy_default_rounding_and_victory(self):
        s = self.game()
        s['sides']['b'].update(hp=5, max_hp=7, shield=9)
        reduce_max_hp(s, 'b', 'player', 1)
        self.assertEqual((s['sides']['b']['hp'], s['sides']['b']['max_hp'], s['sides']['b']['shield']), (4, 6, 9))
        reduce_max_hp(s, 'b', 'player', 1)
        self.assertEqual((s['sides']['b']['hp'], s['sides']['b']['max_hp']), (3, 5))
        s['sides']['b'].pop('max_hp')
        s['sides']['b']['hp'] = 20
        reduce_max_hp(s, 'b', 'player', 1)
        self.assertEqual((s['sides']['b']['hp'], s['sides']['b']['max_hp']), (19, 29))
        s['sides']['b']['life_locked'] = True
        self.assertEqual(reduce_max_hp(s, 'b', 'player', 1), 0)
        s['sides']['b'].pop('life_locked')
        reduce_max_hp(s, 'b', 'player', 29)
        self.assertEqual((s['phase'], s['winner']), ('finished', 'a'))


if __name__ == '__main__':
    unittest.main()
