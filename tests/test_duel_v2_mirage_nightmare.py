"""User rules: immediate mirage, combat immunity and nightmare role targets."""
import json
import unittest

from tests import test_duel_v2_continuous_cast as continuous
from app.modules.card_game.engine.duel_v2 import apply_action, new_game, observe
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

    def additional_damage_game(self):
        deck = continuous._deck(('canhong', 'lingke', 'zero', 'iloy'), ('C', 'K', 'Z', 'Y'))
        s = new_game(seed=3, first_side='a', skip_mulligan=True,
                     decks={'a': deck, 'b': deck}, escalation=False)
        for team in s['sides'].values():
            team.update(hand=[], shield=0, hp=200)
            for h in team['characters'].values():
                h.update(hp=40, max_hp=40)
        return self.play(s, 'K08')

    def test_mirage_battle_damage_is_additional_without_changing_counter_or_resources(self):
        s = self.additional_damage_game()
        s['sides']['b']['front'] = 'zero'
        hero(s, 'a', 'canhong')['harmony'] = 0
        seq = s['event_seq']
        s = self.play(s, 'C04')
        hits = [e for e in s['events'] if e['seq'] > seq and e['type'] == 'combat']
        outgoing = next(e for e in hits if e['source'] == 'a:canhong')
        counter = next(e for e in hits if e['source'] == 'b:zero')
        self.assertEqual(outgoing['amount'], 3)  # 2 attack + K08 once.
        self.assertTrue(outgoing['additional_attack'])
        self.assertEqual(counter['amount'], 2)
        self.assertFalse(counter.get('additional_attack', False))
        h = hero(s, 'a', 'canhong')
        self.assertEqual((h['hp'], h['harmony'], h['energy']), (38, 1, 2))
        self.assertEqual(s['sides']['a']['ap'], 9)
        self.assertTrue(s['sides']['a']['normal_attack_available'])
        self.assertEqual(front_debuff(s, 'b')['dots']['etch']['stacks'], 2)

    def test_mirage_damage_classification_is_source_relative_and_does_not_double_bonus(self):
        from app.modules.card_game.content.duel_v2.characters.canhong import _enter_mirage
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        s = self.additional_damage_game()
        _enter_mirage(EffectContext(s, 'a', 'canhong', {'side': 'a'}))
        s = clone_state(json.loads(json.dumps(s)))
        for target_side, source, explicit, expected, marked in (
                ('b', 'a:canhong', False, 2, True),
                ('b', 'a:canhong', True, 2, True),
                ('b', 'a:zero', False, 1, False),
                ('a', 'b:canhong', False, 1, False)):
            with self.subTest(source=source, explicit=explicit):
                opening, seq = capture_public_board(s), s['event_seq']
                before = s['sides'][target_side]['hp']
                damage(s, target_side, 'player', 1, source=source, additional_attack=explicit)
                self.assertEqual(s['sides'][target_side]['hp'], before - expected)
                events = [e for e in s['events'] if e['seq'] > seq]
                hit = next(e for e in events if e['type'] == 'damage')
                self.assertEqual(hit.get('additional_attack', False), marked)
                for viewer in ('a', 'b'):
                    public = observe(s, viewer)['presentation']['events']
                    projected = next(e for e in public if e['seq'] == hit['seq'])
                    self.assertEqual(projected.get('additional_attack', False), marked)
                self.assertEqual(replay_public_board(opening, events), capture_public_board(s))

    def test_mirage_noncombat_weapon_damage_gets_additional_bonus_and_expiry_clears_it(self):
        from app.modules.card_game.engine.duel_v2.flow import begin_turn
        s = self.additional_damage_game()
        s = self.play(s, 'C06')
        s = self.play(s, 'C04')
        seq = s['event_seq']
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        hit = next(e for e in s['events'] if e['seq'] > seq and e['type'] == 'damage'
                   and e.get('source') == 'a:canhong')
        self.assertEqual(hit['amount'], 2)
        self.assertTrue(hit['additional_attack'])
        for _ in range(3):
            begin_turn(s, 'a')
        damage(s, 'b', 'player', 1, source='a:canhong')
        hit = next(e for e in reversed(s['events']) if e['type'] == 'damage')
        self.assertEqual(hit['amount'], 1)
        self.assertFalse(hit.get('additional_attack', False))

    def test_mirage_counter_uses_its_own_side_bonus_and_disabled_counter_stays_disabled(self):
        from app.modules.card_game.content.duel_v2.characters.canhong import _enter_mirage
        from app.modules.card_game.engine.duel_v2.combat import additional_attack
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        from app.modules.card_game.engine.duel_v2.flow import begin_turn
        for counter_enabled in (True, False):
            with self.subTest(counter=counter_enabled):
                s = self.additional_damage_game()
                s['sides']['a']['front'] = 'canhong'
                _enter_mirage(EffectContext(s, 'a', 'canhong', {'side': 'a'}))
                seq = s['event_seq']
                if counter_enabled:
                    begin_turn(s, 'b')
                    s['sides']['b']['hand'] = []
                    s = apply_action(s, 'b', {'type': 'attack', 'character_id': 'zero'})
                else:
                    additional_attack(EffectContext(s, 'b', 'zero', {'side': 'b'}), '光')
                attack = next(e for e in s['events'] if e['seq'] > seq and e['type'] == 'attack')
                counter = next(e for e in s['events'] if e['seq'] > seq and e['type'] == 'combat'
                               and e.get('source') == 'a:canhong')
                expected = 3 if counter_enabled else 0
                self.assertEqual(attack['counter'], expected)
                self.assertEqual(counter['amount'], expected)
                self.assertEqual(counter.get('additional_attack', False), counter_enabled)
                self.assertEqual(hero(s, 'b', 'zero')['hp'], 40 - expected)
                self.assertEqual(hero(s, 'a', 'canhong')['energy'], 0)

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
