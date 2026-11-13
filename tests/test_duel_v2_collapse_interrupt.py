"""User correction: collapse before combat damage interrupts the pending hit."""
import json
import unittest
from unittest.mock import patch

from app.modules.card_game.content.duel_v2 import CARDS, validate_deck
from app.modules.card_game.engine.duel_v2 import apply_action, legal_actions, new_game, observe
from app.modules.card_game.engine.duel_v2 import combat
from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.simulation import simulate_action
from app.modules.card_game.engine.duel_v2.state import card_instance, collapsed, force_collapse, hero


def game():
    decks = {}
    for side, ids in (('a', ['anhunqu', 'nanali', 'haiyue', 'lingke']),
                      ('b', ['dafutier', 'adler', 'zero', 'zaowu'])):
        decks[side] = validate_deck(dict(id=side, name=side, character_ids=ids,
            card_ids=[cid for owner in ids for cid, card in CARDS.items()
                      if card['character_id'] == owner and not card.get('derived')]))
    s = new_game(seed=73, first_side='a', skip_mulligan=True, decks=decks, escalation=False)
    for team in s['sides'].values():
        team.update(hand=[], ap=10, shield=0)
    s['sides']['b']['front'] = 'dafutier'
    return s


def give(s, cid, side='a'):
    card = card_instance(s, cid, side)
    s['sides'][side]['hand'].append(card)
    return dict(type='play_card', card_id=card['instance_id'])


def resources(s):
    return {(side, cid): (h['harmony'], h['energy'])
            for side, team in s['sides'].items() for cid, h in team['characters'].items()}


def paid_resources(before, *actors):
    expected = dict(before)
    for actor in actors:
        harmony, energy = expected[actor]
        expected[actor] = (min(3, harmony + 1), energy)
    return expected


class CollapseInterruptTest(unittest.TestCase):
    def test_lake_interrupts_anhunqu_before_damage_passive_and_resources(self):
        s = game()
        hero(s, 'b', 'dafutier')['hp'] = 4
        hero(s, 'a', 'anhunqu')['flags']['collapse_count'] = 2
        give(s, 'F06', 'b')
        give(s, 'D03', 'b')
        before = resources(s)
        seq = s['event_seq']
        opening = capture_public_board(s)
        action = dict(type='attack', character_id='anhunqu')
        after = apply_action(s, 'a', action)
        events = [e for e in after['events'] if e['seq'] > seq]
        self.assertTrue(collapsed(hero(after, 'a', 'anhunqu')))
        self.assertEqual(hero(after, 'b', 'dafutier')['hp'], 4)
        self.assertEqual(hero(after, 'a', 'anhunqu')['hp'], 5)
        self.assertEqual(resources(after), paid_resources(before, ('a','anhunqu'), ('b','dafutier')))
        self.assertEqual(after['sides']['a']['ap'], 9)
        self.assertFalse(after['sides']['a']['normal_attack_available'])
        self.assertEqual(after['sides']['b']['ap'], 9)
        self.assertEqual(after['sides']['a']['front'], 'anhunqu')
        self.assertEqual([c['card_id'] for c in after['sides']['b']['discard']], ['F06'])
        self.assertIn('D03', [c['card_id'] for c in after['sides']['b']['hand']])
        self.assertFalse(any(e['type'] in ('attack', 'combat', 'resource') for e in events))
        self.assertFalse(any('噩梦' in e['text'] for e in events))
        stopped = [e for e in events if '攻击被打断' in e['text']]
        self.assertEqual(len(stopped), 1)
        self.assertTrue(stopped[0]['present'])
        self.assertNotIn(hero(after, 'a', 'anhunqu').entity_id,
                         after['sides']['a'].get('attacked_this_turn', []))
        self.assertNotIn(action, legal_actions(after, 'a'))
        self.assertEqual(replay_public_board(opening, events), capture_public_board(after))
        self.assertEqual(capture_public_board(simulate_action(s, 'a', action)), capture_public_board(after))
        restored = apply_action(json.loads(json.dumps(s)), 'a', action)
        self.assertEqual(capture_public_board(restored), capture_public_board(after))

    def test_lake_below_threshold_still_allows_attack(self):
        s = game(); give(s, 'F06', 'b')
        after = apply_action(s, 'a', dict(type='attack', character_id='anhunqu'))
        self.assertFalse(collapsed(hero(after, 'a', 'anhunqu')))
        self.assertEqual(hero(after, 'b', 'dafutier')['hp'], 3)
        self.assertEqual(hero(after, 'a', 'anhunqu')['hp'], 3)
        self.assertEqual(hero(after, 'a', 'anhunqu')['energy'], 2)

    def test_interrupted_battle_card_is_spent_without_followup(self):
        s = game(); give(s, 'F06', 'b')
        hero(s, 'a', 'nanali')['flags']['collapse_count'] = 1
        action = give(s, 'N03')
        before = resources(s); seq = s['event_seq']
        after = apply_action(s, 'a', action)
        self.assertEqual(hero(after, 'b', 'dafutier')['hp'], 5)
        self.assertEqual(hero(after, 'a', 'nanali')['hp'], 5)
        self.assertEqual(resources(after), paid_resources(before, ('a','nanali'), ('b','dafutier')))
        self.assertEqual(after['sides']['a']['ap'], 9)
        self.assertEqual([c['card_id'] for c in after['sides']['a']['discard']], ['N03'])
        self.assertFalse(any(e['type'] in ('attack', 'followup') for e in after['events'] if e['seq'] > seq))
        self.assertIsNone(after['operation'])

    def test_weapon_collapses_after_normal_and_ranged_sorties(self):
        # F08 is an after-sortie effect; it does not interrupt a completed hit.
        for cid, ranged in (('anhunqu', False), ('haiyue', True)):
            with self.subTest(cid=cid):
                s = game(); hero(s, 'b', 'dafutier')['shape'] = 'F08'
                hero(s, 'a', cid)['flags']['collapse_count'] = 2
                if ranged:
                    s['sides']['a']['star_trigger_turn'] = s['turn']
                seq=s['event_seq'];action=dict(type='attack',character_id=cid)
                after=apply_action(s,'a',action)
                self.assertTrue(collapsed(hero(after,'a',cid)))
                self.assertEqual(hero(after,'b','dafutier')['hp'],2 if ranged else 3)
                self.assertEqual(hero(after,'a',cid)['harmony'],1)
                self.assertEqual(hero(after,'a',cid)['energy'],2)
                self.assertEqual(after['sides']['a']['front'],None if ranged else cid)
                self.assertEqual(sum(e['type']=='attack' for e in after['events'] if e['seq']>seq),1)

    def test_post_sortie_weapon_keeps_resolved_sync_and_borrowed_callback(self):
        s=game();hero(s,'b','dafutier')['shape']='F08'
        hero(s,'a','lingke')['flags']['collapse_count']=2
        from app.modules.card_game.content.duel_v2.characters.anhunqu import Anhunqu
        action=give(s,'K05');action['target_id']='a:anhunqu'
        with patch.object(Anhunqu,'on_support_resolved',create=True) as borrowed:
            after=apply_action(s,'a',action)
            borrowed.assert_called_once()
        self.assertTrue(collapsed(hero(after,'a','lingke')))
        self.assertEqual(hero(after,'b','dafutier')['hp'],1)
        self.assertEqual(sum(e['type']=='attack' for e in after['events']),1)

    def test_any_pre_damage_timing_can_interrupt_without_card_specific_checks(self):
        from app.modules.card_game.engine.events import GameEvent
        for timing in ('on_sortie_ready', 'before_hits', 'before_attack',
                       GameEvent.V2_OPPONENT_ATTACK, GameEvent.V2_BEFORE_ATTACK_DAMAGE):
            with self.subTest(timing=timing):
                s = game(); original = combat.fire; before = resources(s)
                def interrupt(ctx, event, **data):
                    original(ctx, event, **data)
                    if event == timing:
                        force_collapse(ctx.state, 'a', 'anhunqu', by='b')
                with patch.object(combat, 'fire', side_effect=interrupt):
                    after = apply_action(s, 'a', dict(type='attack', character_id='anhunqu'))
                self.assertTrue(collapsed(hero(after, 'a', 'anhunqu')))
                self.assertEqual(hero(after, 'b', 'dafutier')['hp'], 5)
                self.assertEqual(hero(after, 'a', 'anhunqu')['hp'], 5)
                self.assertEqual(resources(after), paid_resources(before, ('a','anhunqu')))
                self.assertFalse(any(e['type'] == 'attack' for e in after['events']))
                self.assertEqual(sum('攻击被打断' in e['text'] for e in after['events']), 1)

    def test_already_collapsed_actor_cannot_enter_or_generate_support(self):
        for additional in (False, True):
            with self.subTest(additional=additional):
                s = game(); force_collapse(s, 'a', 'anhunqu', by='b')
                c = operation(s, 'a', 'anhunqu'); before = resources(s)
                if additional:
                    combat.additional_attack(c, '暗', sync_attack=True)
                else:
                    c.sortie()
                finish_operation(s)
                self.assertIsNone(s['sides']['a']['front'])
                self.assertEqual(s['sides']['a'].get('support_count', 0), 0)
                self.assertEqual(resources(s), before)
                self.assertEqual(hero(s, 'b', 'dafutier')['hp'], 5)

    def test_later_collapse_stops_remaining_hits_without_undoing_first(self):
        s = game(); s['sides']['b']['front'] = None
        c = operation(s, 'a', 'anhunqu')
        original = combat.fire
        def collapse_after_first(ctx, timing, **data):
            original(ctx, timing, **data)
            if timing == 'after_hits':
                force_collapse(ctx.state, ctx.side, ctx.cid, by='b')
        with patch.object(combat, 'fire', side_effect=collapse_after_first):
            c.sortie(hits=2)
        finish_operation(s)
        self.assertEqual(s['sides']['b']['hp'], 28)
        self.assertEqual(len([e for e in s['events'] if e['type'] == 'attack']), 1)
        self.assertEqual(hero(s, 'a', 'anhunqu')['harmony'], 0)
        self.assertEqual(hero(s, 'a', 'anhunqu')['energy'], 2)


if __name__ == '__main__':
    unittest.main()
