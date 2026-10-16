"""Formal differentials for supported domain SoA transitions; unknown work rejects."""
import unittest

import torch

from app.modules.card_game.rl.batched_duel.domain_soa_transition import (
    ACTION_ATTACK, ACTION_CONCEDE, ACTION_END_TURN, ACTION_PLAY_CARD, CONT_CHOICE,
    DomainSoAEvaluation, DomainSoARecurrentFn, EVALUATION_ERROR, EVENT_FINISH,
    ILLEGAL_ACTION, INVALID_ROW, INTEGER_OVERFLOW, PreparedDecisionMasks, QUEUE_OVERFLOW, UNKNOWN_HOOK,
    UNSUPPORTED_ACTION, WINNER_ONGOING, bind_columns, coverage, empty_fields, expire_and_compact,
    grant_energy, grant_shield, make_state, select_legal_action, settle_damage, supports_full_duel_rules,
    transition,
)


def ints(values, device='cpu'):
    return torch.tensor(values, dtype=torch.int32, device=device)


def bits(values, device='cpu'):
    return torch.tensor(values, dtype=torch.bool, device=device)


def pack(rows=1, device='cpu', actors=None, finished=None, winners=None, errors=None, **updates):
    fields = empty_fields(rows, device=device)
    for name, value in updates.items():
        fields[name].copy_(torch.as_tensor(value, device=device, dtype=fields[name].dtype))
    actors = torch.zeros(rows, dtype=torch.int64, device=device) if actors is None else actors
    finished = torch.zeros(rows, dtype=torch.bool, device=device) if finished is None else finished
    winners = torch.full((rows,), WINNER_ONGOING, dtype=torch.int64, device=device) if winners is None else winners
    errors = torch.zeros(rows, dtype=torch.int32, device=device) if errors is None else errors
    return make_state(fields, actors, finished, winners, errors)


def combat_masks(**kwargs):
    rows = kwargs['attack_amount'].shape[0]
    device = kwargs['attack_amount'].device
    base = dict(
        attacker_side=torch.zeros(rows, dtype=torch.int32, device=device),
        attacker_seat=torch.zeros(rows, dtype=torch.int32, device=device),
        defender_seat=torch.zeros(rows, dtype=torch.int32, device=device),
        target_is_player=torch.zeros(rows, dtype=torch.bool, device=device),
        counter_amount=torch.zeros(rows, dtype=torch.int32, device=device),
        overflow_amount=torch.zeros(rows, dtype=torch.int32, device=device),
        ap_cost=torch.ones(rows, dtype=torch.int32, device=device),
        bypass=torch.zeros(rows, dtype=torch.bool, device=device),
        defender_immune=torch.zeros(rows, dtype=torch.bool, device=device),
        attacker_immune=torch.zeros(rows, dtype=torch.bool, device=device),
        defender_life_locked=torch.zeros(rows, dtype=torch.bool, device=device),
        attacker_life_locked=torch.zeros(rows, dtype=torch.bool, device=device),
        defender_floor=torch.zeros(rows, dtype=torch.int32, device=device),
        attacker_floor=torch.zeros(rows, dtype=torch.int32, device=device),
        defender_limit=torch.full((rows,), -1, dtype=torch.int32, device=device),
        attacker_limit=torch.full((rows,), -1, dtype=torch.int32, device=device),
    )
    base.update(kwargs)
    return PreparedDecisionMasks(**base)


class DomainSoATransitionTest(unittest.TestCase):
    def test_coverage_does_not_claim_full_duel(self):
        self.assertFalse(supports_full_duel_rules)
        report = coverage()
        self.assertFalse(report.supports_full_duel_rules)
        self.assertTrue(report.remaining_for_complete_search)
        self.assertNotIn(ACTION_PLAY_CARD, report.supported_actions)
        self.assertFalse(report.schema_bound)

    def test_dead_attacker_does_not_grant_surviving_team_energy(self):
        state = pack(1, char_hp=[[[1, 4, 4, 4], [5, 4, 4, 4]]],
                     char_energy=[[[2, 0, 0, 0], [0, 0, 0, 0]]], player_ap=[[2, 2]])
        out = transition(state, torch.tensor([ACTION_ATTACK]), combat_masks(
            attack_amount=ints([1]), counter_amount=ints([8])), bits([True]))
        self.assertEqual(out.errors.tolist(), [0])
        self.assertEqual(out.fields['char_energy'][0, 0].tolist(), [2, 0, 0, 0])

    def test_harmony_is_independent_of_energy_eligibility_and_uses_current_cap(self):
        state = pack(1, turn=13, escalation_enabled=True,
                     char_hp=[[[5, 4, 4, 4], [5, 4, 4, 4]]],
                     char_energy=[[[3, 4, 0, 0], [0, 0, 0, 0]]], player_ap=[[2, 2]])
        can = bits([[[False, True, True, True], [True, True, True, True]]])
        out = transition(state, torch.tensor([ACTION_ATTACK]), combat_masks(
            attack_amount=ints([1]), counter_amount=ints([0]), can_gain_energy=can), bits([True]))
        self.assertEqual(out.errors.tolist(), [0])
        self.assertEqual(out.fields['char_energy'][0, 0, 0].item(), 3)
        self.assertEqual(out.fields['char_energy'][0, 0, 1].item(), 4)
        self.assertEqual(out.fields['char_harmony'][0, 0, 0].item(), 1)

    def test_disabled_escalation_does_not_grant_three_ap_at_turn_twenty(self):
        state = pack(1, turn=19, escalation_enabled=False, front=[[-1, -1]])
        out = transition(state, torch.tensor([ACTION_END_TURN]),
                         PreparedDecisionMasks(clock_blockers=bits([False])), bits([True]))
        self.assertEqual(out.errors.tolist(), [0])
        self.assertEqual(out.fields['player_ap'][0, 1].item(), 2)

    def test_clock_overflow_rolls_back_and_wrong_actor_is_rejected(self):
        state = pack(1, turn=2147483647, front=[[-1, -1]])
        out = transition(state, torch.tensor([ACTION_END_TURN]),
                         PreparedDecisionMasks(clock_blockers=bits([False])), bits([True]))
        self.assertEqual(out.errors.tolist(), [INTEGER_OVERFLOW])
        self.assertEqual(out.fields['turn'].tolist(), [2147483647])
        wrong = pack(1, active_side=1)
        out = transition(wrong, torch.tensor([ACTION_END_TURN]),
                         PreparedDecisionMasks(), bits([True]))
        self.assertEqual(out.errors.tolist(), [INVALID_ROW])
        self.assertEqual(out.fields['active_side'].tolist(), [1])

    def test_bind_columns_rejects_unknown_and_keeps_named_fields(self):
        fields = empty_fields(2)
        bound = bind_columns(fields)
        self.assertEqual(bound['player_hp'].shape, (2, 2))
        with self.assertRaises(ValueError):
            bind_columns({**fields, 'mystery': ints([0, 0])})

    def test_schema_character_columns_bind_as_execution_aliases(self):
        from app.modules.card_game.rl.batched_duel import domain_soa_schema
        fields = empty_fields(1)
        hp = fields.pop('char_hp')
        fields['character_hp'] = hp
        state = make_state(fields, torch.zeros(1, dtype=torch.int64), bits([False]),
                           torch.full((1,), WINNER_ONGOING), torch.zeros(1, dtype=torch.int32))
        self.assertTrue(torch.equal(state.fields['char_hp'], hp))
        self.assertIn('character_hp', state.fields)
        self.assertTrue(domain_soa_schema.ARRAY_NAMES)
        out = transition(state, torch.tensor([ACTION_CONCEDE]), PreparedDecisionMasks(), bits([True]))
        self.assertTrue(out.finished[0])
        self.assertEqual(int(out.fields['character_hp'][0, 0, 0]), int(hp[0, 0, 0]))

    def test_concede_matches_apply_action_and_preserves_inactive(self):
        from app.modules.card_game.engine.duel_v2 import apply_action, new_game
        game = new_game(seed=3, first_side='a', skip_mulligan=True)
        after = apply_action(game, 'a', {'type': 'concede'})
        state = pack(2, player_hp=[[game['sides']['a']['hp'], game['sides']['b']['hp']]] * 2,
                     player_ap=[[game['sides']['a']['ap'], game['sides']['b']['ap']]] * 2,
                     turn=game['turn'])
        out = transition(state, torch.tensor([ACTION_CONCEDE, ACTION_CONCEDE]),
                         PreparedDecisionMasks(), bits([True, False]))
        self.assertEqual(after['winner'], 'b')
        self.assertEqual(after['phase'], 'finished')
        self.assertEqual(out.winners.tolist(), [1, WINNER_ONGOING])
        self.assertEqual(out.finished.tolist(), [True, False])
        self.assertEqual(out.errors.tolist(), [0, 0])
        self.assertEqual(out.fields['phase'].tolist(), [3, 1])
        self.assertTrue(out.fields['event_present'][0].any())
        self.assertEqual(int(out.fields['event_kind'][0, 0]), EVENT_FINISH)
        self.assertFalse(out.fields['event_present'][1].any())

    def test_damage_and_resources_match_formal_primitives(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.state import damage, energy, shield
        game = new_game(skip_mulligan=True)
        h = game['sides']['a']['characters']['zero']
        h.update(hp=5, shield=2, energy=2)
        loss = damage(game, 'a', 'zero', 3, bypass=False, settle=False)
        energy(game, 'a', 'zero', 1)
        shield(game, 'a', 'zero', 3)
        hit = settle_damage(ints([5]), ints([2]), ints([3]), bits([False]), bits([False]),
                            bits([False]), ints([0]), ints([-1]), bits([True]))
        gained = grant_energy(ints([2]), ints([1]), ints([5]), bits([True]), bits([True]))
        shielded = grant_shield(ints([hit.shield.tolist()[0]]), ints([3]), bits([True]), bits([True]))
        self.assertEqual(hit.hp.tolist(), [h['hp']])
        self.assertEqual(hit.loss.tolist(), [loss])
        self.assertEqual(gained.value.tolist(), [game['sides']['a']['characters']['zero']['energy']])
        self.assertEqual(shielded.value.tolist(), [h['shield']])

    def test_ordinary_combat_matches_simultaneous_formal_damage_and_resources(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.escalation import energy_cap, harmony_cap
        from app.modules.card_game.engine.duel_v2.state import damage, energy
        game = new_game(skip_mulligan=True)
        attacker = game['sides']['a']['characters']['zero']
        defender = game['sides']['b']['characters']['zero']
        attacker.update(hp=6, shield=1, energy=1, harmony=0)
        defender.update(hp=5, shield=2, energy=0, harmony=0)
        for cid, hero in game['sides']['a']['characters'].items():
            if cid != 'zero':
                hero.update(hp=4, energy=0)
        damage(game, 'b', 'zero', 4, settle=False)
        damage(game, 'a', 'zero', 2, settle=False)
        expected_def = (defender['hp'], defender['shield'])
        expected_atk = (attacker['hp'], attacker['shield'])
        if game['phase'] != 'finished' and attacker['hp'] > 0:
            for cid, hero in game['sides']['a']['characters'].items():
                if hero['hp'] > 0:
                    energy(game, 'a', cid, 1)
            energy(game, 'a', 'zero', 1)
            attacker['harmony'] = min(harmony_cap(game), attacker['harmony'] + 1)
        state = pack(1, char_hp=[[[6, 4, 4, 4], [5, 10, 10, 10]]],
                     char_shield=[[[1, 0, 0, 0], [2, 0, 0, 0]]],
                     char_energy=[[[1, 0, 0, 0], [0, 0, 0, 0]]],
                     char_harmony=[[[0, 0, 0, 0], [0, 0, 0, 0]]],
                     player_ap=[[2, 2]], turn=game['turn'])
        out = transition(state, torch.tensor([ACTION_ATTACK]), combat_masks(attack_amount=ints([4]),
                         counter_amount=ints([2])), bits([True]))
        self.assertEqual(out.errors.tolist(), [0])
        self.assertEqual(out.fields['char_hp'][0, 1, 0].item(), expected_def[0])
        self.assertEqual(out.fields['char_shield'][0, 1, 0].item(), expected_def[1])
        self.assertEqual(out.fields['char_hp'][0, 0, 0].item(), expected_atk[0])
        self.assertEqual(out.fields['char_shield'][0, 0, 0].item(), expected_atk[1])
        self.assertEqual(out.fields['char_energy'][0, 0, 0].item(),
                         game['sides']['a']['characters']['zero']['energy'])
        self.assertEqual(out.fields['char_harmony'][0, 0, 0].item(),
                         game['sides']['a']['characters']['zero']['harmony'])
        self.assertEqual(out.fields['player_ap'][0, 0].item(), 1)
        self.assertEqual(energy_cap(game, game['sides']['a']['characters']['zero']), 5)

    def test_player_lethal_is_true_terminal_for_fixed_viewer(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.state import damage, victory
        game = new_game(skip_mulligan=True)
        game['sides']['b'].update(hp=3, shield=0)
        damage(game, 'b', 'player', 9, settle=False)
        victory(game)
        state = pack(2, player_hp=[[30, 3], [30, 3]], player_ap=[[2, 2], [2, 2]])
        masks = combat_masks(attack_amount=ints([9, 9]), target_is_player=bits([True, True]),
                             ap_cost=ints([1, 1]))
        out = transition(state, torch.tensor([ACTION_ATTACK, ACTION_ATTACK]), masks, bits([True, False]))
        self.assertEqual(game['winner'], 'a')
        self.assertEqual(out.finished.tolist(), [True, False])
        self.assertEqual(out.winners.tolist(), [0, WINNER_ONGOING])
        self.assertEqual(out.fields['player_hp'][1].tolist(), [30, 3])

        def evaluate(result, viewers, active):
            wdl = torch.zeros((2, 3), dtype=torch.float32)
            wdl[:, 1] = 1
            valid = torch.ones((2, 3), dtype=torch.bool)
            return DomainSoAEvaluation(torch.zeros((2, 3)), valid, wdl, torch.zeros(2, dtype=torch.int32))

        fn = DomainSoARecurrentFn(evaluate, max_actions=3, device='cpu', allow_cpu_reference=True)
        rec = fn(state, torch.tensor([ACTION_ATTACK, ACTION_ATTACK]), torch.tensor([0, 1]),
                 bits([True, False]), masks)
        self.assertEqual(rec.reward.tolist(), [1.0, 0.0])
        self.assertEqual(rec.discount.tolist(), [0.0, 0.0])
        self.assertEqual(rec.valid[0].tolist(), [True, False, False])

    def test_knockdown_keeps_energy_and_clears_front(self):
        state = pack(1, char_hp=[[[1, 4, 4, 4], [1, 4, 4, 4]]],
                     char_shield=[[[2, 0, 0, 0], [3, 0, 0, 0]]],
                     char_energy=[[[4, 0, 0, 0], [0, 0, 0, 0]]],
                     char_harmony=[[[2, 0, 0, 0], [0, 0, 0, 0]]],
                     char_growth=[[[1, 0, 0, 0], [0, 0, 0, 0]]],
                     front=[[0, 0]], player_ap=[[2, 2]])
        out = transition(state, torch.tensor([ACTION_ATTACK]),
                         combat_masks(attack_amount=ints([8]), counter_amount=ints([0])), bits([True]))
        self.assertEqual(out.errors.tolist(), [0])
        self.assertEqual(out.fields['char_hp'][0, 1, 0].item(), 0)
        self.assertEqual(out.fields['char_down_turns'][0, 1, 0].item(), 3)
        self.assertEqual(out.fields['char_shield'][0, 1, 0].item(), 0)
        self.assertEqual(out.fields['char_growth'][0, 1, 0].item(), 0)
        self.assertEqual(out.fields['front'][0, 1].item(), -1)
        self.assertEqual(out.fields['char_energy'][0, 0, 0].item(), 5)

    def test_effect_expiry_matches_tick_and_end_turn_clock(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.engine.duel_v2.escalation import turn_action_points
        from app.modules.card_game.engine.duel_v2.modifiers import add_effect, tick
        game = new_game(skip_mulligan=True)
        h = game['sides']['a']['characters']['zero']
        h['effects'] = []
        records = [('a', 'turn_end', 1), ('b', 'turn_start', 1)]
        for index, (side, phase, turn) in enumerate(records):
            add_effect(h, 'test.' + str(index), 'attack.panel', index,
                       expiry=dict(side=side, phase=phase, turn=turn))
        game['sides']['a']['turn_count'] = 1
        game['sides']['b']['turn_count'] = 0
        expired_end = tick(game, 'a', 'turn_end')
        state = pack(1, turn=1, turn_count=[[1, 0]], active_side=0, front=[[-1, -1]],
                     player_ap=[[2, 0]],
                     effect_present=[[True, True] + [False] * 14],
                     effect_layer=[[0, 0] + [0] * 14],
                     effect_op=[[1, 1] + [0] * 14],
                     effect_amount=[[0, 1] + [0] * 14],
                     effect_expiry_side=[[0, 1] + [0] * 14],
                     effect_expiry_phase=[[1, 0] + [-1] * 14],
                     effect_expiry_turn=[[1, 1] + [0] * 14],
                     effect_owner_side=[[0, 0] + [0] * 14])
        fields, expired, errors = expire_and_compact(dict(state.fields), ints([0]), ints([1]), bits([True]))
        self.assertEqual(errors.tolist(), [0])
        self.assertEqual(expired[0, :2].tolist(), [True, False])
        self.assertEqual(len(expired_end), 1)
        out = transition(state, torch.tensor([ACTION_END_TURN]),
                         PreparedDecisionMasks(can_gain_energy=state.fields['char_present'].clone()),
                         bits([True]))
        self.assertEqual(out.errors.tolist(), [0])
        self.assertEqual(out.fields['turn'].item(), 2)
        self.assertEqual(out.fields['active_side'].item(), 1)
        self.assertEqual(out.fields['turn_count'].tolist(), [[1, 1]])
        self.assertEqual(out.fields['player_ap'][0, 1].item(), turn_action_points({'turn': 2}))
        self.assertFalse(out.fields['effect_present'][0, 0])

    def test_unknown_action_hook_and_illegal_slot_reject_without_mutation(self):
        state = pack(3, player_hp=[[30, 30]] * 3, player_ap=[[2, 2]] * 3)
        saved = state.fields['player_hp'].clone()
        play = transition(state, torch.tensor([ACTION_PLAY_CARD, ACTION_CONCEDE, ACTION_CONCEDE]),
                          PreparedDecisionMasks(hooks=ints([[0], [7], [0]])),
                          bits([True, True, False]))
        self.assertEqual(play.errors.tolist(), [UNSUPPORTED_ACTION, UNKNOWN_HOOK, 0])
        self.assertTrue(torch.equal(play.fields['player_hp'], saved))
        self.assertFalse(play.finished.any())
        legal = bits([[True, False, True]])
        types = ints([[ACTION_CONCEDE, ACTION_PLAY_CARD, ACTION_END_TURN]])
        selected, errors = select_legal_action(legal, types, torch.tensor([1]), bits([True]))
        self.assertEqual(errors.tolist(), [ILLEGAL_ACTION])
        slotted = transition(pack(1, front=[[-1, -1]]), torch.zeros(1, dtype=torch.int64),
                             PreparedDecisionMasks(legal=legal, legal_types=types),
                             bits([True]), slots=torch.tensor([2]))
        self.assertEqual(slotted.errors.tolist(), [0])
        self.assertEqual(slotted.fields['active_side'].item(), 1)

    def test_unknown_continuation_and_queue_overflow_fail_closed(self):
        fields = empty_fields(2)
        fields['event_present'][0] = True
        fields['event_kind'][0] = EVENT_FINISH
        state = make_state(fields, torch.zeros(2, dtype=torch.int64), bits([False, False]),
                           torch.full((2,), WINNER_ONGOING), torch.zeros(2, dtype=torch.int32))
        overflow = transition(state, torch.tensor([ACTION_CONCEDE, ACTION_CONCEDE]),
                              PreparedDecisionMasks(), bits([True, False]))
        self.assertEqual(overflow.errors[0].item(), QUEUE_OVERFLOW)
        self.assertFalse(overflow.finished[0])
        unknown = transition(pack(1), torch.tensor([ACTION_CONCEDE]),
                             PreparedDecisionMasks(continuation_kind=ints([99])), bits([True]))
        self.assertEqual(unknown.errors.tolist(), [UNKNOWN_HOOK])
        self.assertFalse(unknown.finished.any())
        known = transition(pack(1), torch.tensor([ACTION_CONCEDE]),
                           PreparedDecisionMasks(continuation_kind=ints([CONT_CHOICE])), bits([True]))
        self.assertEqual(known.errors.tolist(), [0])
        self.assertTrue(known.fields['cont_present'][0, 0])

    def test_invalid_metadata_and_evaluation_rollback(self):
        state = pack(2, actors=torch.tensor([3, 0]))
        out = transition(state, torch.tensor([ACTION_CONCEDE, ACTION_CONCEDE]),
                         PreparedDecisionMasks(), bits([True, True]))
        self.assertEqual(out.errors.tolist(), [INVALID_ROW, 0])
        self.assertEqual(out.finished.tolist(), [False, True])

        def evaluate(result, viewers, active):
            wdl = torch.tensor([[0.2, 0.3, 0.5], [float('nan'), 0.0, 0.0]], dtype=torch.float32)
            valid = torch.ones((2, 2), dtype=torch.bool)
            return DomainSoAEvaluation(torch.zeros((2, 2)), valid, wdl, torch.zeros(2, dtype=torch.int32))

        live = pack(2, front=[[-1, -1], [-1, -1]])
        fn = DomainSoARecurrentFn(evaluate, max_actions=2, device='cpu', allow_cpu_reference=True)
        rec = fn(live, torch.tensor([ACTION_END_TURN, ACTION_END_TURN]), torch.tensor([0, 0]),
                 bits([True, True]), PreparedDecisionMasks(can_gain_energy=live.fields['char_present'].clone()))
        self.assertEqual(rec.errors[1].item(), EVALUATION_ERROR)
        self.assertEqual(rec.state.fields['turn'][1].item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; CPU parity does not qualify device execution')
    def test_cuda_graph_replay_supported_transition(self):
        device = 'cuda'
        state = pack(2, device=device, player_ap=[[2, 2], [2, 2]], front=[[-1, -1], [-1, -1]])
        actions = torch.tensor([ACTION_CONCEDE, ACTION_END_TURN], device=device)
        active = bits([True, True], device=device)
        masks = PreparedDecisionMasks(can_gain_energy=state.fields['char_present'].clone())
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(2):
                transition(state, actions, masks, active)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = transition(state, actions, masks, active)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(output.finished.cpu().tolist(), [True, False])
        self.assertEqual(output.state.fields['active_side'][1].item(), 1)


if __name__ == '__main__':
    unittest.main()
