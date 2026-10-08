"""Combat-slice IR: one program, CPU and CUDA backends, Python engine as oracle."""
from __future__ import annotations

import copy
import importlib.util
import random
import unittest

TORCH = importlib.util.find_spec('torch')


def _pack(state, opts):
    from app.modules.card_game.engine.duel_v2.state import attack_value
    a, b = state['sides']['a'], state['sides']['b']
    ca, cb = a['characters']['zero'], b['characters']['zero']
    return [
        ca['hp'], ca['shield'], attack_value(ca),
        cb['hp'], cb['shield'], attack_value(cb),
        a['hp'], a['shield'], b['hp'], b['shield'],
        int(a['front'] == 'zero'), int(b['front'] == 'zero'),
        ca['down_turns'], cb['down_turns'],
        int(state['phase'] == 'finished'),
        {'a': 0, 'b': 1, 'draw': 2}.get(state.get('winner'), -1),
        *opts,
    ]


def _reference(state, opts, side):
    from app.modules.card_game.engine.duel_v2.combat import attack
    from app.modules.card_game.engine.duel_v2.context import EffectContext
    side_s = 'ab'[side]
    hero = state['sides'][side_s]['characters']['zero']
    if state['phase'] == 'finished' or hero['hp'] <= 0 or state['sides'][side_s]['front'] != 'zero':
        return
    op = dict(side=side_s, actor='zero', participants=[], first_bonus=0, harmony=None)
    attack(EffectContext(state, side_s, 'zero', op), dict(
        no_counter=bool(opts[0]), penetrate_low_hp=bool(opts[1]), low_hp_max=100, bonus=opts[2],
    ), 0)


def _make_cases(n, seed=714):
    from app.modules.card_game.engine import duel_v2 as eng
    template = eng.new_game(seed=7)
    rng = random.Random(seed)
    states, opts = [], []
    for i in range(n):
        state = copy.deepcopy(template)
        state.update(phase='playing', active_side='a', winner=None, reason='', events=[], event_seq=0)
        for side in 'ab':
            team = state['sides'][side]
            team['hand'] = []
            team['front_debuff'] = {}
            team['front'] = 'zero' if rng.random() < 0.85 else None
            team['hp'] = rng.randint(1, 12)
            team['shield'] = rng.randint(0, 5)
            for hero in team['characters'].values():
                hero.update(shape=None, awakened=False, ultimate_turns=0, flags={}, growth=0, gaze=0, dream=0)
            z = team['characters']['zero']
            z.update(hp=rng.randint(1, 5), shield=rng.randint(0, 5), base_attack=rng.randint(0, 7))
        if i % 17 == 0:
            state.update(phase='finished', winner='a', reason='player_hp')
            state['sides']['b']['hp'] = 0
        states.append(state)
        opts.append((int(rng.random() < 0.3), int(rng.random() < 0.5), rng.randint(0, 3)))
    return states, opts


@unittest.skipUnless(TORCH, 'PyTorch is required for rule IR')
class CombatSliceIrTest(unittest.TestCase):
    def test_ir_rejects_unknown_ops(self):
        from app.modules.card_game.rl.rule_ir.ast import Bind, Const, Name, Program, Store, validate
        from app.modules.card_game.rl.rule_ir.combat_slice import COMBAT_SLICE
        validate(COMBAT_SLICE)
        bad = Program(
            name='bad', version='x',
            skip_when=Const(0),
            binds=(Bind('z', Name('nope')),),
            stores=(Store('done', Const(1)),),
        )
        with self.assertRaises(ValueError):
            validate(bad)

    def test_cpu_matches_python_engine(self):
        import torch
        from app.modules.card_game.rl.rule_ir import compile_program
        step = compile_program()['cpu']
        states, opts = _make_cases(64)
        x = torch.tensor([_pack(s, o) for s, o in zip(states, opts)], dtype=torch.int32)
        for k in range(4):
            side = k % 2
            for state, opt in zip(states, opts):
                _reference(state, opt, side)
            x = step(x, side)
            expected = torch.tensor([_pack(s, o) for s, o in zip(states, opts)], dtype=torch.int32)
            self.assertTrue(torch.equal(x, expected), msg=f'step {k}')

    def test_cuda_matches_cpu_if_available(self):
        import torch
        from app.modules.card_game.rl.rule_ir import compile_program
        if not torch.cuda.is_available():
            self.skipTest('CUDA not available')
        compiled = compile_program(cuda=True)
        states, opts = _make_cases(128)
        rows = torch.tensor([_pack(s, o) for s, o in zip(states, opts)], dtype=torch.int32)
        expected = compiled['cpu_steps'](rows, 4)
        gpu = rows.cuda()
        out = torch.empty_like(gpu)
        compiled['cuda'](gpu, out, 4)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out.cpu(), expected))

    def test_generated_cuda_matches_handwritten_and_throughput(self):
        import statistics
        import time
        import torch
        from app.modules.card_game.rl.rule_ir import compile_program
        from app.modules.card_game.rl.rule_ir.handwritten_combat import HANDWRITTEN_FN, HANDWRITTEN_SOURCE
        from app.modules.card_game.rl.rule_ir.lower_cuda import compile_cuda_source
        if not torch.cuda.is_available():
            self.skipTest('CUDA not available')
        compiled = compile_program(cuda=True)
        hand = compile_cuda_source(HANDWRITTEN_SOURCE, HANDWRITTEN_FN)
        states, opts = _make_cases(256)
        rows = torch.tensor([_pack(s, o) for s, o in zip(states, opts)], dtype=torch.int32)
        gpu = rows.cuda()
        gen_out = torch.empty_like(gpu)
        hand_out = torch.empty_like(gpu)
        compiled['cuda'](gpu, gen_out, 4)
        hand(gpu, hand_out, 4)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(gen_out, hand_out))
        self.assertTrue(torch.equal(gen_out.cpu(), compiled['cpu_steps'](rows, 4)))

        def median_ms(launch, n, repeats=80):
            data = rows.cuda().repeat((n + rows.shape[0] - 1) // rows.shape[0], 1)[:n]
            out = torch.empty_like(data)
            for _ in range(5):
                launch(data, out, 4)
            torch.cuda.synchronize()
            samples = []
            for _ in range(3):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(repeats):
                    launch(data, out, 4)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) / repeats)
            return statistics.median(samples) * 1000

        gen_ms = median_ms(compiled['cuda'], 4096)
        hand_ms = median_ms(hand, 4096)
        ratio = hand_ms / gen_ms if gen_ms else 0
        # 80% is the report's engineering target. Sub-millisecond times are noisy under load.
        if hand_ms >= 0.05:
            self.assertGreaterEqual(ratio, 0.8, msg=f'generated {gen_ms:.4f}ms vs handwritten {hand_ms:.4f}ms ({ratio:.2f}x)')


class StarterEffectIrTest(unittest.TestCase):
    def test_starter_ir_covers_appearing_cards(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECK
        from app.modules.card_game.rl.rule_ir.starter import STARTER_APPEARING, STARTER_CARDS, validate_starter_ir  # noqa: PLC0415
        validate_starter_ir()
        appearing = set(STARTER_APPEARING)
        self.assertTrue(set(STARTER_DECK['card_ids']) <= appearing)
        self.assertIn('NF01', appearing)
        self.assertEqual({program.card_id for program in STARTER_CARDS}, appearing)

    def test_effect_tables_match_catalog_index(self):
        from app.modules.card_game.rl.gpu_duel.catalog import CARD_IDS, CARD_INDEX
        from app.modules.card_game.rl.rule_ir.lower_effects import compile_effect_tables
        tables = compile_effect_tables(CARD_IDS)
        self.assertEqual(tables['sortie'][CARD_INDEX['N03']], 1)
        self.assertEqual(tables['followup'][CARD_INDEX['N03']], 1)
        self.assertEqual(tables['form'][CARD_INDEX['N07']], 1)
        self.assertEqual(tables['draw_owned_target'][CARD_INDEX['Y02']], 1)
        self.assertEqual(tables['inspect_top'][CARD_INDEX['J01']], 3)
        self.assertEqual(tables['pact_front'][CARD_INDEX['J02']], 1)
        self.assertEqual(tables['count_seed'][CARD_INDEX['NF01']], 1)
        self.assertEqual(tables['perm_atk'][CARD_INDEX['Y05']], 1)
        self.assertEqual(tables['perm_hp'][CARD_INDEX['Y05']], 1)
        self.assertEqual(tables['hooks']['turn_end_atk'], 1)
        self.assertEqual(tables['sortie'][CARD_INDEX['M01']], 1)
        self.assertEqual(tables['perm_atk'][CARD_INDEX['M01']], 1)
        self.assertEqual(tables['ignore_shield'][CARD_INDEX['M03']], 1)
        self.assertEqual(tables['form'][CARD_INDEX['M08']], 1)
        self.assertEqual(tables['self_damage'][CARD_INDEX['B08']], 4)
        self.assertEqual(tables['sortie_allied_extra'][CARD_INDEX['B06']], 1)
