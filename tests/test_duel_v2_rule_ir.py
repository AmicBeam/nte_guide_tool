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
        cpu, launch = compiled['cpu'], compiled['cuda']
        states, opts = _make_cases(128)
        rows = torch.tensor([_pack(s, o) for s, o in zip(states, opts)], dtype=torch.int32)
        x = rows.clone()
        gpu = rows.cuda()
        out = torch.empty_like(gpu)
        for k in range(4):
            side = k % 2
            x = cpu(x, side)
            launch(gpu, out, side)
            torch.cuda.synchronize()
            gpu = out.clone()
            self.assertTrue(torch.equal(gpu.cpu(), x), msg=f'step {k}')
