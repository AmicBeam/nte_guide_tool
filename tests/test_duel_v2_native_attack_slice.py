"""Canonical-rule differential tests for the VM-free numeric combat sample."""
import ctypes
from copy import deepcopy
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from app.modules.card_game.engine.duel_v2 import new_game, legal_actions
from app.modules.card_game.engine.duel_v2.simulation import simulate_action
from app.modules.card_game.rl.batched_duel.native_attack_slice import (
    ROSTER, FIELDS, LEGAL_WIDTH, UnsupportedNativeAttack, pack_roots,
    decode_legal, emit_source, combat_projection, output_projection,
)


def fixture_cases(count=80):
    """Synthetic legal snapshots, without overriding any registered kit hooks."""
    rng = random.Random(208)
    base = new_game(seed=208, skip_mulligan=True)
    out = []
    for i in range(count):
        s = deepcopy(base)
        s['active_side'] = ('a', 'b')[i % 2]
        s['turn'] = (1, 5, 6, 12, 13, 20)[i % 6]
        s['escalation_enabled'] = bool(i % 3)
        for side in ('a', 'b'):
            t = s['sides'][side]
            t['deck'].extend(t['hand'])
            t['hand'] = []
            t['ap'] = rng.randrange(4)
            t['normal_attack_available'] = bool(rng.randrange(2))
            t['ultimate_available'] = bool(rng.randrange(2))
            t['ultimates_used'] = rng.randrange(3)
            t['front'] = rng.choice((None,) + ROSTER)
            t['last_front'] = rng.choice((None,) + ROSTER) if t['front'] is None else None
            t['hp'] = rng.randrange(1, 31)
            t['shield'] = rng.randrange(5)
            t['nanali_family'] = rng.randrange(4)
            for cid in ROSTER:
                h = t['characters'][cid]
                h['hp'] = rng.randrange(1, h['max_hp'] + 1)
                h['shield'] = rng.randrange(5)
                h['energy'] = rng.randrange(6)
                h['harmony'] = rng.randrange(3)
                h['ultimate_used_turn'] = s['turn'] if rng.randrange(3) == 0 else -1
        # Ensure coverage includes successful battles, not mostly no-op roots.
        if i % 2 == 0:
            t = s['sides'][s['active_side']]
            t['normal_attack_available'], t['ap'] = True, 2
        out.append(s)
    return out


def boundary_cases():
    base = fixture_cases(1)[0]
    cases = []
    for actor in ('a', 'b'):
        for a in range(4):
            for d in range(-1, 4):
                for hp in (1, 5):
                    s = deepcopy(base)
                    s['active_side'] = actor
                    enemy = 'b' if actor == 'a' else 'a'
                    s['sides'][actor].update(front=ROSTER[a], ap=2,
                        normal_attack_available=True, last_front=None)
                    s['sides'][enemy].update(front=None if d < 0 else ROSTER[d],
                                            last_front=None, hp=1, shield=0)
                    for side in ('a', 'b'):
                        s['sides'][side]['nanali_family'] = 0
                        for h in s['sides'][side]['characters'].values():
                            h.update(hp=min(hp, h['max_hp']), shield=0, harmony=0, energy=5)
                    cases.append((s, a))
    return cases


class NativeSliceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if compiler is None:
            raise unittest.SkipTest('Native C++ compiler unavailable')
        cls.tmp = tempfile.TemporaryDirectory(prefix='nte-native-attack-')
        root = Path(cls.tmp.name)
        (root / 'slice.cpp').write_text(emit_source())
        target = root / 'slice.so'
        subprocess.run([compiler, '-std=c++17', '-O2', '-shared', '-fPIC',
                        str(root / 'slice.cpp'), '-o', str(target)],
                       check=True, capture_output=True, timeout=30)
        cls.lib = ctypes.CDLL(str(target))
        cls.fn = cls.lib.native_attack_slice
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_int32)] * 5 + [ctypes.c_int]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_native(self, states, actions):
        x = pack_roots(states)
        y = np.empty_like(x)
        legal = np.empty((LEGAL_WIDTH, len(states)), dtype=np.int32)
        status = np.empty(len(states), dtype=np.int32)
        a = np.asarray(actions, dtype=np.int32)
        self.fn(*(v.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
                  for v in (x, y, legal, status, a)), len(states))
        self.assertEqual(x.shape[0], FIELDS)
        return x, y, legal, status

    def test_canonical_legal_order_and_combat_fields(self):
        states = fixture_cases(160)
        before = deepcopy(states)
        counts = {0: 0, 1: 0, 2: 0}
        for action in range(4):
            x, y, legal, status = self.run_native(states, [action] * len(states))
            for row, s in enumerate(states):
                side = s['active_side']
                canonical = legal_actions(s, side)
                self.assertEqual(decode_legal(legal[:, row]), canonical)
                a = {'type': 'attack', 'character_id': ROSTER[action]}
                counts[int(status[row])] += 1
                if status[row]:
                    self.assertTrue(np.array_equal(y[:, row], x[:, row]))
                    if status[row] == 1:
                        self.assertNotIn(a, canonical)
                    else:
                        self.assertIn(a, canonical)
                        from app.modules.card_game.engine.duel_v2.combat import harmony_eligible
                        self.assertTrue(harmony_eligible(s, side, ROSTER[action]))
                    continue
                expected = simulate_action(s, side, a)
                self.assertEqual(output_projection(y[:, row]), combat_projection(expected),
                                 (row, side, action))
        self.assertGreater(counts[0], 100)
        self.assertGreater(counts[1], 100)
        self.assertGreater(counts[2], 20)
        self.assertEqual(states, before)

    def test_mutual_knockdown_rout_and_terminal_resource_boundary(self):
        # All attacker/defender combinations: shield blocks, simultaneous deaths,
        # Nanali's attack/counter rout and capped energy.
        cases = boundary_cases()
        x, y, _, status = self.run_native([s for s, _ in cases], [a for _, a in cases])
        self.assertTrue(np.all(status == 0))
        for row, (s, a) in enumerate(cases):
            expected = simulate_action(s, s['active_side'], {'type': 'attack', 'character_id': ROSTER[a]})
            self.assertEqual(output_projection(y[:, row]), combat_projection(expected), (row, a))

    def test_downed_teammate_no_share_and_legality_only_no_mutation(self):
        s = fixture_cases(1)[0]
        s['sides']['a'].update(front='jiuyuan', ap=2, normal_attack_available=True)
        s['sides']['a']['characters']['iloy'].update(hp=0, down_turns=2, energy=3)
        s['sides']['b']['characters']['zero'].update(hp=5, shield=4)
        x, y, legal, status = self.run_native([s], [-1])
        self.assertEqual(status.tolist(), [0])
        self.assertTrue(np.array_equal(x, y))
        self.assertEqual(decode_legal(legal[:, 0]), legal_actions(s, 'a'))
        _, y, _, status = self.run_native([s], [3])
        self.assertEqual(status.tolist(), [0])
        expected = simulate_action(s, 'a', {'type': 'attack', 'character_id': 'jiuyuan'})
        self.assertEqual(output_projection(y[:, 0]), combat_projection(expected))
        self.assertEqual(expected['sides']['a']['characters']['iloy']['energy'], 3)

    def test_unsupported_effects_fail_before_packing_and_input_stays_pure(self):
        base = fixture_cases(1)[0]
        for modify in (
            lambda s: s.update(tutorial={'disable_energy_gain': True}),
            lambda s: s['sides']['a']['characters']['iloy']['flags'].update(next_bonus=1),
            lambda s: s['sides']['a']['characters']['nanali'].update(awakened=True),
            lambda s: s['sides']['b']['characters']['zero'].update(shape='Z08'),
            lambda s: s['sides']['a'].update(effects=[{'property': 'rule.normal_attack_instant'}]),
            lambda s: s['sides']['b'].update(life_locked=True),
        ):
            s = deepcopy(base)
            modify(s)
            before = deepcopy(s)
            with self.assertRaises(UnsupportedNativeAttack):
                pack_roots([base, s])
            self.assertEqual(s, before)

    def test_device_rejects_malformed_numeric_addresses_atomically(self):
        x = pack_roots(fixture_cases(3))
        x[2, 0] = 2       # invalid side must never address character arrays
        x[12, 1] = 99     # invalid front slot
        x[27, 2] = -1    # invalid character life
        y = np.empty_like(x)
        legal = np.empty((LEGAL_WIDTH, 3), dtype=np.int32)
        status = np.empty(3, dtype=np.int32)
        actions = np.zeros(3, dtype=np.int32)
        self.fn(*(v.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
                  for v in (x, y, legal, status, actions)), 3)
        self.assertEqual(status.tolist(), [4, 4, 4])
        self.assertTrue(np.array_equal(y, x))
        self.assertTrue(np.all(legal == -1))


if __name__ == '__main__':
    unittest.main()
