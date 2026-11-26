"""Phase-isolated seven-team policy: opening/mulligan vs playing/choice."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, observe, legal_actions
from app.modules.card_game.rl import phase_policy, phase_runtime, phase_checkpoint, seven_runtime, seven_policy
from app.modules.card_game.rl.league_learning import tensors
from app.modules.card_game.rl.seven_lineup import encode

try:
    import torch
except ImportError:
    torch = None

DECKS = {deck['id']: deck for deck in STARTER_DECKS}
ROOT = Path(__file__).resolve().parents[1]


def _pair(key, skip_mulligan):
    decks = {'a': DECKS[key], 'b': DECKS[key]}
    state = new_game(seed=7, first_side='a', skip_mulligan=skip_mulligan, decks=decks)
    actions = [action for action in legal_actions(state, 'a') if action['type'] != 'concede']
    x, c = encode(observe(state, 'a', include_previews=False), actions)
    return actions, x, c


class PhasePolicyTests(unittest.TestCase):
    def test_modules_do_not_import_torch_at_import(self):
        for name in ('phase_policy.py', 'phase_runtime.py', 'phase_checkpoint.py'):
            tree = ast.parse((ROOT / 'app/modules/card_game/rl' / name).read_text(encoding='utf8'))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name == 'torch' or alias.name.startswith('torch.') for alias in node.names))
                if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('torch'):
                    self.fail(name)

    def test_phase_encoding_uses_public_mulligan_and_playing_codes(self):
        _, opening, _ = _pair('lingke', False)
        _, playing, _ = _pair('lingke', True)
        self.assertEqual(phase_policy.phase_of_observation(opening), 'opening')
        self.assertEqual(phase_policy.phase_of_observation(playing), 'playing')
        choice = playing.copy(); choice[phase_policy.PHASE_INDEX] = np.float32(0.2)
        finished = playing.copy(); finished[phase_policy.PHASE_INDEX] = np.float32(0.3)
        self.assertEqual(list(phase_policy.phase_of_observation(np.stack([opening, playing, choice, finished]))),
                         ['opening', 'playing', 'playing', 'playing'])
        with self.assertRaises(ValueError):
            phase_policy.phase_of_observation(np.array([0, 0, 0, 0.9], dtype=np.float32))

    @unittest.skipIf(torch is None, 'optional Torch')
    def test_isolated_branches_and_single_phase_fastpath(self):
        torch.manual_seed(16)
        net = phase_policy.create_network(16, 'cpu')
        self.assertEqual(net.policy_schema, phase_policy.SCHEMA)
        self.assertTrue(net.grounded)
        opening_ids = {id(param) for param in net.policy_parameters('opening')}
        playing_ids = {id(param) for param in net.policy_parameters('playing')}
        value_ids = {id(param) for param in net.value_parameters()}
        self.assertTrue(opening_ids.isdisjoint(playing_ids) and opening_ids.isdisjoint(value_ids) and playing_ids.isdisjoint(value_ids))
        self.assertEqual(opening_ids | playing_ids | value_ids, {id(param) for param in net.parameters()})
        for name, tensor in net.state_dict().items():
            if name.startswith(('state_net.', 'cand_net.', 'score.')):
                self.assertTrue(torch.equal(tensor, net.state_dict()['opening.' + name]), name)
        _, xo, co = _pair('lingke', False)
        _, xp, cp = _pair('lingke', True)
        xo_t, co_t, mo = tensors([(xo, co)], 'cpu')
        xp_t, cp_t, mp = tensors([(xp, cp)], 'cpu')
        with patch.object(net.state_net, 'forward', wraps=net.state_net.forward) as playing:
            with patch.object(net.opening.state_net, 'forward', wraps=net.opening.state_net.forward) as opening:
                net.forward_phase(xo_t, co_t, mo, 'opening')
                opening.assert_called(); playing.assert_not_called()
                opening.reset_mock(); playing.reset_mock()
                net.policy_logits(xp_t, cp_t, mp, 'playing')
                playing.assert_called(); opening.assert_not_called()
                opening.reset_mock(); playing.reset_mock()
                mixed_x, mixed_c, mixed_m = tensors([(xo, co), (xp, cp)], 'cpu')
                scores, wdl = net.forward(mixed_x, mixed_c, mixed_m)
                opening.assert_called(); playing.assert_called()
        fast_o, wdl_o = net.forward_phase(xo_t, co_t, mo, 'opening')
        fast_p, wdl_p = net.forward_phase(xp_t, cp_t, mp, 'playing')
        torch.testing.assert_close(scores[0, :co.shape[0]], fast_o[0, :co.shape[0]])
        torch.testing.assert_close(scores[1, :cp.shape[0]], fast_p[0, :cp.shape[0]])
        torch.testing.assert_close(wdl[0], wdl_o[0]); torch.testing.assert_close(wdl[1], wdl_p[0])
        before = {name: tensor.detach().clone() for name, tensor in net.state_dict().items() if not name.startswith('opening.')}
        net.policy_logits(xo_t, co_t, mo, 'opening')[mo].sum().backward()
        for param in list(net.policy_parameters('playing')) + list(net.value_parameters()):
            self.assertIsNone(param.grad)
        torch.optim.Adam(list(net.policy_parameters('opening')), lr=0.01).step()
        for name, tensor in before.items():
            self.assertTrue(torch.equal(tensor, net.state_dict()[name]), name)


class PhaseRuntimeTests(unittest.TestCase):
    @unittest.skipIf(torch is None, 'optional Torch')
    def test_migration_numpy_torch_greedy_parity_and_tamper(self):
        key = 'lingke'
        actions_o, xo, co = _pair(key, False)
        actions_p, xp, cp = _pair(key, True)
        torch.manual_seed(21)
        source_net = seven_runtime.create_network(key, 16, 'cpu')
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'seven'; target = Path(folder) / 'phase'
            seven_runtime.export(source_net, source, key, DECKS[key], dict(synthetic_fixture=True))
            old = seven_runtime.model(source, key)
            with self.assertRaises(ValueError):
                phase_runtime.migrate(source, source, key)
            net, origin = phase_runtime.migrate(source, target, key)
            self.assertFalse(origin['phase_isolation']['prior_ready_adopted'])
            policy = phase_runtime.PhaseModel(target, key)
            with self.assertRaisesRegex(ValueError, 'new output'):
                phase_runtime.migrate(source, target, key)
            self.assertEqual(policy.schema, phase_policy.SCHEMA)
            self.assertFalse(policy.manifest['validation']['approved'])
            self.assertFalse(policy.manifest['automatic_serving_approval'])
            with self.assertRaises(ValueError):
                policy.validate_human(DECKS[key])
            restored, build, _ = phase_runtime.restore(target, key)
            self.assertEqual(build['card_ids'], sorted(DECKS[key]['card_ids']))
            for name, array in old.weights.items():
                np.testing.assert_array_equal(array, policy.weights[name])
                if name.startswith(('state_net.', 'cand_net.', 'score.')):
                    np.testing.assert_array_equal(array, policy.weights['opening.' + name])
            for phase, x, c, actions in (('opening', xo, co, actions_o), ('playing', xp, cp, actions_p)):
                xx, cc, mask = tensors([(x, c)], 'cpu')
                torch_scores, torch_wdl = restored.forward_phase(xx, cc, mask, phase)
                numpy_scores, numpy_wdl = policy.scores_value(x, c)
                np.testing.assert_allclose(torch_scores.detach().numpy()[0, :len(c)], numpy_scores, rtol=2e-4, atol=2e-4)
                np.testing.assert_allclose(old.scores(x, c), numpy_scores, rtol=2e-4, atol=2e-4)
                self.assertEqual(int(numpy_scores.argmax()), int(torch_scores[0, :len(c)].argmax()))
                self.assertLess(int(numpy_scores.argmax()), len(actions))
                np.testing.assert_allclose([numpy_wdl], torch_wdl.detach().numpy(), rtol=2e-4, atol=2e-4)
                np.testing.assert_allclose(old.wdl(x, True), policy.wdl(x, True), rtol=2e-4, atol=2e-4)
            mixed_x, mixed_c, mixed_m = tensors([(xo, co), (xp, cp)], 'cpu')
            mixed_scores, _ = restored.forward(mixed_x, mixed_c, mixed_m)
            np.testing.assert_allclose(mixed_scores[0, :len(co)].detach().numpy(), policy.scores(xo, co), rtol=2e-4, atol=2e-4)
            np.testing.assert_allclose(mixed_scores[1, :len(cp)].detach().numpy(), policy.scores(xp, cp), rtol=2e-4, atol=2e-4)
            probe = subprocess.check_output([sys.executable, '-c',
                "import sys; sys.modules['torch']=None; from app.modules.card_game.rl.phase_runtime import PhaseModel; "
                "m=PhaseModel(%r, %r); print(m.schema)" % (str(target), key)], cwd=str(ROOT))
            self.assertIn(phase_policy.SCHEMA.encode(), probe)
            manifest = json.loads((target / (key + '.json')).read_text())
            manifest['schema'] = seven_policy.SCHEMA
            (target / (key + '.json')).write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                phase_runtime.PhaseModel(target, key)
            manifest['schema'] = phase_policy.SCHEMA
            manifest['validation'] = {'approved': True}
            (target / (key + '.json')).write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                phase_runtime.PhaseModel(target, key)
            manifest['validation'] = {'approved': False}
            payload = {name: array.copy() for name, array in np.load(target / (key + '.npz'), allow_pickle=False).items()}
            payload['opening.state_net.1.weight'][0, 0] = np.nan
            tmp = target / 'bad.npz'
            np.savez_compressed(tmp, **payload); tmp.replace(target / (key + '.npz'))
            manifest['sha256'] = sha256((target / (key + '.npz')).read_bytes()).hexdigest()
            (target / (key + '.json')).write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                phase_runtime.PhaseModel(target, key)

    @unittest.skipIf(torch is None, 'optional Torch')
    def test_frozen_teacher_is_shared_across_branches(self):
        key = 'starter'
        _, xo, co = _pair(key, False)
        _, xp, cp = _pair(key, True)
        torch.manual_seed(22)
        source_net = seven_runtime.create_network(key, 16, 'cpu', legacy=True)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'seven'; target = Path(folder) / 'phase'
            seven_runtime.export(source_net, source, key, DECKS[key], dict(synthetic_fixture=True))
            old = seven_runtime.model(source, key)
            self.assertIsNotNone(old.teacher)
            net, _ = phase_runtime.migrate(source, target, key)
            policy = phase_runtime.PhaseModel(target, key)
            self.assertIsNotNone(policy.teacher)
            self.assertTrue(hasattr(net, 'legacy'))
            self.assertFalse(any(name.startswith('opening.legacy.') for name in net.state_dict()))
            for x, c in ((xo, co), (xp, cp)):
                np.testing.assert_allclose(policy.scores(x, c), old.scores(x, c), rtol=2e-4, atol=2e-4)
                xx, cc, mask = tensors([(x, c)], 'cpu')
                opening = net.policy_logits(xx, cc, mask, 'opening').detach().numpy()[0, :len(c)]
                playing = net.policy_logits(xx, cc, mask, 'playing').detach().numpy()[0, :len(c)]
                np.testing.assert_allclose(opening, playing, rtol=2e-4, atol=2e-4)
                np.testing.assert_allclose(opening, old.scores(x, c), rtol=2e-4, atol=2e-4)


class PhaseCheckpointTests(unittest.TestCase):
    @unittest.skipIf(torch is None, 'optional Torch')
    def test_restore_matches_next_adam_step_and_rejects_group_tamper(self):
        key = 'lingke'
        _, xo, co = _pair('lingke', False)
        _, xp, cp = _pair('lingke', True)
        xo_t, co_t, mo = tensors([(xo, co)], 'cpu')
        xp_t, cp_t, mp = tensors([(xp, cp)], 'cpu')
        torch.manual_seed(23)
        net = phase_policy.create_network(16, 'cpu')
        opts = phase_checkpoint.create_optimizers(net)

        def step(module, optimizers):
            module.zero_grad(set_to_none=True)
            open_scores, wdl = module.forward_phase(xo_t, co_t, mo, 'opening')
            play_scores, _ = module.forward_phase(xp_t, cp_t, mp, 'playing')
            (open_scores[mo].pow(2).sum() + play_scores[mp].pow(2).sum() + wdl.pow(2).sum()).backward()
            for optimizer in optimizers.values():
                optimizer.step()

        step(net, opts)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / 'model'
            ckpt = Path(folder) / 'restore.pt'
            phase_runtime.export(net, directory, key, DECKS[key], dict(synthetic_fixture=True))
            phase_checkpoint.save(ckpt, net, opts, directory=directory, key=key,
                                  rng_state={'numpy': 23}, metadata={'step': 1})
            live = deepcopy(net.state_dict())
            rng = torch.get_rng_state()
            step(net, opts)
            after = deepcopy(net.state_dict())
            after_opt = {name: deepcopy(opt.state_dict()) for name, opt in opts.items()}
            restored, restored_opts, data = phase_checkpoint.restore(ckpt, directory=directory, key=key)
            self.assertTrue(data['diagnostic_restore_only'])
            self.assertFalse(data['quality_approved'])
            self.assertEqual(data['rng_state'], {'numpy': 23})
            for name, tensor in live.items():
                self.assertTrue(torch.equal(tensor, restored.state_dict()[name]), name)
            torch.set_rng_state(rng)
            step(restored, restored_opts)
            for name, tensor in after.items():
                self.assertTrue(torch.equal(tensor, restored.state_dict()[name]), name)
            for branch in ('opening', 'playing', 'value'):
                for index, value in after_opt[branch]['state'].items():
                    other = restored_opts[branch].state_dict()['state'][index]
                    for field in value:
                        if torch.is_tensor(value[field]):
                            self.assertTrue(torch.equal(value[field], other[field]), branch + ':' + field)
            clean_payload = torch.load(ckpt, map_location='cpu', weights_only=True)
            corrupt = deepcopy(clean_payload)
            moment = next(iter(corrupt['optimizers']['opening']['state'].values()))['exp_avg']
            moment.flatten()[0] = float('nan')
            torch.save(corrupt, ckpt)
            with self.assertRaisesRegex(ValueError, 'Adam'):
                phase_checkpoint.restore(ckpt, directory=directory, key=key)
            torch.save(clean_payload, ckpt)
            payload = torch.load(ckpt, map_location='cpu', weights_only=True)
            payload['optimizer_groups']['opening'], payload['optimizer_groups']['playing'] = (
                list(payload['optimizer_groups']['playing']), list(payload['optimizer_groups']['opening']))
            torch.save(payload, ckpt)
            with self.assertRaises(ValueError):
                phase_checkpoint.restore(ckpt, directory=directory, key=key)
            payload['optimizer_groups']['opening'], payload['optimizer_groups']['playing'] = (
                list(payload['optimizer_groups']['playing']), list(payload['optimizer_groups']['opening']))
            del payload['optimizers']['value']
            torch.save(payload, ckpt)
            with self.assertRaises(ValueError):
                phase_checkpoint.restore(ckpt, directory=directory, key=key)


if __name__ == '__main__':
    unittest.main()
