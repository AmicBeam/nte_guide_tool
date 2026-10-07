import json
from pathlib import Path
import tempfile
import time
import unittest
from copy import deepcopy
import numpy as np
from app.modules.card_game.rl import fixed_lineup as f
from app.modules.card_game.rl import league_schema as old
from app.modules.card_game.rl.league_observation import encode_league
from app.modules.card_game.rl.league_policy import LeagueModel, shapes
from app.modules.card_game.rl.fixed_lineup_training import prepare, play, summary
from app.modules.card_game.engine.duel_v2 import new_game, observe

SOURCE = Path(__file__).resolve().parents[1] / 'app/modules/card_game/engine/ai/models'


class FixedLineupsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        cls.source = cls.root / 'source'; cls.source.mkdir()
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        for i, key in enumerate(old.PRESETS):
            build = next(d for d in STARTER_DECKS if d['id'] == key)
            rng = np.random.default_rng(i)
            weights = {n: rng.normal(0, .05, shape).astype(np.float32) for n, shape in shapes(4).items()}
            path = cls.source / f'{key}.npz'; np.savez_compressed(path, **weights)
            import hashlib
            metadata = dict(schema=old.SCHEMA, deck=key, features=old.feature_names(),
                seat_ids=list(old.SEATS), card_ids=list(old.CARD_IDS), cand_dim=old.CAND_DIM,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), build=build,
                build_sha256=old.build_hash(build), hidden=4, rule_hash=old.rule_hash())
            (cls.source / f'{key}.json').write_text(json.dumps(metadata))
        cls.builds = prepare(cls.source, cls.root)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_five_fixed_builds_and_exact_source_cards(self):
        self.assertEqual(set(self.builds), set(f.KEYS))
        self.assertEqual(self.builds['quick-rush']['character_ids'], ['xiaozhi', 'zero', 'bohe', 'jiuyuan'])
        self.assertEqual(self.builds['midrange']['character_ids'], ['nanali', 'baicang', 'zero', 'iloy'])
        for key in ('starter', 'weave-rush'):
            source = json.loads((self.source / f'{key}.json').read_text())
            self.assertEqual(self.builds[key]['card_ids'], source['build']['card_ids'])

    def test_old_logits_preserved_in_old_supported_observation(self):
        weights, metadata = f.expand_source(self.source, 'starter')
        state = new_game(seed=52, decks={'a': self.builds['starter'], 'b': self.builds['weave-rush']})
        view = observe(state, 'a', include_previews=False)
        actions = [e['action'] for e in view['legal_actions'] if e['action']['type'] != 'concede']
        x, c = encode_league(view, actions); xx, cc = f.encode(view, actions)
        with np.load(self.source / 'starter.npz', allow_pickle=False) as data:
            source = LeagueModel.__new__(LeagueModel); source.weights = {k: data[k].copy() for k in data.files}; source.hidden = metadata['hidden']
        expanded = f.FixedModel(self.root / 'reference', 'starter')
        np.testing.assert_allclose(source.scores(x, c), expanded.scores(xx, cc), rtol=1e-5, atol=1e-5)
        self.assertAlmostEqual(source.scores_value(x, c)[1], expanded.scores_value(xx, cc)[1], places=5)
        self.assertEqual(len(set(f.all_features())), len(f.all_features()))
        self.assertEqual(len(set(f.candidate_names())), f.CAND_DIM)
        # Existing weights and shifted candidate option/mark columns map by name.
        names = f.candidate_names()
        with np.load(self.source / 'starter.npz', allow_pickle=False) as data:
            for i, name in enumerate(f.candidate_names(old.CARD_IDS)):
                np.testing.assert_array_equal(weights['cand_net.0.weight'][:, names.index(name)], data['cand_net.0.weight'][:, i])

    def test_new_resources_and_genesis_visible(self):
        state = new_game(seed=7, decks={'a': self.builds['zhenhong'], 'b': self.builds['starter']})
        state['sides']['a']['characters']['yi']['beast_fangs'] = 4
        state['sides']['a']['characters']['zhenhong']['extra_attacks'] = 6
        state['sides']['b']['genesis_player_hits'] = 9
        view = observe(state, 'a', include_previews=False)
        actions = [e['action'] for e in view['legal_actions'] if e['action']['type'] != 'concede']
        x, _ = f.encode(view, actions); names = f.all_features()
        self.assertAlmostEqual(x[names.index('fixed:beast_fangs:0:yi')], .4)
        self.assertAlmostEqual(x[names.index('fixed:extra_attacks:0:zhenhong')], .6)
        self.assertAlmostEqual(x[names.index('fixed:genesis_player_hits:1')], .9)

    def test_unknown_hands_and_deck_order_are_not_encoded(self):
        state = new_game(seed=9, decks={'a': self.builds['zhenhong'], 'b': self.builds['quick-rush']})
        before = observe(state, 'a', include_previews=False)
        state['sides']['b']['hand'][0], state['sides']['b']['deck'][0] = state['sides']['b']['deck'][0], state['sides']['b']['hand'][0]
        state['sides']['a']['deck'].reverse()
        after = observe(state, 'a', include_previews=False)
        actions = [e['action'] for e in before['legal_actions'] if e['action']['type'] != 'concede']
        for a, b in zip(f.encode(before, actions), f.encode(after, actions)): np.testing.assert_array_equal(a, b)

    def test_transformed_hand_encoded_as_rf01(self):
        state = new_game(seed=9, decks={'a': self.builds['zhenhong'], 'b': self.builds['starter']})
        state['sides']['a']['characters']['zhenhong']['awakened'] = True
        view = observe(state, 'a', include_previews=False)
        actions = [e['action'] for e in view['legal_actions'] if e['action']['type'] != 'concede']
        x, c = f.encode(view, actions)
        self.assertTrue(all(card['card_id'] == 'RF01' for card in view['sides']['a']['hand']))
        self.assertEqual(x[f.all_features().index('cards:0:RF01')], 2.5)
        self.assertEqual(c.shape[1], f.CAND_DIM)

    def test_real_new_teams_finish_and_opponents_stay_unchanged(self):
        before = {p.name: p.read_bytes() for p in (self.root / 'reference').iterdir()}
        for i, key in enumerate(('quick-rush', 'zhenhong', 'midrange')):
            result = play(dict(seed=73 + i, first='a', deadline=time.time() + 60,
                decks={'a': self.builds[key], 'b': self.builds['starter']},
                policies={'a': (str(self.root / 'initial'), key), 'b': (str(self.root / 'reference'), 'starter')}))
            self.assertTrue(result['complete'], result)
            self.assertIn(result['winner'], ('a', 'b', 'draw'))
            self.assertEqual(result['trajectory'], [])
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.root / 'reference').iterdir()})

    def test_legacy_fixed_migration_preserves_numeric_columns(self):
        from app.modules.card_game.rl.fixed_lineup_training import migrate_fixed
        source = self.root / 'legacy-fixed'; source.mkdir(exist_ok=True)
        current = f.FixedModel(self.root / 'initial', 'zhenhong')
        names = [n for n in f.all_features() if not n.startswith('fixed:damage_immune:')]
        weights = {n: v.copy() for n,v in current.weights.items()}
        indices = [f.all_features().index(n) for n in names]
        weights['state_net.0.weight'] = weights['state_net.0.weight'][:,indices]
        p=source/'zhenhong.npz';np.savez_compressed(p,**weights)
        import hashlib
        metadata = dict(current.manifest, features=names, rule_hash='previous-rule', sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        (source/'zhenhong.json').write_text(json.dumps(metadata))
        output=self.root/'migrated'
        migrate_fixed(source,'zhenhong',output,self.builds['zhenhong'])
        migrated=f.FixedModel(output,'zhenhong')
        np.testing.assert_array_equal(migrated.weights['state_net.0.weight'][:,indices],weights['state_net.0.weight'])
        extra=[i for i,n in enumerate(f.all_features()) if n.startswith('fixed:damage_immune:')]
        self.assertFalse(migrated.weights['state_net.0.weight'][:,extra].any())
        self.assertFalse(migrated.manifest['validation']['approved'])
        self.assertEqual(migrated.manifest['origin']['source_rule_hash'],'previous-rule')

    def test_legacy_full_immunity_column_remains_visible_in_fixed_observation(self):
        state=new_game(seed=17,decks={'a':self.builds['zhenhong'],'b':self.builds['starter']})
        hero=state['sides']['a']['characters']['zhenhong']
        # Preserve the old fixed-model column; live Zhenhong uses damage_limit=1,
        # covered by test_duel_v2_recovery_observation, not this historical field.
        hero['flags']['damage_immunity_until']=state['sides']['a']['turn_count']+1
        view=observe(state,'a',include_previews=False)
        actions=[e['action'] for e in view['legal_actions'] if e['action']['type']!='concede']
        x,_=f.encode(view,actions)
        self.assertAlmostEqual(x[f.all_features().index('fixed:damage_immune:0:zhenhong')],.1)

    def test_missing_eval_never_complete(self):
        self.assertTrue(all(not row['complete'] for row in summary([], 32).values()))


if __name__ == '__main__': unittest.main()
