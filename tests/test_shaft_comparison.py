import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

from flask import Flask

from app.modules.shaft import blueprint
from app.modules.shaft.domain.catalog import load_shaft_catalog
from tests.test_solo_room_flow import RoomFlowTestCase

ROOT = Path(__file__).resolve().parents[1]


class ShaftComparisonTest(unittest.TestCase):
    def test_comparison_page_is_public_and_assets_are_available(self):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        client = app.test_client()
        self.assertEqual(client.get('/shaft/compare?axis=1').status_code, 404)
        self.assertEqual(client.get('/shaft/axes/1/compare').status_code, 404)
        response = client.get('/shaft/axes/1/compare?character=example&guide_character=1042&locked=1')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('window.location.replace', response.text)
        self.assertIn('data-axis-id="1"', response.text)
        self.assertIn('id="compare-matrices"', response.text)
        self.assertIn('id="compare-detail"', response.text)
        self.assertNotIn('compare-draft', response.text)
        self.assertNotIn('compare-axis"', response.text)
        for asset in ['js/shaft_compare_page.js', 'js/shaft_comparison.js',
                      'js/shaft_comparison_worker.js', 'css/shaft-compare.css']:
            with client.get('/static/shaft/' + asset) as response:
                self.assertEqual(response.status_code, 200, asset)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for numerical comparison regression')
    def test_comparison_preserves_inputs_and_matches_direct_engine_recalculation(self):
        script = r'''
const assert = require('assert/strict');
const fs = require('fs');
const engine = require('./app/modules/shaft/static/js/shaft_engine.js');
const comparison = require('./app/modules/shaft/static/js/shaft_comparison.js');
const checks = require('./app/modules/shaft/static/js/shaft_self_check.js');
const catalog = JSON.parse(fs.readFileSync(0,'utf8'));
const clone = x => JSON.parse(JSON.stringify(x));
const axis = clone(catalog.starter_axis);
const member = axis.team[0];
const snapshot = engine.buildSnapshot(member, catalog, axis.team_panel_bonus);
const selection = {character_id: member.character_id, arc_ids:[member.arc_id],
  cartridge_ids:[member.cartridge_id], main_stats:[snapshot.build_options.cartridge_main_stat],
  refinement:snapshot.arc_refinement};
const original = JSON.stringify(axis);
const messages = [...comparison.compare(axis, catalog, selection, engine, checks)];
assert.equal(messages[1].row.baseline_percent, 100, 'Original equipment must be exactly 100% of baseline');
assert.equal(messages[1].row.increase_percent, 0, 'Original equipment must have exactly zero difference');
assert.deepEqual(messages[1].row.warnings, messages[0].baseline.warnings);
assert.equal(JSON.stringify(axis), original, 'The source axis and its build mirror must not be mutated');

const arc = catalog.arcs.find(a => a.adaptation === snapshot.character.adaptation && a.id !== member.arc_id);
const changed = {...selection, arc_ids:[arc.id], main_stats:['暴伤'], refinement:1};
const output = [...comparison.compare(axis, catalog, changed, engine, checks)];
const candidate = clone(axis);
Object.assign(candidate.team[0], {arc_id:arc.id, arc_refinement:1, cartridge_main_stat:'暴伤'});
const expected = engine.simulateAxis(candidate, catalog);
assert.equal(output[1].row.total_damage, expected.summary.total_damage);
assert.equal(output[1].row.baseline_percent, expected.summary.total_damage / output[0].baseline.total_damage * 100);
assert.equal(output[1].row.own_damage, expected.damage_by_slot.find(m => m.slot === member.slot).damage);
assert.equal(output[1].row.harmony_damage, expected.summary.harmony_damage);
assert.equal(output[1].row.stagger_damage, expected.summary.stagger_damage);
assert.deepEqual(axis.team.slice(1), candidate.team.slice(1));
assert.equal(JSON.stringify(axis), original);

const wrongType = catalog.arcs.find(a => a.adaptation !== snapshot.character.adaptation);
assert.throws(() => [...comparison.compare(axis,catalog,{...selection,arc_ids:[wrongType.id]},engine,checks)], /适应类型/);
assert.throws(() => [...comparison.compare(axis,catalog,{...selection,main_stats:['invalid']},engine,checks)], /未知主词条/);
assert.throws(() => [...comparison.compare(axis,catalog,{...selection,arc_ids:[]},engine,checks)], /至少选择/);
assert.throws(() => [...comparison.compare(axis,catalog,{...selection,arc_ids:['']},engine,checks)], /适应类型/);
assert.throws(() => [...comparison.compare(axis,catalog,{...selection,cartridge_ids:['']},engine,checks)], /未知卡带/);
assert.throws(() => [...comparison.compare(axis,catalog,{...selection,refinement:6},engine,checks)], /精炼档位/);
assert.throws(() => [...comparison.compare(axis,catalog,{...selection,character_id:'absent'},engine,checks)], /没有这个角色/);
const restricted = clone(catalog);
restricted.characters.find(c => c.id === member.character_id).selection_disabled = true;
assert.throws(() => [...comparison.compare(axis,restricted,selection,engine,checks)], /当前账号/);
const tooMany = {...selection,arc_ids:catalog.arcs.filter(a=>a.adaptation===snapshot.character.adaptation).map(a=>a.id),
  cartridge_ids:catalog.cartridges.map(c=>c.id),main_stats:Object.keys(catalog.formula_constants.cartridge_main_stat_options)};
assert.throws(() => [...comparison.compare(axis,catalog,tooMany,engine,checks)], /600/);

const unequipped = clone(axis);
unequipped.team[0].arc_id = '';
unequipped.team[0].cartridge_id = '';
if (unequipped.character_builds?.[member.character_id]) {
  unequipped.character_builds[member.character_id].arc_id = '';
  unequipped.character_builds[member.character_id].cartridge_id = '';
}
const equippedCandidates = [...comparison.compare(unequipped,catalog,selection,engine,checks)];
assert.equal(equippedCandidates[0].baseline_options.arc_id,'');
assert.equal(equippedCandidates[0].baseline_options.cartridge_id,'');
assert.equal(equippedCandidates[1].row.arc_id, member.arc_id);
assert.ok(Number.isFinite(equippedCandidates[1].row.baseline_percent));

const empty = clone(axis);
empty.steps = [{id:'noop',slot:member.slot,action_id:'action_none_'+member.character_id.replace('char_',''),start_tick:0}];
empty.team = [member];
const zero = [...comparison.compare(empty,catalog,selection,engine,checks)];
assert.equal(zero[0].baseline.total_damage,0);
assert.equal(zero[1].row.baseline_percent,null);
assert.equal(zero[1].row.increase_percent,null, 'A zero baseline must not produce Infinity or a fabricated percent');
console.log('comparison invariants passed');
'''
        result = subprocess.run(['node', '-e', script], cwd=ROOT,
                                input=json.dumps(load_shaft_catalog()), text=True,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ShaftRelatedAxisTest(RoomFlowTestCase):
    def test_market_pagination_has_no_missing_axis_between_pages(self):
        import importlib
        from datetime import datetime, timedelta
        models = importlib.import_module('app.models')
        player = models.Player.create(player_uid='wiki-pagination', nickname='Pagination')
        created = []
        for index in range(7):
            axis = models.ShaftAxis.create(owner=player, title=f'Axis {index}', visibility='public',
                                          published_at=datetime(2026, 1, 1) + timedelta(seconds=index))
            created.append(axis.id)
        pages = [self._get(f'/api/shaft/market?sort=new&page_size=2&page={page}') for page in range(1, 5)]
        actual = [axis['id'] for page in pages for axis in page['items']]
        self.assertEqual(actual, list(reversed(created)))
        self.assertEqual([page['has_more'] for page in pages], [True, True, True, False])

    def test_anonymous_preview_does_not_bypass_private_axis_access(self):
        import importlib
        models = importlib.import_module('app.models')
        player = models.Player.create(player_uid='wiki-preview', nickname='Preview')
        private = models.ShaftAxis.create(owner=player, title='Private preview', visibility='private')
        public = models.ShaftAxis.create(owner=player, title='Public preview', visibility='public')
        with patch.dict(self.app.config, {'SHAFT_LOGIN_REQUIRED': True}):
            private_page = self.client.get(f'/shaft/rotation?preview={private.id}').text
            public_page = self.client.get(f'/shaft/rotation?preview={public.id}').text
        self.assertEqual(self.client.get('/shaft/compare').status_code, 404)
        self.assertIn('window.location.replace', private_page)
        self.assertNotIn('window.location.replace', public_page)
        embedded = self.client.get(f'/shaft/rotation?preview={public.id}&embedded=1').text
        self.assertIn('data-preview-only="true"', embedded)
        self.assertIn('shaft-preview-embedded.css', embedded)
        self.assertNotIn('href="/shaft/compare"', embedded)
        self.assertEqual(self.client.get(f'/api/shaft/axes/{private.id}').status_code, 400)
        self.assertEqual(self.client.get(f'/api/shaft/axes/{public.id}').status_code, 200)


if __name__ == '__main__':
    unittest.main()
