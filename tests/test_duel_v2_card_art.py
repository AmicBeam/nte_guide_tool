"""New character card art comes from public API data, not a JS ID allowlist."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node.js required')
class CardArtTest(unittest.TestCase):
    def test_catalog_live_and_replay_register_new_character_art(self):
        script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
let payload;
const window = {localStorage: {getItem: () => '', removeItem: () => {}}, location: {}};
const sandbox = {window, console, fetch: async () => ({ok: true, status: 200,
  headers: {get: () => ''}, json: async () => payload})};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_api.js', 'utf8'), sandbox);
vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_table/render.js', 'utf8'), sandbox);
const api = window.NTE_V2;
const table = window.NTE_V2_TABLE;
const card = id => ({character_id: id, instance_id: 'test-' + id, name: '测试牌', type: 'battle'});
(async () => {
  // Previously unknown IDs require no frontend edits.
  assert.equal(api.characterFallback('future_character'), null);
  payload = {characters: [{id: 'future_character', name: '未来角色', portrait: '/future.webp'}]};
  await api.getCatalog();
  assert.ok(table.handCardHtml(card('future_character'), {legal:true}).includes('/future.webp'));
  // Table can open directly, without first requesting a catalog.
  payload = {game: {sides: {a: {characters: [{id: 'xiaozhi', name: '小吱',
    portrait: '/static/images/characters/portrait/小吱.webp', hp: 1}]}}}};
  await api.getState();
  assert.ok(table.handCardHtml(card('xiaozhi'), {phase:'mulligan'}).includes('小吱.webp'));
  assert.equal(api.characterFallback('xiaozhi').hp, undefined);
  // A partial update must not erase an already known image.
  payload = {game: {sides: {a: {characters: [{id:'xiaozhi', portrait:'', hp:0}]}}}};
  await api.getState();
  assert.ok(api.characterAsset({id:'xiaozhi'}, 'portrait').endsWith('小吱.webp'));
  // Archived replay opening metadata also registers without catalog access.
  payload = {replay: {opening_board: {sides: {b: {characters: [{id:'haiyue', name:'海月',
    portrait:'/static/images/characters/portrait/海月.webp'}]}}}}};
  await api.getReplay('test');
  assert.ok(table.handCardHtml(card('haiyue'), {legal:true}).includes('海月.webp'));
  // Hidden cards must remain plain backs, even with a known owner.
  const hidden = table.handCardHtml({...card('haiyue'), hidden:true});
  assert.ok(!hidden.includes('海月.webp'));
  // Server metadata overrides an old fallback path for existing IDs too.
  payload = {characters: [{id:'zero', portrait:'/current-zero.webp'}]};
  await api.getCatalog();
  assert.equal(api.characterAsset({id:'zero'}, 'portrait'), '/current-zero.webp');
  assert.ok(api.characterFallback('zero').avatar);
})().catch(error => {console.error(error); process.exitCode=1;});
'''
        result = subprocess.run(['node', '-e', script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
