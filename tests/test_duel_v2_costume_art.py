"""Equipment artwork follows each public entity; it must never replace default card art."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CostumeArtTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js required')
    def test_equipment_replacement_replay_and_default_fallback(self):
        script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const window = {localStorage: {getItem:()=>''}, matchMedia:()=>({matches:false})};
const sandbox = {window, console, document:{getElementById:()=>null}, requestAnimationFrame:()=>{}};
vm.createContext(sandbox);
for (const path of ['v2_api.js','v2_table/helpers.js','v2_table/render.js']) {
  vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/'+path,'utf8'),sandbox);
}
const api=window.NTE_V2, table=window.NTE_V2_TABLE;
const original={id:'nanali',name:'娜娜莉',portrait:'/default.webp',avatar:'/avatar.webp',hp:5,max_hp:5};
const equipped={...original,shape_id:'N08'};
assert.match(api.battlePortrait(equipped), /Fashion_1010_nighty.webp/);
assert.equal(api.characterAsset(equipped,'portrait'),'/default.webp');
assert.equal(api.battlePortrait(original),'/default.webp'); // Same template on the other side.
assert.match(api.battlePortrait({...equipped,shape_id:'N07'}),/Fashion_1010_3.webp/);
for (const delta of [{shape_id:null},{shape_id:'unknown'},{shape_id:'J08'},{down_turns:3},{summoned:true}]) {
  assert.equal(api.battlePortrait({...equipped,...delta}),'/default.webp');
}
// Reconstructed historical public state and current state use identical selection.
assert.equal(api.battlePortrait(JSON.parse(JSON.stringify(equipped))),api.battlePortrait(equipped));
assert.equal(api.battlePortrait({...original,shape:'预备备'}),'/default.webp'); // No guessed ID in old replays.
assert.match(api.battlePortrait({id:'anhunqu',shape_id:'A07'}),/Fashion_1004_1.webp/);
assert.match(api.battlePortrait({id:'lingke',shape_id:'K07'}),/Fashion_1072_school.webp/);
assert.match(api.characterAsset({avatar:'/static/images/characters/avatar/灵可.png'}), /灵可.png\?nte_art=/);
assert.equal(api.characterAsset({avatar:'/static/kongmu/images/characters/player_yiluoyi_256.webp'}), '/static/kongmu/images/characters/player_yiluoyi_256.webp');
// Real renderer: updates the same node on equip/change/down, preserving fallback on asset failure.
const attrs={};let image;
const node={style:{},classList:{toggle(){},remove(){}},setAttribute(k,v){attrs[k]=v},
 getAttribute(k){return attrs[k]},removeAttribute(k){delete attrs[k]},querySelectorAll(){return []},
 querySelector(){return image},set innerHTML(value){this.html=value;image={src:'',onerror:null}},get innerHTML(){return this.html}};
for (const [shape,expected] of [['N08','Fashion_1010_nighty.webp'],['N07','Fashion_1010_3.webp'],[null,'/default.webp']]) {
 table.fillCharacterCard(node,{...original,shape_id:shape},{side:'a',usePortrait:true});
 assert.ok(node.innerHTML.includes(expected));
 if (shape) {image.onerror();assert.equal(image.src,'/default.webp');assert.equal(image.onerror,null);}
}
const hand=table.handCardHtml({character_id:'nanali',name:'测试',type:'form',id:'N08'},{legal:true});
assert.ok(!hand.includes('Fashion_'));
assert.ok(!table.handCardHtml({character_id:'nanali',hidden:true}).includes('Fashion_'));
// Every selected fashion belongs to its card owner and exists; also covers test-only characters.
const catalog=JSON.parse(fs.readFileSync('app/modules/card_game/content/duel_v2/catalog.json'));
const manifest=JSON.parse(fs.readFileSync('docs/character-image-assets-manifest.json'));
for (const asset of manifest.assets.filter(a=>a.purpose==='costume')) {
 for (const id of asset.card_ids) {
  const card=catalog.cards.find(c=>c.id===id);
  assert.equal(card.type,'form');
  const owner=catalog.characters.find(c=>c.id===card.character_id);
  assert.equal(String(owner.everness_id),asset.character_id);
  assert.ok(api.battlePortrait({id:owner.id,shape_id:id}).includes(asset.fashion_id+'.webp'));
 }
}
'''
        result = subprocess.run(['node', '-e', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_imported_asset_integrity(self):
        manifest = json.loads((ROOT / 'docs/character-image-assets-manifest.json').read_text())
        for asset in manifest['assets']:
            with self.subTest(path=asset['output']):
                content = (ROOT / asset['output']).read_bytes()
                self.assertEqual(content[:4], b'RIFF')
                self.assertEqual(content[8:12], b'WEBP')
                self.assertEqual(hashlib.sha256(content).hexdigest(), asset['output_sha256'])
                for alias in asset['aliases']:
                    data = (ROOT / alias).read_bytes()
                    self.assertTrue(data.startswith(b'\x89PNG') if alias.endswith('.png') else data == content)
