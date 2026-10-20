const {test}=require('node:test');
const assert=require('node:assert/strict');
const S=require('../app/modules/guide/static/infographic_state.js');
const data={character_id:'1042',game_version:'1.4.16',characters:[{id:'1042',name:'黑羽'}],arcs:[{id:'arc'}],cores:[{id:'core'}],recommended_arc:'arc',recommended_core:'core'};
test('import requires matching character and validated local IDs',()=>{
  const initial=S.defaults(data),packed=S.pack(initial,data);
  assert.deepEqual(S.unpack(packed,data),initial);
  assert.throws(()=>S.unpack({...packed,character_id:'1057'},data));
  for(const patch of [{radar:[101,0,0,0,0,0]},{modules:{}},{order:['stats']},{theme:'unknown'},{title:'x'.repeat(61)},{art_scale:Infinity}])assert.throws(()=>S.patch(initial,patch,data));
  const wrong=S.clone(initial);wrong.plans[0].arc='unavailable';assert.throws(()=>S.validate(wrong,data));
});
test('patch cannot inject fields; original snapshot is preserved for undo',()=>{
  const initial=S.defaults(data),snapshot=S.clone(initial);
  const changed=S.patch(initial,{orientation:'portrait',rotation:'普通攻击 → 变轨攻击'},data);
  assert.equal(changed.orientation,'portrait');assert.deepEqual(initial,snapshot);
  assert.throws(()=>S.patch(initial,JSON.parse('{"__proto__":{"polluted":true}}'),data));
  assert.equal({}.polluted,undefined);
});
