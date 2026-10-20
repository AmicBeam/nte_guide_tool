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

test('editable radar labels and board positions survive config roundtrip; legacy drafts migrate',()=>{
  const c=S.defaults(data);c.radar_labels=['输出','治疗','生存','破韧','增益','操作'];c.positions={stats:{x:400,y:260}};
  assert.deepEqual(S.unpack(S.pack(c,data),data),c);
  const old=S.pack(c,data);delete old.config.radar_labels;delete old.config.positions;
  const restored=S.unpack(old,data);assert.equal(restored.radar_labels.length,6);assert.deepEqual(restored.positions,{});
  for(const changes of [{radar_labels:['A']},{radar_labels:['','a','b','c','d','e']},{positions:{'bad-key':{x:1,y:2}}},{positions:{stats:{x:NaN,y:4}}},{positions:{stats:{x:1,y:-1}}}])assert.throws(()=>S.patch(c,changes,data));
});

test('canvas scene shares moved text bounds and renders custom radar names and identity icons',async()=>{
  global.window={};global.Image=class {set src(v){this.url=v;this.width=256;this.height=256;queueMicrotask(()=>this.onload());}};
  require('../app/modules/guide/static/infographic_canvas.js');
  const text=[],images=[];
  const ctx={measureText:v=>({width:String(v).length*11}),fillText:v=>text.push(v),drawImage:img=>images.push(img.url)};
  for(const name of ['beginPath','roundRect','fill','rect','clip','save','restore','moveTo','lineTo','closePath','stroke','fillRect'])ctx[name]=()=>{};
  const canvas={getContext:()=>ctx};
  const d={...data,characters:[{id:'1042',name:'黑羽',assets:{fullbody:'portrait',avatar:'avatar'},element_icon:'dark-icon',rarity_icon:'rank-icon',tags:[]}],modules:Object.fromEntries(S.modules.map(k=>[k,k])),radar_labels:['爆发','持续','倾陷','生存','辅助','机动']};
  const c=S.defaults(d);c.radar_labels[0]='自由维度';c.stats='自填数值';c.positions.stats={x:600,y:900};
  const result=await window.GuideInfographicCanvas.render(canvas,c,d);
  const board=result.scene.boards.find(b=>b.key==='stats'),field=result.scene.fields.find(f=>f.path==='stats');
  assert.equal(board.x,600);assert.equal(board.y,900);assert.equal(field.x,board.x+26);assert.ok(field.y>board.y);
  assert.ok(text.includes('自由维度'));assert.ok(images.includes('dark-icon'));assert.ok(images.includes('rank-icon'));
  assert.ok(result.scene.fields.every(f=>!f.path.includes('.arc')&&!f.path.includes('.core')&&!f.path.includes('.members')));
});

test('candidate lists use the full roster and migrate old single-member drafts',()=>{
  const d={...data,characters:[...data.characters,...Array.from({length:12},(_,i)=>({id:String(i),name:`候选${i}`}))]};
  const c=S.defaults(d);c.teams[0].candidates[0]=d.characters.slice(1).map(c=>c.id);
  assert.equal(S.validate(c,d).teams[0].candidates[0].length,12);
  assert.deepEqual(S.unpack(S.pack(c,d),d),c);
  const duplicate=S.clone(c);duplicate.teams[0].candidates[0].push('1042');assert.throws(()=>S.validate(duplicate,d));
  const legacy=S.pack(c,d);for(const t of legacy.config.teams){delete t.candidates;delete t.folded;}
  const restored=S.unpack(legacy,d);assert.deepEqual(restored.teams[0].members,c.teams[0].members);assert.deepEqual(restored.teams[0].candidates,[[],[],[],[]]);
});
test('candidate groups stay in their slot, render every candidate, and fold compactly',async()=>{
  const d={...data,characters:[{id:'1042',name:'黑羽',assets:{avatar:'main',fullbody:'portrait'},tags:[]},...Array.from({length:5},(_,i)=>({id:String(i),name:`候选${i}`,assets:{avatar:`candidate${i}`}}))],modules:Object.fromEntries(S.modules.map(k=>[k,k]))};
  const c=S.defaults(d);c.teams[0].candidates[2]=d.characters.slice(1).map(c=>c.id);
  const drawn=[];const ctx={measureText:v=>({width:String(v).length*11}),fillText:()=>{},drawImage:img=>drawn.push(img.url)};
  for(const name of ['beginPath','roundRect','fill','rect','clip','save','restore','moveTo','lineTo','closePath','stroke','fillRect'])ctx[name]=()=>{};
  const render=()=>window.GuideInfographicCanvas.render({getContext:()=>ctx},c,d);
  const folded=await render();assert.equal(folded.scene.slots[0].path,'teams.0.folded.2');
  for(let i=0;i<5;i++)assert.ok(drawn.includes(`candidate${i}`));
  c.teams[0].folded[2]=false;const expanded=await render();
  assert.ok(expanded.scene.boards.find(b=>b.key==='teams').h>folded.scene.boards.find(b=>b.key==='teams').h);
});
test('JS declarations generate bounded parameters without executing code and travel with templates',()=>{
  const P=require('../app/modules/guide/static/infographic_script.js');
  const script='/* @editor {"builtin":[],"controls":[{"key":"title","label":"标题","type":"text","default":"新图"},{"key":"size","label":"尺寸","type":"number","default":20,"min":10,"max":40}]} */\nglobalThis.shouldNotExecute=true;';
  const c=S.defaults(data);c.script=script;c.script_params=P.defaults(script);assert.equal(global.shouldNotExecute,undefined);
  assert.deepEqual(S.unpack(S.pack(c,data),data),c);assert.equal(c.script_params.size,20);
  const old=S.pack(c,data);delete old.config.script;delete old.config.script_params;
  assert.ok(S.unpack(old,data).script.includes('@editor'));
  for(const source of [script.replace('"size"','"constructor"'),script.replace('"min":10','"min":50'),script.replace('"controls"','"unknown"'),'while(true){}'])assert.throws(()=>P.declaration(source));
  assert.throws(()=>S.validate({...c,script_params:{title:'x',size:999}},data));
});
